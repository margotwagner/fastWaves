#!/usr/bin/env python3
"""Diagnose where Wave/FastWave fails on the eight-arm bump task.

Outputs
-------
probe_results.csv
    Ridge linear-probe performance for the eight-bit visited-arm set from
    x, v, [x,v], F, Fq, and the projected fast-drive field.

ablation_metrics.csv
    Autonomous-rollout metrics for baseline and causal interventions at the
    first choice onset.

example_states.npz
    Full time-resolved states for a small example batch. This includes the
    complete F tensor and Fq at every timestep for FastWave.

static_weights.png
    Input projection, depthwise wave kernels, and output readout weights.

This script imports src/models_diagnostics.py, so the training src/models.py does not need to be changed while a sweep is running.
Existing checkpoints remain compatible because the model parameters are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyze import eight_arm_traj_metrics  # noqa: E402
from src.tasks import build_dataset  # noqa: E402
from src.models_diagnostics import WaveRNN, LocalFastWaveRNN  # noqa: E402




def build_diagnostic_model(args):
    common = dict(
        input_dim=args.n_space,
        n_space=args.n_space,
        output_dim=args.n_space,
        channels=args.channels,
        kernel_size=args.kernel_size,
        dt=args.dt,
        omega=args.omega,
        damping=args.damping,
        readout_state=getattr(args, "wave_readout", "x"),
    )
    if args.model == "wave":
        return WaveRNN(**common)
    if args.model == "fastwave":
        return LocalFastWaveRNN(
            **common,
            patch_size=args.patch_size,
            lam=args.lam,
            eta=args.eta,
            beta=args.beta,
            fast_update=args.fast_update,
        )
    raise ValueError("Use a Wave or FastWave checkpoint")


def safe_torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def model_state_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in ckpt:
            return ckpt[key]
    raise KeyError("Checkpoint has no model_state/model_state_dict/state_dict key")


def diagnostic_times_and_labels(dataset) -> tuple[list[int], list[str], torch.Tensor]:
    """Return shared probe times and visited-set labels [B,K,8]."""
    if not hasattr(dataset, "valid_choice_masks"):
        raise ValueError("This diagnostic requires EightArmBumpTrajectoryDataset")

    B = len(dataset)
    n_arms = int(dataset.n_arms)
    selection = dataset.valid_choice_masks.sum(dim=-1) > 0.5
    times0 = torch.where(selection[0])[0]
    if len(times0) != int(dataset.n_choice):
        raise RuntimeError("Unexpected number of choice events")
    for b in range(1, B):
        if not torch.equal(torch.where(selection[b])[0], times0):
            raise RuntimeError("Choice event times differ across trials")

    forced_end = int(dataset.n_forced * dataset.visit_len - 1)
    after_settle = int(dataset.first_choice_frame - 1)
    choice_times = [int(t) for t in times0.tolist()]

    times = [forced_end, after_settle] + choice_times
    names = ["end_forced", "after_settle"] + [
        f"before_choice_{j + 1}" for j in range(len(choice_times))
    ]

    labels = torch.zeros(B, len(times), n_arms, dtype=torch.float32)
    forced = torch.zeros(B, n_arms, dtype=torch.float32)
    forced.scatter_(1, dataset.forced_orders.long(), 1.0)
    labels[:, 0] = forced
    labels[:, 1] = forced

    for j, t in enumerate(choice_times):
        # At a selection event valid_choice_masks marks all still-unvisited arms.
        labels[:, 2 + j] = 1.0 - dataset.valid_choice_masks[:, t, :]

    return times, names, labels


def flatten_feature(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().reshape(x.shape[0], -1).numpy()


def collect_probe_features(
    model: torch.nn.Module,
    dataset,
    times: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return feature arrays shaped [B,K,D]."""
    chunks: dict[str, list[np.ndarray]] = {}
    model.eval()

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            x = dataset.x[start:stop].to(device)
            _, extras = model(x, record_times=times)

            x_post = extras["wave_state"][:, times]
            v_post = extras["wave_velocity"][:, times]
            features: dict[str, torch.Tensor] = {
                "x": x_post,
                "v": v_post,
                "xv": torch.cat([x_post, v_post], dim=-2),
            }

            if "fast_memory_read" in extras:
                features.update(
                    {
                        "F": extras["fast_memory_read"],
                        "Fq": extras["fast_retrieved"],
                        "fast_drive": extras["fast_drive_raw"],
                        "q": extras["fast_query"],
                    }
                )

            for name, tensor in features.items():
                arr = tensor.detach().float().cpu().reshape(
                    tensor.shape[0], tensor.shape[1], -1
                ).numpy()
                chunks.setdefault(name, []).append(arr)

    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def ridge_probe(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int,
    train_fraction: float = 0.7,
    alpha: float = 10.0,
) -> dict[str, float]:
    """Dual-form ridge probe for an eight-bit target."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_train = max(2, int(round(train_fraction * len(X))))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    if len(test_idx) < 1:
        raise ValueError("Need more probe samples")

    Xtr = X[train_idx].astype(np.float64, copy=False)
    Xte = X[test_idx].astype(np.float64, copy=False)
    Ytr = Y[train_idx].astype(np.float64, copy=False)
    Yte = Y[test_idx].astype(np.float64, copy=False)

    x_mean = Xtr.mean(axis=0, keepdims=True)
    x_std = Xtr.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    Xtr = (Xtr - x_mean) / x_std
    Xte = (Xte - x_mean) / x_std

    y_mean = Ytr.mean(axis=0, keepdims=True)
    Ytr_centered = Ytr - y_mean

    # Dual ridge is efficient when F is high-dimensional: invert n_train x n_train.
    gram = Xtr @ Xtr.T
    dual = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=gram.dtype),
        Ytr_centered,
    )
    scores = Xte @ (Xtr.T @ dual) + y_mean
    pred = scores >= 0.5
    truth = Yte >= 0.5

    per_arm_acc = (pred == truth).mean(axis=0)
    hamming_acc = float((pred == truth).mean())
    exact_set_acc = float(np.all(pred == truth, axis=1).mean())

    # A simple continuous score diagnostic. Values near 1 mean strong linear fit.
    ss_res = ((Yte - scores) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    valid = ss_tot > 1e-12
    mean_r2 = float(np.mean(1.0 - ss_res[valid] / ss_tot[valid])) if valid.any() else float("nan")

    return {
        "hamming_accuracy": hamming_acc,
        "exact_set_accuracy": exact_set_acc,
        "mean_per_arm_accuracy": float(per_arm_acc.mean()),
        "min_per_arm_accuracy": float(per_arm_acc.min()),
        "mean_r2": mean_r2,
        "n_train": float(len(train_idx)),
        "n_test": float(len(test_idx)),
        "n_features": float(X.shape[1]),
    }


def write_probe_results(
    path: Path,
    features: dict[str, np.ndarray],
    labels: torch.Tensor,
    time_names: list[str],
    seed: int,
    alpha: float,
) -> None:
    rows: list[dict[str, Any]] = []
    Y = labels.numpy()
    for representation, array in features.items():
        for k, time_name in enumerate(time_names):
            metrics = ridge_probe(array[:, k], Y[:, k], seed=seed + k, alpha=alpha)
            rows.append(
                {
                    "representation": representation,
                    "time": time_name,
                    **metrics,
                }
            )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def slice_dataset(dataset, n: int):
    """Shallow-copy a generated dataset and slice trial-indexed tensors."""
    out = copy.copy(dataset)
    original_n = len(dataset)
    for name, value in vars(dataset).items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == original_n:
            setattr(out, name, value[:n].clone())
    out.n_samples = n
    return out


def autonomous_rollout_with_intervention(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    intervention: dict[str, Any] | None,
) -> torch.Tensor:
    """One-pass autoregressive rollout using the model's recurrent step API."""
    x = dataset.x.to(device)
    prefix_len = int(dataset.rollout_prefix_len)
    B, T, _ = x.shape
    preds: list[torch.Tensor] = []

    schedule = dataset.valid_choice_masks.to(device).sum(dim=-1) > 0.5
    a0 = int(dataset.arm_choice_start)
    a1 = int(dataset.arm_choice_end)
    state = model.initial_state(B, x)
    next_input = None

    model.eval()
    with torch.no_grad():
        for t in range(T):
            if t < prefix_len:
                u_t = x[:, t]
            else:
                if next_input is None:
                    raise RuntimeError("Autoregressive input was not initialized")
                u_t = next_input

            pred_t, state, _ = model.step(
                u_t, state, t=t, intervention=intervention
            )
            pred_t = pred_t.clone()

            active = torch.where(schedule[:, t])[0]
            if len(active) > 0:
                choices = pred_t[active, a0:a1].argmax(dim=-1)
                pred_t[active, a0:a1] = 0.0
                pred_t[active, a0 + choices] = 1.0

            preds.append(pred_t)
            next_input = pred_t

    return torch.stack(preds, dim=1)


def run_ablation_suite(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    out_path: Path,
) -> None:
    start_t = int(dataset.first_choice_frame)
    is_fastwave = hasattr(model, "fast_to_site")

    conditions: dict[str, dict[str, Any] | None] = {
        "baseline": None,
        "zero_velocity_once": {
            "start_t": start_t,
            "zero_velocity_once": True,
        },
    }
    if is_fastwave:
        conditions.update(
            {
                "erase_fast_from_choice": {
                    "start_t": start_t,
                    "erase_fast_once": True,
                    "disable_fast_write_after": True,
                },
                "shuffle_fast_at_choice": {
                    "start_t": start_t,
                    "shuffle_fast_once": True,
                    "disable_fast_write_after": True,
                },
                "disable_fast_drive_from_choice": {
                    "start_t": start_t,
                    "disable_fast_drive_after": True,
                },
            }
        )

    rows: list[dict[str, Any]] = []
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    for condition, intervention in conditions.items():
        yhat = autonomous_rollout_with_intervention(
            model=model,
            dataset=dataset,
            device=device,
            intervention=intervention,
        )
        metrics = eight_arm_traj_metrics(
            yhat=yhat,
            x=x,
            y=y,
            dataset=dataset,
            prefix="eightarm_rollout",
            rollout_mode=True,
        )
        for metric, value in metrics.items():
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "value": value,
                }
            )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def save_example_states(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    n_examples: int,
    out_path: Path,
) -> None:
    n = min(n_examples, len(dataset))
    x = dataset.x[:n].to(device)
    model.eval()
    with torch.no_grad():
        yhat, extras = model(x, record_all=True)

    arrays: dict[str, np.ndarray] = {
        "input": dataset.x[:n].numpy(),
        "target": dataset.y[:n].numpy(),
        "prediction": yhat.detach().cpu().numpy(),
        "forced_orders": dataset.forced_orders[:n].numpy(),
        "choice_orders": dataset.choice_orders[:n].numpy(),
        "valid_choice_masks": dataset.valid_choice_masks[:n].numpy(),
    }
    for key, value in extras.items():
        if torch.is_tensor(value):
            arrays[key] = value.detach().cpu().numpy()
    np.savez_compressed(out_path, **arrays)


def plot_static_weights(model: torch.nn.Module, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    inp = model.input_proj.weight.detach().cpu().numpy()
    axes[0].imshow(inp, aspect="auto")
    axes[0].set_title("Input projection")
    axes[0].set_xlabel("Task input feature")
    axes[0].set_ylabel("Latent channel × ring site")

    kernel = model.wave_conv.conv.weight.detach().cpu().numpy()[:, 0, :]
    for c in range(kernel.shape[0]):
        axes[1].plot(kernel[c], marker="o", label=f"channel {c}")
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_title("Circular depthwise wave kernels")
    axes[1].set_xlabel("Kernel offset")
    axes[1].set_ylabel("Weight")
    if kernel.shape[0] <= 8:
        axes[1].legend(fontsize=7)

    out = model.out.weight.detach().cpu().numpy()
    axes[2].imshow(out, aspect="auto")
    axes[2].set_title("Output readout")
    axes[2].set_xlabel("Latent readout feature")
    axes[2].set_ylabel("Task output feature")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--ablation-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--example-trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260716)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--out-dir", default="auto")
    p.add_argument("--cpu-threads", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu" or not torch.cuda.is_available():
        torch.set_num_threads(max(1, int(args.cpu_threads)))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = safe_torch_load(args.ckpt, device)
    train_args = SimpleNamespace(**ckpt["args"])
    train_args.device = str(device)

    if train_args.model not in {"wave", "fastwave"}:
        raise ValueError("Use this script with a Wave or FastWave checkpoint")
    if train_args.task != "eight_arm_bump_traj":
        raise ValueError("Use this script with the eight_arm_bump_traj task")

    model = build_diagnostic_model(train_args).to(device)
    model.load_state_dict(model_state_from_checkpoint(ckpt))

    dataset = build_dataset(train_args, n_samples=args.n_samples, seed=args.seed)
    times, time_names, labels = diagnostic_times_and_labels(dataset)

    out_dir = (
        Path(args.ckpt).resolve().parent / "state_diagnostics"
        if args.out_dir == "auto"
        else Path(args.out_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    features = collect_probe_features(
        model=model,
        dataset=dataset,
        times=times,
        batch_size=args.batch_size,
        device=device,
    )
    write_probe_results(
        path=out_dir / "probe_results.csv",
        features=features,
        labels=labels,
        time_names=time_names,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )

    ablation_ds = slice_dataset(dataset, min(args.ablation_samples, len(dataset)))
    run_ablation_suite(
        model=model,
        dataset=ablation_ds,
        device=device,
        out_path=out_dir / "ablation_metrics.csv",
    )

    save_example_states(
        model=model,
        dataset=dataset,
        device=device,
        n_examples=args.example_trials,
        out_path=out_dir / "example_states.npz",
    )
    plot_static_weights(model, out_dir / "static_weights.png")

    with (out_dir / "diagnostic_times.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "timestep"])
        writer.writerows(zip(time_names, times))

    print(f"Saved diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
