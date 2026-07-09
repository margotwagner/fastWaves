import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import (
    VanillaRNN,
    WaveRNN,
    GlobalFastRNN,
    LocalFastRNN,
    LocalFastWaveRNN,
)
from src.tasks import build_dataset


def build_model(args):
    if args.model == "vanilla":
        return VanillaRNN(args.n_space, args.hidden_dim, args.n_space)
    if args.model == "globalfast":
        return GlobalFastRNN(
            input_dim=args.n_space,
            hidden_dim=args.hidden_dim,
            output_dim=args.n_space,
            lam=args.lam,
            eta=args.eta,
            beta=args.beta,
            fast_update=args.fast_update,
        )
    if args.model == "wave":
        return WaveRNN(
            input_dim=args.n_space,
            n_space=args.n_space,
            output_dim=args.n_space,
            channels=args.channels,
            kernel_size=args.kernel_size,
            dt=args.dt,
            omega=args.omega,
            damping=args.damping,
        )
    if args.model == "localfast":
        return LocalFastRNN(
            input_dim=args.n_space,
            n_space=args.n_space,
            output_dim=args.n_space,
            channels=args.channels,
            kernel_size=args.kernel_size,
            patch_size=args.patch_size,
            lam=args.lam,
            eta=args.eta,
            beta=args.beta,
            fast_update=args.fast_update,
        )
    if args.model == "fastwave":
        return LocalFastWaveRNN(
            input_dim=args.n_space,
            n_space=args.n_space,
            output_dim=args.n_space,
            channels=args.channels,
            kernel_size=args.kernel_size,
            patch_size=args.patch_size,
            dt=args.dt,
            omega=args.omega,
            damping=args.damping,
            lam=args.lam,
            eta=args.eta,
            beta=args.beta,
            fast_update=args.fast_update,
        )
    raise ValueError(f"Unknown model: {args.model}")



def unpack_batch(batch, device):
    """Support datasets returning (x, y) or (x, y, weights)."""
    if len(batch) == 3:
        x, y, weights = batch
        return x.to(device), y.to(device), weights.to(device)
    if len(batch) == 2:
        x, y = batch
        return x.to(device), y.to(device), None
    raise ValueError(f"Expected batch length 2 or 3, got {len(batch)}")


def sequence_mse_loss(yhat, y, weights=None):
    """
    MSE with optional weights.

    weights may be [B, T] timestep weights or [B, T, N] elementwise weights.
    """
    sq = (yhat - y) ** 2
    if weights is None:
        return sq.mean()
    if weights.ndim == 2:
        loss_per_t = sq.mean(dim=-1)
        return (loss_per_t * weights).sum() / weights.sum().clamp_min(1e-8)
    if weights.ndim == 3:
        return (sq * weights).sum() / weights.sum().clamp_min(1e-8)
    raise ValueError(f"Expected weights ndim 2 or 3, got {weights.ndim}")

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    train_ds = build_dataset(args, n_samples=args.n_train, seed=args.seed)
    val_ds = build_dataset(args, n_samples=args.n_val, seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    run_name = args.run_name if args.run_name is not None else args.model
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y, weights = unpack_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            yhat, extras = model(x)
            loss = sequence_mse_loss(yhat, y, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y, weights = unpack_batch(batch, device)
                yhat, extras = model(x)
                val_loss += sequence_mse_loss(yhat, y, weights).item() * x.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args)},
                out_dir / "best.pt",
            )

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if "fast_weight_norm" in extras:
            row["extras/fast_weight_norm_mean"] = (
                extras["fast_weight_norm"].mean().item()
            )
        if "fast_drive_norm" in extras:
            row["extras/fast_drive_norm_mean"] = extras["fast_drive_norm"].mean().item()
        history.append(row)

        if epoch == 1 or epoch % args.print_every == 0:
            msg = f"epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f}"
            if "fast_weight_norm" in extras:
                msg += f" | F_norm {extras['fast_weight_norm'].mean().item():.4f}"
            print(msg)

    # Save a simple long-format metrics.csv compatible with compare_metrics.py.
    try:
        import pandas as pd

        final = history[-1]
        metrics = [
            {"metric": "teacher_forced_mse", "value": float(final["val_loss"])},
            {"metric": "final_train_loss", "value": float(final["train_loss"])},
            {"metric": "final_val_loss", "value": float(final["val_loss"])},
        ]
        for k in ["extras/fast_weight_norm_mean", "extras/fast_drive_norm_mean"]:
            if k in final:
                metrics.append({"metric": k, "value": float(final[k])})
        pd.DataFrame(metrics).to_csv(out_dir / "metrics.csv", index=False)
    except Exception as e:
        print(f"Warning: could not write metrics.csv: {e}")

    print(f"Saved best checkpoint to {out_dir / 'best.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["vanilla", "wave", "globalfast", "localfast", "fastwave"],
        default="fastwave",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="data/runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--task",
        choices=[
            "ring",
            "ambiguous_ring",
            "eight_arm",
            "eight_arm_traj",
            "eight_arm_bump_traj",
        ],
        default="ring",
    )
    p.add_argument("--bump-sigma", type=float, default=0.75)
    p.add_argument("--forced-departure-weight", type=float, default=3.0)
    p.add_argument("--choice-departure-weight", type=float, default=10.0)
    p.add_argument("--arm-choice-weight", type=float, default=50.0)
    p.add_argument("--routing-weight", type=float, default=20.0)

    p.add_argument("--n-space", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--velocity", type=int, default=1)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--settle-steps", type=int, default=2)
    p.add_argument("--arm-len", type=int, default=3)
    p.add_argument("--reward-hold-steps", type=int, default=1)
    p.add_argument("--center-hold-steps", type=int, default=0)
    p.add_argument("--choice-order", choices=["random", "ascending"], default="random")
    p.add_argument("--n-arms", type=int, default=8)
    p.add_argument("--n-forced", type=int, default=4)
    p.add_argument("--expose-visited-memory", action="store_true")
    p.add_argument(
        "--no-availability", dest="include_availability", action="store_false"
    )
    p.set_defaults(include_availability=True)
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-val", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--kernel-size", type=int, default=7)
    p.add_argument("--patch-size", type=int, default=5)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=0.2)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument(
        "--fast-update", choices=["autoassoc", "transition"], default="transition"
    )

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--print-every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
