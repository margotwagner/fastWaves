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
            readout_state=getattr(args, "wave_readout", "x"),
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
            readout_state=getattr(args, "wave_readout", "x"),
            lam=args.lam,
            eta=args.eta,
            beta=args.beta,
            fast_update=args.fast_update,
            fast_write_phase=getattr(args, "fast_write_phase", "all"),
            fast_nonwrite_mode=getattr(args, "fast_nonwrite_mode", "decay"),
            fast_write_cue_index=getattr(args, "fast_write_cue_index", None),
            fast_write_reward_cue_index=getattr(
                args, "fast_write_reward_cue_index", None
            ),
        )
    raise ValueError(f"Unknown model: {args.model}")



def unpack_batch(batch, device):
    """Support datasets returning 2, 3, or 4 tensors."""
    if len(batch) == 4:
        x, y, weights, valid_choice_masks = batch
        return (
            x.to(device),
            y.to(device),
            weights.to(device),
            valid_choice_masks.to(device),
        )
    if len(batch) == 3:
        x, y, weights = batch
        return x.to(device), y.to(device), weights.to(device), None
    if len(batch) == 2:
        x, y = batch
        return x.to(device), y.to(device), None, None
    raise ValueError(f"Expected batch length 2, 3, or 4, got {len(batch)}")


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

def valid_choice_set_loss(
    yhat,
    valid_choice_masks,
    arm_choice_start,
    arm_choice_end,
):
    """
    Partial-label categorical loss for the memory-dependent arm decision.

    At an active choice event, valid_choice_masks[b, t, a] is one for every
    arm that is still unvisited in that teacher-forced history. The mask is
    supervision metadata only; it is never passed into model(x).

    The loss is -log of the total softmax probability assigned to valid arms.
    """
    if valid_choice_masks is None:
        return yhat.new_zeros(())

    active = valid_choice_masks.sum(dim=-1) > 0.5
    if not active.any():
        return yhat.new_zeros(())

    logits = yhat[..., arm_choice_start:arm_choice_end][active]
    masks = valid_choice_masks[active].bool()
    log_probs = F.log_softmax(logits, dim=-1)
    valid_log_probs = log_probs.masked_fill(~masks, float("-inf"))
    log_valid_mass = torch.logsumexp(valid_log_probs, dim=-1)
    return -log_valid_mass.mean()


def valid_choice_commitment_loss(
    yhat,
    valid_choice_masks,
    arm_choice_start,
    arm_choice_end,
):
    """Encourage a decisive categorical choice while accepting any valid arm.

    For each action-selection event, find the model's highest-logit arm among
    the currently valid arms, detach that pseudo-target, and train the full
    eight-way action head toward it. Invalid arms remain competitors in the
    cross-entropy denominator, so they are penalized. The valid mask is loss
    metadata only and is never passed to model(x).
    """
    if valid_choice_masks is None:
        return yhat.new_zeros(())

    active = valid_choice_masks.sum(dim=-1) > 0.5
    if not active.any():
        return yhat.new_zeros(())

    logits = yhat[..., arm_choice_start:arm_choice_end][active]
    valid = valid_choice_masks[active].bool()
    valid_logits = logits.masked_fill(~valid, float("-inf"))
    chosen_valid_arm = valid_logits.detach().argmax(dim=-1)
    return F.cross_entropy(logits, chosen_valid_arm)


def compute_choice_loss(
    yhat,
    valid_choice_masks,
    dataset,
    choice_objective,
):
    """Dispatch the choice loss while preserving exact one-hot supervision."""
    if valid_choice_masks is None or choice_objective == "exact":
        return yhat.new_zeros(())

    kwargs = dict(
        yhat=yhat,
        valid_choice_masks=valid_choice_masks,
        arm_choice_start=dataset.arm_choice_start,
        arm_choice_end=dataset.arm_choice_end,
    )
    if choice_objective == "valid_set":
        return valid_choice_set_loss(**kwargs)
    if choice_objective == "commitment":
        return valid_choice_commitment_loss(**kwargs)
    raise ValueError(f"Unknown choice_objective: {choice_objective}")


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    train_ds = build_dataset(args, n_samples=args.n_train, seed=args.seed)
    val_ds = build_dataset(args, n_samples=args.n_val, seed=args.seed + 1)

    # Resolve task cue indices before constructing FastWave. The indices are
    # saved in checkpoint args so diagnostic reconstruction remains exact.
    write_phase = getattr(args, "fast_write_phase", "all")
    if args.model == "fastwave" and write_phase in {"forced", "forced_reward"}:
        if not hasattr(train_ds, "cue_forced"):
            raise ValueError(
                f"--fast-write-phase {write_phase} requires a task with a "
                "cue_forced attribute, such as eight_arm_bump_traj"
            )
        args.fast_write_cue_index = int(train_ds.cue_forced)
        if (
            not hasattr(val_ds, "cue_forced")
            or int(val_ds.cue_forced) != args.fast_write_cue_index
        ):
            raise RuntimeError(
                "Training and validation datasets have inconsistent forced cues"
            )
    else:
        args.fast_write_cue_index = None

    if args.model == "fastwave" and write_phase == "forced_reward":
        if not hasattr(train_ds, "cue_reward"):
            raise ValueError(
                "--fast-write-phase forced_reward requires a task with a "
                "cue_reward attribute, such as eight_arm_bump_traj"
            )
        args.fast_write_reward_cue_index = int(train_ds.cue_reward)
        if (
            not hasattr(val_ds, "cue_reward")
            or int(val_ds.cue_reward) != args.fast_write_reward_cue_index
        ):
            raise RuntimeError(
                "Training and validation datasets have inconsistent reward cues"
            )
    else:
        args.fast_write_reward_cue_index = None

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
        train_mse = 0.0
        train_choice = 0.0
        for batch in train_loader:
            x, y, weights, valid_choice_masks = unpack_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            yhat, extras = model(x)

            mse_loss = sequence_mse_loss(yhat, y, weights)
            choice_loss = compute_choice_loss(
                yhat=yhat,
                valid_choice_masks=valid_choice_masks,
                dataset=train_ds,
                choice_objective=getattr(args, "choice_objective", "exact"),
            )
            loss = mse_loss + args.valid_choice_loss_weight * choice_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            batch_n = x.size(0)
            train_loss += loss.item() * batch_n
            train_mse += mse_loss.item() * batch_n
            train_choice += choice_loss.item() * batch_n

        train_loss /= len(train_ds)
        train_mse /= len(train_ds)
        train_choice /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_choice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y, weights, valid_choice_masks = unpack_batch(batch, device)
                yhat, extras = model(x)

                mse_loss = sequence_mse_loss(yhat, y, weights)
                choice_loss = compute_choice_loss(
                    yhat=yhat,
                    valid_choice_masks=valid_choice_masks,
                    dataset=val_ds,
                    choice_objective=getattr(args, "choice_objective", "exact"),
                )
                loss = mse_loss + args.valid_choice_loss_weight * choice_loss

                batch_n = x.size(0)
                val_loss += loss.item() * batch_n
                val_mse += mse_loss.item() * batch_n
                val_choice += choice_loss.item() * batch_n

        val_loss /= len(val_ds)
        val_mse /= len(val_ds)
        val_choice /= len(val_ds)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args)},
                out_dir / "best.pt",
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_valid_choice_loss": train_choice,
            "val_valid_choice_loss": val_choice,
        }
        if "fast_weight_norm" in extras:
            row["extras/fast_weight_norm_mean"] = (
                extras["fast_weight_norm"].mean().item()
            )
        if "fast_drive_norm" in extras:
            row["extras/fast_drive_norm_mean"] = extras["fast_drive_norm"].mean().item()
        if "fast_write_gate" in extras:
            row["extras/fast_write_gate_mean"] = extras["fast_write_gate"].mean().item()
        history.append(row)

        if epoch == 1 or epoch % args.print_every == 0:
            msg = (
                f"epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f}"
                f" | mse {val_mse:.6f} | choice {val_choice:.6f}"
            )
            if "fast_weight_norm" in extras:
                msg += f" | F_norm {extras['fast_weight_norm'].mean().item():.4f}"
            print(msg)

    # Save a simple long-format metrics.csv compatible with compare_metrics.py.
    try:
        import pandas as pd

        final = history[-1]
        metrics = [
            {"metric": "teacher_forced_mse", "value": float(final["val_mse"])},
            {"metric": "final_train_loss", "value": float(final["train_loss"])},
            {"metric": "final_val_loss", "value": float(final["val_loss"])},
            {"metric": "final_train_mse", "value": float(final["train_mse"])},
            {"metric": "final_val_mse", "value": float(final["val_mse"])},
            {
                "metric": "final_train_valid_choice_loss",
                "value": float(final["train_valid_choice_loss"]),
            },
            {
                "metric": "final_val_valid_choice_loss",
                "value": float(final["val_valid_choice_loss"]),
            },
            {
                "metric": "final_train_choice_loss",
                "value": float(final["train_valid_choice_loss"]),
            },
            {
                "metric": "final_val_choice_loss",
                "value": float(final["val_valid_choice_loss"]),
            },
        ]
        for k in [
            "extras/fast_weight_norm_mean",
            "extras/fast_drive_norm_mean",
            "extras/fast_write_gate_mean",
        ]:
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
    p.add_argument(
        "--action-hold-steps",
        type=int,
        default=3,
        help=(
            "Number of center frames carrying the selected action before the "
            "spatial trajectory must leave the center."
        ),
    )
    p.add_argument("--choice-order", choices=["random", "ascending"], default="random")
    p.add_argument(
        "--choice-objective",
        choices=["exact", "valid_set", "commitment"],
        default="commitment",
        help=(
            "exact uses the sampled one-hot target; valid_set rewards total "
            "probability on unvisited arms; commitment reinforces the model's "
            "preferred currently valid arm as a decisive categorical action."
        ),
    )
    p.add_argument(
        "--valid-choice-loss-weight",
        type=float,
        default=0.1,
        help="Multiplier on the valid-set or commitment arm-choice loss.",
    )
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
    p.add_argument(
        "--wave-readout",
        choices=["x", "xv"],
        default="xv",
        help=(
            "For Wave/FastWave, decode outputs from activity x alone or from "
            "the full second-order state [x, v]."
        ),
    )
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument(
        "--fast-update", choices=["autoassoc", "transition"], default="transition"
    )
    p.add_argument(
        "--fast-write-phase",
        choices=["all", "forced", "forced_reward"],
        default="all",
        help=(
            "For FastWave, update fast weights at every timestep, during the "
            "forced phase only, or during the forced phase plus reward events."
        ),
    )
    p.add_argument(
        "--fast-nonwrite-mode",
        choices=["decay", "hold"],
        default="decay",
        help=(
            "Fast-memory behavior outside the permitted write phase: "
            "'decay' applies F <- lambda F; 'hold' keeps F unchanged."
        ),
    )

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--print-every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
