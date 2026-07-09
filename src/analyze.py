import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.tasks import build_dataset
from src.train import build_model


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def circular_decode(y):
    """
    Decode ring position from activity.

    y: [B, T, N]
    returns:
        angles: [B, T]
        positions: [B, T]
    """
    B, T, N = y.shape
    p = torch.softmax(y, dim=-1)

    theta = torch.linspace(0, 2 * torch.pi, N + 1, device=y.device)[:-1]

    c = torch.sum(p * torch.cos(theta), dim=-1)
    s = torch.sum(p * torch.sin(theta), dim=-1)

    ang = torch.atan2(s, c) % (2 * torch.pi)
    pos = ang / (2 * torch.pi) * N

    return ang, pos


def circular_error(pred_ang, true_ang):
    """
    Smallest absolute circular angle error in radians.
    """
    return torch.abs(
        torch.atan2(
            torch.sin(pred_ang - true_ang),
            torch.cos(pred_ang - true_ang),
        )
    )


def _mean_or_nan(vals):
    vals = list(vals)
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def _masked_mean_tensor(values, mask):
    if mask.sum().item() == 0:
        return float("nan")
    return values[mask].mean().item()


def eight_arm_traj_metrics(yhat, x, y, dataset, prefix="eightarm"):
    """
    Task-specific metrics for EightArmTrajectoryDataset.

    yhat, x, y: [B, T, N]
    dataset must expose the metadata produced by EightArmTrajectoryDataset:
        n_pos, arm_len, n_arms, cue_* indices, forced_orders, choice_orders.
    """
    B, T, N = y.shape
    n_pos = int(dataset.n_pos)
    arm_len = int(dataset.arm_len)
    n_arms = int(dataset.n_arms)

    # Decode the discrete maze position from the position channels only.
    pred_pos = yhat[:, :, :n_pos].argmax(dim=-1)
    true_pos = y[:, :, :n_pos].argmax(dim=-1)
    x_pos = x[:, :, :n_pos].argmax(dim=-1)

    pos_correct = pred_pos.eq(true_pos)
    mse_per_tn = F.mse_loss(yhat, y, reduction="none").mean(dim=-1)

    # Masks based on the target state's phase cues.
    forced_mask = y[:, :, dataset.cue_forced] > 0.5
    choice_mask = y[:, :, dataset.cue_choice] > 0.5
    settle_mask = y[:, :, dataset.cue_settle] > 0.5
    reward_mask = y[:, :, dataset.cue_reward] > 0.5
    outbound_mask = y[:, :, dataset.cue_outbound] > 0.5
    inbound_mask = y[:, :, dataset.cue_inbound] > 0.5
    center_mask = y[:, :, dataset.cue_center] > 0.5

    metrics = {
        f"{prefix}/position_acc": pos_correct.float().mean().item(),
        f"{prefix}/position_acc_forced": _masked_mean_tensor(
            pos_correct.float(), forced_mask
        ),
        f"{prefix}/position_acc_choice": _masked_mean_tensor(
            pos_correct.float(), choice_mask
        ),
        f"{prefix}/position_acc_settle": _masked_mean_tensor(
            pos_correct.float(), settle_mask
        ),
        f"{prefix}/mse_forced": _masked_mean_tensor(mse_per_tn, forced_mask),
        f"{prefix}/mse_choice": _masked_mean_tensor(mse_per_tn, choice_mask),
        f"{prefix}/mse_settle": _masked_mean_tensor(mse_per_tn, settle_mask),
        f"{prefix}/reward_site_acc": _masked_mean_tensor(
            pos_correct.float(), reward_mask
        ),
        f"{prefix}/center_state_acc": _masked_mean_tensor(
            pos_correct.float(), center_mask
        ),
    }

    # Phase-cue classification accuracy.
    phase_cues = torch.tensor(
        [dataset.cue_forced, dataset.cue_choice, dataset.cue_settle],
        device=y.device,
        dtype=torch.long,
    )
    pred_phase = yhat[:, :, phase_cues].argmax(dim=-1)
    true_phase = y[:, :, phase_cues].argmax(dim=-1)
    metrics[f"{prefix}/phase_cue_acc"] = pred_phase.eq(true_phase).float().mean().item()

    direction_cues = torch.tensor(
        [
            dataset.cue_outbound,
            dataset.cue_inbound,
            dataset.cue_reward,
            dataset.cue_center,
        ],
        device=y.device,
        dtype=torch.long,
    )
    pred_dir = yhat[:, :, direction_cues].argmax(dim=-1)
    true_dir = y[:, :, direction_cues].argmax(dim=-1)
    metrics[f"{prefix}/direction_cue_acc"] = pred_dir.eq(true_dir).float().mean().item()

    # Convert position index to arm index. Center has arm = -1.
    def pos_to_arm(pos):
        arm = torch.full_like(pos, -1)
        is_arm = (pos >= 1) & (pos < n_pos)
        arm[is_arm] = (pos[is_arm] - 1) // arm_len
        return arm

    pred_arm = pos_to_arm(pred_pos)
    true_arm = pos_to_arm(true_pos)

    # Choice departures are the important memory-dependent moments:
    # current input is center, target next position is an arm position, target phase is choice.
    choice_departure_mask = (
        (x_pos == dataset.center_idx) & choice_mask & (true_arm >= 0)
    )

    exact_departure = []
    first_choice_exact = []
    valid_unvisited = []
    invalid_forced = []
    invalid_choice_reentry = []
    no_arm_departure = []

    forced_orders = dataset.forced_orders.to(y.device)
    choice_orders = dataset.choice_orders.to(y.device)

    for b in range(B):
        ts = torch.where(choice_departure_mask[b])[0]
        # The target dataset has one center->arm departure per choice arm.
        # Sort by time so j indexes first, second, third, fourth choice.
        for j, t in enumerate(ts.tolist()):
            pa = int(pred_arm[b, t].item())
            ta = int(true_arm[b, t].item())

            exact_departure.append(float(pa == ta))
            no_arm_departure.append(float(pa < 0))

            if j == 0:
                first_choice_exact.append(float(pa == ta))

            forced_set = set(int(a.item()) for a in forced_orders[b])
            already_choice_set = set(int(a.item()) for a in choice_orders[b, :j])
            remaining_choice_set = set(int(a.item()) for a in choice_orders[b, j:])

            valid_unvisited.append(float(pa in remaining_choice_set))
            invalid_forced.append(float(pa in forced_set))
            invalid_choice_reentry.append(
                float(pa in already_choice_set or pa in forced_set)
            )

    metrics.update(
        {
            f"{prefix}/choice_departure_exact_arm_acc": _mean_or_nan(exact_departure),
            f"{prefix}/first_choice_exact_arm_acc": _mean_or_nan(first_choice_exact),
            f"{prefix}/valid_unvisited_choice_rate": _mean_or_nan(valid_unvisited),
            f"{prefix}/invalid_forced_reentry_rate": _mean_or_nan(invalid_forced),
            f"{prefix}/invalid_choice_reentry_rate": _mean_or_nan(
                invalid_choice_reentry
            ),
            f"{prefix}/choice_departure_no_arm_rate": _mean_or_nan(no_arm_departure),
            f"{prefix}/n_choice_departures": float(len(exact_departure)),
        }
    )

    # Optional auxiliary arm-choice head metrics. These exist for the new
    # eight_arm_bump_traj version with n_space >= 40. They directly evaluate
    # the 8 action/arm channels rather than decoding the arm only from position.
    if hasattr(dataset, "arm_choice_start") and hasattr(dataset, "arm_choice_end"):
        a0 = int(dataset.arm_choice_start)
        a1 = int(dataset.arm_choice_end)
        if a1 <= yhat.shape[-1]:
            true_action_active = y[:, :, a0:a1].sum(dim=-1) > 0.5
            pred_action_arm = yhat[:, :, a0:a1].argmax(dim=-1)
            true_action_arm = y[:, :, a0:a1].argmax(dim=-1)
            action_exact = pred_action_arm.eq(true_action_arm)
            metrics[f"{prefix}/arm_choice_head_acc"] = _masked_mean_tensor(
                action_exact.float(), true_action_active
            )
            metrics[f"{prefix}/arm_choice_head_acc_choice_departure"] = _masked_mean_tensor(
                action_exact.float(), choice_departure_mask
            )
            action_mse = F.mse_loss(yhat[:, :, a0:a1], y[:, :, a0:a1], reduction="none").mean(dim=-1)
            metrics[f"{prefix}/arm_choice_head_mse"] = _masked_mean_tensor(
                action_mse, true_action_active
            )
            metrics[f"{prefix}/arm_choice_head_mse_choice_departure"] = _masked_mean_tensor(
                action_mse, choice_departure_mask
            )

    return metrics


def _format_int_list(vals):
    return " ".join(str(int(v)) for v in vals)


def eight_arm_choice_debug_rows(
    yhat, x, y, dataset, split_name="rollout", max_trials=16
):
    """
    Return row dictionaries describing the model's arm choices at the
    memory-critical center->choice-arm departure steps.

    This is a debugging table, not a scalar metric. It lets you inspect
    whether zero valid-choice scores are real failures or decoding issues.
    """
    B, T, N = y.shape
    n_pos = int(dataset.n_pos)
    arm_len = int(dataset.arm_len)
    n_arms = int(dataset.n_arms)

    pred_pos = yhat[:, :, :n_pos].argmax(dim=-1)
    true_pos = y[:, :, :n_pos].argmax(dim=-1)
    x_pos = x[:, :, :n_pos].argmax(dim=-1)

    choice_mask = y[:, :, dataset.cue_choice] > 0.5

    def pos_to_arm(pos):
        arm = torch.full_like(pos, -1)
        is_arm = (pos >= 1) & (pos < n_pos)
        arm[is_arm] = (pos[is_arm] - 1) // arm_len
        return arm

    pred_arm = pos_to_arm(pred_pos)
    true_arm = pos_to_arm(true_pos)

    choice_departure_mask = (
        (x_pos == dataset.center_idx) & choice_mask & (true_arm >= 0)
    )

    forced_orders = dataset.forced_orders.to(y.device)
    choice_orders = dataset.choice_orders.to(y.device)

    rows = []
    n_show = min(B, max_trials)

    for b in range(n_show):
        forced_list = [int(a.item()) for a in forced_orders[b]]
        choice_list = [int(a.item()) for a in choice_orders[b]]
        forced_set = set(forced_list)

        ts = torch.where(choice_departure_mask[b])[0]

        for j, t in enumerate(ts.tolist()):
            pa = int(pred_arm[b, t].item())
            ta = int(true_arm[b, t].item())

            already_choice_set = set(choice_list[:j])
            remaining_choice_set = set(choice_list[j:])

            valid_unvisited = pa in remaining_choice_set
            invalid_forced = pa in forced_set
            invalid_choice_reentry = pa in already_choice_set or pa in forced_set
            no_arm = pa < 0
            exact = pa == ta

            top_vals, top_idx = torch.topk(yhat[b, t, :n_pos], k=min(5, n_pos))
            top_pos = [int(z.item()) for z in top_idx]
            top_val = [float(z.item()) for z in top_vals]
            top_arm = []
            for pp in top_pos:
                if pp == int(dataset.center_idx):
                    top_arm.append("center")
                elif 1 <= pp < n_pos:
                    top_arm.append(str((pp - 1) // arm_len))
                else:
                    top_arm.append("other")

            rows.append(
                {
                    "split": split_name,
                    "trial": b,
                    "t": t,
                    "choice_index": j,
                    "forced_arms": _format_int_list(forced_list),
                    "true_choice_order": _format_int_list(choice_list),
                    "remaining_valid_arms": _format_int_list(choice_list[j:]),
                    "already_choice_arms": _format_int_list(choice_list[:j]),
                    "x_pos": int(x_pos[b, t].item()),
                    "true_pos": int(true_pos[b, t].item()),
                    "pred_pos": int(pred_pos[b, t].item()),
                    "true_arm": ta,
                    "pred_arm": pa,
                    "exact_arm": int(exact),
                    "valid_unvisited": int(valid_unvisited),
                    "invalid_forced_reentry": int(invalid_forced),
                    "invalid_choice_reentry": int(invalid_choice_reentry),
                    "no_arm_departure": int(no_arm),
                    "pred_center_value": float(yhat[b, t, dataset.center_idx].item()),
                    "pred_true_pos_value": float(yhat[b, t, true_pos[b, t]].item()),
                    "pred_pos_top5": _format_int_list(top_pos),
                    "pred_arm_top5": " ".join(top_arm),
                    "pred_value_top5": " ".join(f"{v:.4f}" for v in top_val),
                }
            )

    return rows


def write_eight_arm_choice_debug(
    model, dataset, device, out_dir, prefix_len, max_trials=16
):
    """
    Write a CSV with per-trial choice-departure debug info for teacher-forced
    and autonomous rollout predictions.
    """
    if max_trials <= 0:
        return

    model.eval()
    x = dataset.x.to(device)
    y = dataset.y.to(device)

    with torch.no_grad():
        yhat_tf, _ = model(x)
        yhat_roll = autonomous_rollout(model, x, prefix_len=prefix_len)

    rows = []
    rows.extend(
        eight_arm_choice_debug_rows(
            yhat=yhat_tf,
            x=x,
            y=y,
            dataset=dataset,
            split_name="teacher_forced",
            max_trials=max_trials,
        )
    )
    rows.extend(
        eight_arm_choice_debug_rows(
            yhat=yhat_roll,
            x=x,
            y=y,
            dataset=dataset,
            split_name="rollout",
            max_trials=max_trials,
        )
    )

    if len(rows) == 0:
        print("No eight-arm choice-departure rows found for debugging.")
        return

    out_path = out_dir / "eightarm_choice_debug.csv"
    fieldnames = list(rows[0].keys())

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved eight-arm choice debug table to {out_path}")


def teacher_forced_eight_arm_eval(model, dataset, device):
    model.eval()
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    with torch.no_grad():
        yhat, _ = model(x)
    return eight_arm_traj_metrics(
        yhat=yhat,
        x=x,
        y=y,
        dataset=dataset,
        prefix="eightarm_tf",
    )


def rollout_eight_arm_eval(model, dataset, device, prefix_len):
    model.eval()
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    yhat = autonomous_rollout(model, x, prefix_len=prefix_len)
    return eight_arm_traj_metrics(
        yhat=yhat,
        x=x,
        y=y,
        dataset=dataset,
        prefix="eightarm_rollout",
    )


def _unpack_xy(batch):
    """
    Support datasets that return either:
        (x, y)
    or weighted datasets that return:
        (x, y, loss_weights)

    Analysis only needs x and y.
    """
    if len(batch) == 2:
        x, y = batch
    elif len(batch) == 3:
        x, y, _ = batch
    else:
        raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")
    return x, y


def teacher_forced_eval(model, loader, device):
    model.eval()

    losses = []
    ang_errs = []
    extras_summary = {}

    with torch.no_grad():
        for batch in loader:
            x, y = _unpack_xy(batch)
            x = x.to(device)
            y = y.to(device)

            yhat, extras = model(x)

            loss_per_sample = F.mse_loss(
                yhat,
                y,
                reduction="none",
            ).mean(dim=(1, 2))

            losses.append(loss_per_sample.cpu())

            pred_ang, _ = circular_decode(yhat)
            true_ang, _ = circular_decode(y)

            ang_err = circular_error(pred_ang, true_ang).mean(dim=1)
            ang_errs.append(ang_err.cpu())

            for k, v in extras.items():
                if torch.is_tensor(v):
                    extras_summary.setdefault(k, []).append(
                        v.detach().float().mean().cpu()
                    )

    metrics = {
        "teacher_forced_mse": torch.cat(losses).mean().item(),
        "teacher_forced_angle_error_rad": torch.cat(ang_errs).mean().item(),
    }

    for k, vals in extras_summary.items():
        metrics[f"extras/{k}_mean"] = torch.stack(vals).mean().item()

    return metrics


def autonomous_rollout(model, x, prefix_len):
    """
    Autoregressive rollout.

    First prefix_len inputs are true.
    After that, each previous prediction becomes the next input.

    x: [B, T, N]
    returns:
        yhat_roll: [B, T, N]
    """
    model.eval()

    B, T, N = x.shape
    u_roll = x.clone()
    preds = []

    with torch.no_grad():
        for t in range(T):
            yhat_prefix, _ = model(u_roll[:, : t + 1, :])
            pred_t = yhat_prefix[:, -1, :]

            preds.append(pred_t)

            if t + 1 < T and t + 1 >= prefix_len:
                u_roll[:, t + 1, :] = pred_t

    return torch.stack(preds, dim=1)


def rollout_eval_and_plots(
    model,
    dataset,
    device,
    out_dir,
    prefix_len=5,
    angle_threshold_rad=0.75,
):
    x = dataset.x.to(device)
    y = dataset.y.to(device)

    yhat = autonomous_rollout(model, x, prefix_len=prefix_len)

    mse_t = F.mse_loss(yhat, y, reduction="none").mean(dim=(0, 2)).cpu().numpy()

    pred_ang, pred_pos = circular_decode(yhat)
    true_ang, true_pos = circular_decode(y)

    ang_err = circular_error(pred_ang, true_ang)
    ang_err_t = ang_err.mean(dim=0).cpu().numpy()

    bad = ang_err[:, prefix_len:] > angle_threshold_rad

    horizons = []
    for b in range(bad.shape[0]):
        idx = torch.where(bad[b])[0]
        if len(idx) == 0:
            horizons.append(y.shape[1] - prefix_len)
        else:
            horizons.append(int(idx[0].item()))

    horizon = float(np.mean(horizons))

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(mse_t, label="rollout MSE")
    plt.axvline(prefix_len, linestyle="--", label="end of true prefix")
    plt.xlabel("time step")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rollout_mse.png", dpi=150)
    plt.close()

    i = 0

    plt.figure(figsize=(7, 5))
    plt.imshow(y[i].detach().cpu().T, aspect="auto", origin="lower")
    plt.xlabel("time step")
    plt.ylabel("ring position")
    plt.title("target sequence")
    plt.tight_layout()
    plt.savefig(out_dir / "target_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.imshow(yhat[i].detach().cpu().T, aspect="auto", origin="lower")
    plt.xlabel("time step")
    plt.ylabel("ring position")
    plt.title("autonomous prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(true_pos[i].detach().cpu().numpy(), label="target")
    plt.plot(pred_pos[i].detach().cpu().numpy(), label="prediction")
    plt.axvline(prefix_len, linestyle="--", label="end of true prefix")
    plt.xlabel("time step")
    plt.ylabel("decoded ring position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "decoded_position.png", dpi=150)
    plt.close()

    return {
        "rollout_mse_mean": float(np.mean(mse_t)),
        "rollout_angle_error_rad": float(np.mean(ang_err_t)),
        "prediction_horizon_steps": horizon,
    }


def write_metrics(metrics, out_dir):
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, v])


def main(args):
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    ckpt = safe_torch_load(args.ckpt, device)

    train_args = SimpleNamespace(**ckpt["args"])
    train_args.device = str(device)

    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_ds = build_dataset(
        args=train_args,
        n_samples=args.n_test,
        seed=args.seed,
    )

    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    if args.out_dir == "auto":
        out_dir = Path(args.ckpt).parent / "analysis"
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}

    metrics.update(
        teacher_forced_eval(
            model=model,
            loader=test_loader,
            device=device,
        )
    )

    metrics.update(
        rollout_eval_and_plots(
            model=model,
            dataset=test_ds,
            device=device,
            out_dir=out_dir,
            prefix_len=args.prefix_len,
            angle_threshold_rad=args.angle_threshold_rad,
        )
    )

    if getattr(train_args, "task", None) in {"eight_arm_traj", "eight_arm_bump_traj"}:
        metrics.update(
            teacher_forced_eight_arm_eval(
                model=model,
                dataset=test_ds,
                device=device,
            )
        )
        metrics.update(
            rollout_eight_arm_eval(
                model=model,
                dataset=test_ds,
                device=device,
                prefix_len=args.prefix_len,
            )
        )
        write_eight_arm_choice_debug(
            model=model,
            dataset=test_ds,
            device=device,
            out_dir=out_dir,
            prefix_len=args.prefix_len,
            max_trials=args.debug_trials,
        )

    write_metrics(metrics, out_dir)

    print(f"Analysis for {args.ckpt}")
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v:.6f}")

    print(f"Saved plots and metrics to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="auto")

    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--prefix-len", type=int, default=5)
    p.add_argument("--angle-threshold-rad", type=float, default=0.75)
    p.add_argument("--debug-trials", type=int, default=16)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
