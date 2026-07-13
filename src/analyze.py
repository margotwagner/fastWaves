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


def eight_arm_traj_metrics(
    yhat,
    x,
    y,
    dataset,
    prefix="eightarm",
    rollout_mode=False,
):
    """Task-specific metrics for the eight-arm trajectory tasks.

    Target-relative MSE/accuracy metrics are retained for diagnostics. For the
    action-conditioned bump task, the primary behavioral metrics track the
    model's selected arms and, in rollout mode, update visited history from the
    arms that the model actually enters spatially.
    """
    B, T, _ = y.shape
    n_pos = int(dataset.n_pos)
    arm_len = int(dataset.arm_len)
    n_arms = int(dataset.n_arms)

    pred_pos = yhat[:, :, :n_pos].argmax(dim=-1)
    true_pos = y[:, :, :n_pos].argmax(dim=-1)
    x_pos = x[:, :, :n_pos].argmax(dim=-1)

    pos_correct = pred_pos.eq(true_pos)
    mse_per_tn = F.mse_loss(yhat, y, reduction="none").mean(dim=-1)

    forced_mask = y[:, :, dataset.cue_forced] > 0.5
    choice_mask = y[:, :, dataset.cue_choice] > 0.5
    settle_mask = y[:, :, dataset.cue_settle] > 0.5
    reward_mask = y[:, :, dataset.cue_reward] > 0.5
    center_mask = y[:, :, dataset.cue_center] > 0.5

    metrics = {
        f"{prefix}/position_acc": pos_correct.float().mean().item(),
        f"{prefix}/position_acc_forced": _masked_mean_tensor(
            pos_correct.float(), forced_mask
        ),
        f"{prefix}/position_acc_choice_target_relative": _masked_mean_tensor(
            pos_correct.float(), choice_mask
        ),
        f"{prefix}/position_acc_settle": _masked_mean_tensor(
            pos_correct.float(), settle_mask
        ),
        f"{prefix}/mse_forced": _masked_mean_tensor(mse_per_tn, forced_mask),
        f"{prefix}/mse_choice_target_relative": _masked_mean_tensor(
            mse_per_tn, choice_mask
        ),
        f"{prefix}/mse_settle": _masked_mean_tensor(mse_per_tn, settle_mask),
        f"{prefix}/reward_site_acc_target_relative": _masked_mean_tensor(
            pos_correct.float(), reward_mask
        ),
        f"{prefix}/center_state_acc": _masked_mean_tensor(
            pos_correct.float(), center_mask
        ),
    }

    phase_cues = torch.tensor(
        [dataset.cue_forced, dataset.cue_choice, dataset.cue_settle],
        device=y.device,
        dtype=torch.long,
    )
    pred_phase = yhat[:, :, phase_cues].argmax(dim=-1)
    true_phase = y[:, :, phase_cues].argmax(dim=-1)
    metrics[f"{prefix}/phase_cue_acc"] = (
        pred_phase.eq(true_phase).float().mean().item()
    )

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
    metrics[f"{prefix}/direction_cue_acc"] = (
        pred_dir.eq(true_dir).float().mean().item()
    )

    def pos_to_arm(pos):
        arm = torch.full_like(pos, -1)
        is_arm = (pos >= 1) & (pos < n_pos)
        arm[is_arm] = (pos[is_arm] - 1) // arm_len
        return arm

    pred_arm = pos_to_arm(pred_pos)
    true_arm = pos_to_arm(true_pos)

    # Older trajectory task without an explicit action head.
    if not (
        hasattr(dataset, "arm_choice_start")
        and hasattr(dataset, "arm_choice_end")
        and int(dataset.arm_choice_end) <= yhat.shape[-1]
    ):
        choice_departure_mask = (
            (x_pos == dataset.center_idx) & choice_mask & (true_arm >= 0)
        )
        exact = pred_arm.eq(true_arm)
        departed = pred_arm >= 0
        metrics[f"{prefix}/routing_exact_arm_acc"] = _masked_mean_tensor(
            exact.float(), choice_departure_mask
        )
        metrics[f"{prefix}/routing_departure_rate"] = _masked_mean_tensor(
            departed.float(), choice_departure_mask
        )
        return metrics

    a0 = int(dataset.arm_choice_start)
    a1 = int(dataset.arm_choice_end)
    pred_action_arm = yhat[:, :, a0:a1].argmax(dim=-1)
    true_action_arm = y[:, :, a0:a1].argmax(dim=-1)
    true_action_active = y[:, :, a0:a1].sum(dim=-1) > 0.5
    x_action_active = x[:, :, a0:a1].sum(dim=-1) > 0.5

    if hasattr(dataset, "valid_choice_masks"):
        valid_choice_masks = dataset.valid_choice_masks.to(y.device)
        action_selection_mask = valid_choice_masks.sum(dim=-1) > 0.5
    else:
        action_selection_mask = (
            (x_pos == dataset.center_idx)
            & (~x_action_active)
            & (true_pos == dataset.center_idx)
            & true_action_active
            & choice_mask
        )
        valid_choice_masks = None

    routing_mask = (
        (x_pos == dataset.center_idx)
        & x_action_active
        & choice_mask
        & (true_arm >= 0)
    )

    action_exact = pred_action_arm.eq(true_action_arm)
    action_mse = F.mse_loss(
        yhat[:, :, a0:a1], y[:, :, a0:a1], reduction="none"
    ).mean(dim=-1)
    spatial_departed = pred_arm >= 0
    routing_exact = pred_arm.eq(true_arm) & spatial_departed

    metrics[f"{prefix}/arm_choice_head_acc_all_active_target_relative"] = (
        _masked_mean_tensor(action_exact.float(), true_action_active)
    )
    metrics[f"{prefix}/action_selection_exact_target_arm_acc"] = (
        _masked_mean_tensor(action_exact.float(), action_selection_mask)
    )
    metrics[f"{prefix}/action_selection_mse_target_relative"] = (
        _masked_mean_tensor(action_mse, action_selection_mask)
    )
    metrics[f"{prefix}/routing_exact_target_arm_acc"] = _masked_mean_tensor(
        routing_exact.float(), routing_mask
    )
    metrics[f"{prefix}/routing_departure_rate"] = _masked_mean_tensor(
        spatial_departed.float(), routing_mask
    )

    if valid_choice_masks is not None:
        chosen_valid = torch.gather(
            valid_choice_masks,
            dim=-1,
            index=pred_action_arm.unsqueeze(-1),
        ).squeeze(-1) > 0.5
        metrics[f"{prefix}/action_selection_valid_under_teacher_history_rate"] = (
            _masked_mean_tensor(chosen_valid.float(), action_selection_mask)
        )

    forced_orders = dataset.forced_orders.to(y.device)

    action_valid_vals = []
    action_invalid_forced_vals = []
    action_reentry_vals = []
    route_departed_vals = []
    route_matches_conditioning_vals = []
    route_unvisited_vals = []
    step_success_vals = []
    first_action_valid_vals = []
    complete_trial_vals = []
    unique_unvisited_routed_counts = []
    n_events = 0

    for b in range(B):
        selection_times = torch.where(action_selection_mask[b])[0].tolist()
        forced_set = {int(a.item()) for a in forced_orders[b]}
        remaining_set = set(range(n_arms)) - forced_set
        visited = set(forced_set)
        unique_unvisited_routed = set()
        trial_success = len(selection_times) == int(dataset.n_choice)

        for j, t in enumerate(selection_times):
            route_t = t + 1
            if route_t >= T or not bool(routing_mask[b, route_t].item()):
                trial_success = False
                continue

            selected_action = int(pred_action_arm[b, t].item())
            predicted_route_arm = int(pred_arm[b, route_t].item())
            target_route_arm = int(true_arm[b, route_t].item())

            action_valid = selected_action not in visited
            action_invalid_forced = selected_action in forced_set
            action_reentry = selected_action in visited
            route_departed = predicted_route_arm >= 0
            route_unvisited = route_departed and predicted_route_arm not in visited

            # Under teacher forcing, the next input contains the sampled target
            # action. Under rollout, it contains the model's hardened selection.
            conditioning_action = (
                selected_action if rollout_mode else target_route_arm
            )
            route_matches_conditioning = (
                route_departed and predicted_route_arm == conditioning_action
            )
            step_success = action_valid and route_matches_conditioning

            action_valid_vals.append(float(action_valid))
            action_invalid_forced_vals.append(float(action_invalid_forced))
            action_reentry_vals.append(float(action_reentry))
            route_departed_vals.append(float(route_departed))
            route_matches_conditioning_vals.append(
                float(route_matches_conditioning)
            )
            route_unvisited_vals.append(float(route_unvisited))
            step_success_vals.append(float(step_success))
            if j == 0:
                first_action_valid_vals.append(float(action_valid))
            n_events += 1

            trial_success = trial_success and step_success

            if rollout_mode:
                # Behavioral history follows where the spatial trajectory actually
                # went, not the evaluator's target ordering.
                if route_departed:
                    if predicted_route_arm in remaining_set:
                        unique_unvisited_routed.add(predicted_route_arm)
                    visited.add(predicted_route_arm)
            else:
                # Teacher-forced input history follows the sampled valid route.
                visited.add(target_route_arm)

        if rollout_mode:
            trial_success = (
                trial_success
                and len(unique_unvisited_routed) == int(dataset.n_choice)
                and unique_unvisited_routed == remaining_set
            )
        complete_trial_vals.append(float(trial_success))
        unique_unvisited_routed_counts.append(float(len(unique_unvisited_routed)))

    metrics.update(
        {
            f"{prefix}/dynamic_action_valid_unvisited_rate": _mean_or_nan(
                action_valid_vals
            ),
            f"{prefix}/dynamic_first_action_valid_rate": _mean_or_nan(
                first_action_valid_vals
            ),
            f"{prefix}/dynamic_action_invalid_forced_rate": _mean_or_nan(
                action_invalid_forced_vals
            ),
            f"{prefix}/dynamic_action_reentry_rate": _mean_or_nan(
                action_reentry_vals
            ),
            f"{prefix}/dynamic_routing_departure_rate": _mean_or_nan(
                route_departed_vals
            ),
            f"{prefix}/dynamic_routing_matches_conditioning_action_rate": (
                _mean_or_nan(route_matches_conditioning_vals)
            ),
            f"{prefix}/dynamic_routing_enters_unvisited_rate": _mean_or_nan(
                route_unvisited_vals
            ),
            f"{prefix}/dynamic_step_success_rate": _mean_or_nan(
                step_success_vals
            ),
            f"{prefix}/dynamic_trial_complete_success_rate": _mean_or_nan(
                complete_trial_vals
            ),
            f"{prefix}/dynamic_unique_unvisited_arms_routed_mean": _mean_or_nan(
                unique_unvisited_routed_counts
            ),
            f"{prefix}/n_dynamic_choice_events": float(n_events),
        }
    )

    return metrics

def _format_int_list(vals):
    return " ".join(str(int(v)) for v in vals)


def eight_arm_choice_debug_rows(
    yhat,
    x,
    y,
    dataset,
    split_name="rollout",
    max_trials=16,
):
    """Return per-choice rows using the model's actual action and route."""
    if not hasattr(dataset, "valid_choice_masks"):
        return []

    B, T, _ = y.shape
    n_pos = int(dataset.n_pos)
    arm_len = int(dataset.arm_len)
    n_arms = int(dataset.n_arms)
    a0 = int(dataset.arm_choice_start)
    a1 = int(dataset.arm_choice_end)
    rollout_mode = split_name == "rollout"

    pred_pos = yhat[:, :, :n_pos].argmax(dim=-1)
    true_pos = y[:, :, :n_pos].argmax(dim=-1)
    pred_action = yhat[:, :, a0:a1].argmax(dim=-1)
    true_action = y[:, :, a0:a1].argmax(dim=-1)

    def pos_to_arm(pos):
        arm = torch.full_like(pos, -1)
        is_arm = (pos >= 1) & (pos < n_pos)
        arm[is_arm] = (pos[is_arm] - 1) // arm_len
        return arm

    pred_arm = pos_to_arm(pred_pos)
    true_arm = pos_to_arm(true_pos)
    selection_mask = dataset.valid_choice_masks.to(y.device).sum(dim=-1) > 0.5

    rows = []
    n_show = min(B, max_trials)
    for b in range(n_show):
        forced_list = [int(a.item()) for a in dataset.forced_orders[b]]
        target_choice_list = [int(a.item()) for a in dataset.choice_orders[b]]
        forced_set = set(forced_list)
        visited = set(forced_set)

        for j, t in enumerate(torch.where(selection_mask[b])[0].tolist()):
            route_t = t + 1
            if route_t >= T:
                continue

            selected = int(pred_action[b, t].item())
            target_selected = int(true_action[b, t].item())
            spatial_arm = int(pred_arm[b, route_t].item())
            target_route_arm = int(true_arm[b, route_t].item())
            valid_before = sorted(set(range(n_arms)) - visited)

            action_valid = selected in valid_before
            departed = spatial_arm >= 0
            conditioning_action = selected if rollout_mode else target_route_arm
            route_matches_conditioning = departed and spatial_arm == conditioning_action
            route_matches_selected = departed and spatial_arm == selected

            probs = torch.softmax(yhat[b, t, a0:a1], dim=-1)
            top_vals, top_idx = torch.topk(probs, k=min(4, n_arms))

            rows.append(
                {
                    "split": split_name,
                    "trial": b,
                    "choice_index": j,
                    "selection_t": t,
                    "routing_t": route_t,
                    "forced_arms": _format_int_list(forced_list),
                    "visited_before": _format_int_list(sorted(visited)),
                    "valid_before": _format_int_list(valid_before),
                    "sampled_target_order": _format_int_list(target_choice_list),
                    "selected_action": selected,
                    "sampled_target_action": target_selected,
                    "action_valid_unvisited": int(action_valid),
                    "action_invalid_forced": int(selected in forced_set),
                    "action_reentry": int(selected in visited),
                    "predicted_spatial_arm": spatial_arm,
                    "sampled_target_route_arm": target_route_arm,
                    "spatial_departed": int(departed),
                    "route_matches_selected_action": int(route_matches_selected),
                    "route_matches_conditioning_action": int(
                        route_matches_conditioning
                    ),
                    "top_action_arms": _format_int_list(
                        [int(z.item()) for z in top_idx]
                    ),
                    "top_action_probs": " ".join(
                        f"{float(z.item()):.4f}" for z in top_vals
                    ),
                }
            )

            if rollout_mode:
                if departed:
                    visited.add(spatial_arm)
            else:
                visited.add(target_route_arm)

    return rows


def write_eight_arm_choice_debug(
    model,
    dataset,
    device,
    out_dir,
    prefix_len,
    max_trials=16,
    hard_action_choices=True,
):
    """Write teacher-forced and autonomous per-choice diagnostics."""
    if max_trials <= 0:
        return

    model.eval()
    x = dataset.x.to(device)
    y = dataset.y.to(device)

    with torch.no_grad():
        yhat_tf, _ = model(x)
        yhat_roll = autonomous_rollout(
            model,
            x,
            prefix_len=prefix_len,
            dataset=dataset,
            hard_action_choices=hard_action_choices,
        )

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

    if not rows:
        print("No eight-arm choice rows found for debugging.")
        return

    out_path = out_dir / "eightarm_choice_debug.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
        rollout_mode=False,
    )


def rollout_eight_arm_eval(
    model,
    dataset,
    device,
    prefix_len,
    hard_action_choices=True,
):
    model.eval()
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    yhat = autonomous_rollout(
        model,
        x,
        prefix_len=prefix_len,
        dataset=dataset,
        hard_action_choices=hard_action_choices,
    )
    return eight_arm_traj_metrics(
        yhat=yhat,
        x=x,
        y=y,
        dataset=dataset,
        prefix="eightarm_rollout",
        rollout_mode=True,
    )


def _unpack_xy(batch):
    """Extract x and y from datasets returning 2, 3, or 4 tensors."""
    if len(batch) == 2:
        x, y = batch
    elif len(batch) == 3:
        x, y, _ = batch
    elif len(batch) == 4:
        x, y, _, _ = batch
    else:
        raise ValueError(f"Expected batch of length 2, 3, or 4, got {len(batch)}")
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


def autonomous_rollout(
    model,
    x,
    prefix_len,
    dataset=None,
    hard_action_choices=True,
):
    """Autoregressive rollout with optional discrete arm decisions.

    The evaluator may harden the eight arm-action outputs to a one-hot vector at
    known choice times before feeding them back. It uses only the timing of the
    choice event, never the valid-arm entries, so no visited-set information is
    leaked to the model.
    """
    model.eval()

    _, T, _ = x.shape
    u_roll = x.clone()
    preds = []

    selection_schedule = None
    a0 = a1 = None
    if (
        dataset is not None
        and hard_action_choices
        and hasattr(dataset, "valid_choice_masks")
        and hasattr(dataset, "arm_choice_start")
    ):
        selection_schedule = (
            dataset.valid_choice_masks.to(x.device).sum(dim=-1) > 0.5
        )
        a0 = int(dataset.arm_choice_start)
        a1 = int(dataset.arm_choice_end)

    with torch.no_grad():
        for t in range(T):
            yhat_prefix, _ = model(u_roll[:, : t + 1, :])
            pred_t = yhat_prefix[:, -1, :]

            if selection_schedule is not None:
                active_rows = torch.where(selection_schedule[:, t])[0]
                if len(active_rows) > 0:
                    pred_t = pred_t.clone()
                    choices = pred_t[active_rows, a0:a1].argmax(dim=-1)
                    pred_t[active_rows, a0:a1] = 0.0
                    pred_t[active_rows, a0 + choices] = 1.0

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
    hard_action_choices=True,
):
    x = dataset.x.to(device)
    y = dataset.y.to(device)

    yhat = autonomous_rollout(
        model,
        x,
        prefix_len=prefix_len,
        dataset=dataset,
        hard_action_choices=hard_action_choices,
    )

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

    if args.prefix_len < 0:
        prefix_len = int(getattr(test_ds, "rollout_prefix_len", 5))
    else:
        prefix_len = int(args.prefix_len)
    hard_action_choices = not args.soft_action_rollout

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
            prefix_len=prefix_len,
            angle_threshold_rad=args.angle_threshold_rad,
            hard_action_choices=hard_action_choices,
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
                prefix_len=prefix_len,
                hard_action_choices=hard_action_choices,
            )
        )
        write_eight_arm_choice_debug(
            model=model,
            dataset=test_ds,
            device=device,
            out_dir=out_dir,
            prefix_len=prefix_len,
            max_trials=args.debug_trials,
            hard_action_choices=hard_action_choices,
        )

    write_metrics(metrics, out_dir)

    print(f"Analysis for {args.ckpt}")
    print(f"Rollout prefix length: {prefix_len}")
    print(f"Hard action choices: {hard_action_choices}")
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

    p.add_argument(
        "--prefix-len",
        type=int,
        default=-1,
        help=(
            "Number of true input frames before autoregressive feedback. "
            "Use -1 to infer the first choice point from the dataset."
        ),
    )
    p.add_argument(
        "--soft-action-rollout",
        action="store_true",
        help=(
            "Feed raw arm-head outputs back at choice events. By default the "
            "argmax action is converted to one-hot before routing."
        ),
    )
    p.add_argument("--angle-threshold-rad", type=float, default=0.75)
    p.add_argument("--debug-trials", type=int, default=16)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
