import argparse
from pathlib import Path

import pandas as pd

# Default summary columns for the original ring / ambiguous-ring tasks.
BASE_COLS = [
    "run",
    "teacher_forced_mse",
    "rollout_mse_mean",
    "rollout_angle_error_rad",
    "prediction_horizon_steps",
    "extras/fast_weight_norm_mean",
    "extras/fast_drive_norm_mean",
]

# Task-specific columns emitted by the updated analyze.py for eight_arm_traj.
EIGHT_ARM_COLS = [
    "run",
    # Generic diagnostics
    "teacher_forced_mse",
    "rollout_mse_mean",
    "final_val_valid_choice_loss",
    "final_val_choice_loss",
    # Primary autonomous behavioral metrics. These use the model's own choices
    # and update visited history from the spatial arms it actually enters.
    "eightarm_rollout/dynamic_first_action_valid_rate",
    "eightarm_rollout/action_selection_entropy",
    "eightarm_rollout/action_selection_top1_probability",
    "eightarm_rollout/action_selection_top1_top2_margin",
    "eightarm_rollout/dynamic_action_valid_unvisited_rate",
    "eightarm_rollout/dynamic_action_invalid_forced_rate",
    "eightarm_rollout/dynamic_action_reentry_rate",
    "eightarm_rollout/dynamic_routing_departure_rate",
    "eightarm_rollout/dynamic_routing_matches_conditioning_action_rate",
    "eightarm_rollout/dynamic_routing_enters_unvisited_rate",
    "eightarm_rollout/dynamic_step_success_rate",
    "eightarm_rollout/dynamic_trial_complete_success_rate",
    "eightarm_rollout/dynamic_unique_unvisited_arms_routed_mean",
    # Teacher-forced diagnostics separate choice-memory from routing.
    "eightarm_tf/action_selection_valid_under_teacher_history_rate",
    "eightarm_tf/action_selection_entropy",
    "eightarm_tf/action_selection_top1_probability",
    "eightarm_tf/action_selection_top1_top2_margin",
    "eightarm_tf/action_selection_exact_target_arm_acc",
    "eightarm_tf/routing_exact_target_arm_acc",
    "eightarm_tf/dynamic_routing_matches_conditioning_action_rate",
    # Target-relative rollout values are retained only as secondary diagnostics.
    "eightarm_rollout/action_selection_exact_target_arm_acc",
    "eightarm_rollout/routing_exact_target_arm_acc",
    "eightarm_rollout/mse_choice_target_relative",
    # Fast-weight diagnostics
    "extras/fast_weight_norm_mean",
    "extras/fast_drive_norm_mean",
]

def infer_run_name(path: str) -> str:
    """
    Infer a useful run label from a metrics.csv path.

    Handles both:
        data/runs/<run_name>/metrics.csv
    and:
        data/runs/<run_name>/analysis/metrics.csv
    """
    p = Path(path)

    if p.name == "metrics.csv":
        if p.parent.name == "analysis":
            return p.parent.parent.name
        return p.parent.name

    return p.stem


def load_metrics(path: str) -> dict:
    df = pd.read_csv(path)

    required = {"metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = {row["metric"]: row["value"] for _, row in df.iterrows()}
    out["run"] = infer_run_name(path)
    return out


def select_columns(df: pd.DataFrame, mode: str) -> list[str]:
    if mode == "base":
        desired = BASE_COLS
    elif mode == "eight_arm":
        desired = EIGHT_ARM_COLS
    elif mode == "all":
        desired = ["run"] + [c for c in df.columns if c != "run"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return [c for c in desired if c in df.columns]


def main(args):
    rows = [load_metrics(path) for path in args.metrics]
    df = pd.DataFrame(rows)

    cols = select_columns(df, args.mode)
    df = df[cols]

    # Stable, readable ordering for settle sweeps if run names contain s0/s2/s5/s10.
    if args.sort:
        df = df.sort_values("run")

    print(df.to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved comparison to {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+", help="Paths to metrics.csv files.")
    p.add_argument("--out", default="data/runs/comparison.csv")
    p.add_argument(
        "--mode",
        choices=["base", "eight_arm", "all"],
        default="base",
        help=(
            "Which metric subset to write. Use eight_arm for eight_arm_traj, "
            "all to keep every metric emitted by analyze.py."
        ),
    )
    p.add_argument("--sort", action="store_true", help="Sort rows by run name.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
