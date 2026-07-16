# Normalized FastWave transition-memory update

This update addresses the transition-memory diagnostic in which the correct
key/value direction was present but the retrieved signal was tiny and the
`fast_to_site` bias dominated the fast pathway.

## Changes

- Adds `--fast-patch-norm {none,l2}`.
- Adds `--no-fast-readout-bias`.
- With `l2`, FastWave uses normalized local patches for:
  - the read query `q`,
  - the transition key,
  - the stored value.
- Records both raw and normalized key/value patches.
- Adds query-onset causal ablations:
  - erase fast memory,
  - shuffle fast memory across trials,
  - disable the fast drive.
- Uses new `*_norm_*` run names so old checkpoints are not reused.

Defaults preserve old checkpoint behavior (`none`, bias enabled). The new run
scripts explicitly select `l2` and no bias.

## Install

From the project root:

```bash
unzip eight_arm_transition_memory_normalized_update.zip
cp eight_arm_transition_memory_normalized_update/src/models.py src/models.py
cp eight_arm_transition_memory_normalized_update/src/tasks.py src/tasks.py
cp eight_arm_transition_memory_normalized_update/src/train.py src/train.py
cp eight_arm_transition_memory_normalized_update/scripts/* scripts/
chmod +x scripts/*.sh scripts/analyze_transition_*.py scripts/check_transition_tasks.py scripts/summarize_transition_recall.py
```

## Check task construction

```bash
python scripts/check_transition_tasks.py
```

## Run the normalized two-arm sanity test

```bash
DEVICE=cuda EPOCHS=100 N_TRAIN=2048 N_VAL=256 N_TEST=1024 \
./scripts/run_two_arm_sanity.sh
```

Outputs:

```text
data/runs/transition_sanity_norm_run_level.csv
data/runs/transition_sanity_norm_summary.csv
```

## Run normalized independent transition pairs

```bash
DEVICE=cuda EPOCHS=100 N_TRAIN=4096 N_VAL=512 N_TEST=1024 \
./scripts/run_independent_transition_pairs.sh
```

Outputs:

```text
data/runs/transition_independent_norm_run_level.csv
data/runs/transition_independent_norm_summary.csv
```

FastWave runs also save:

```text
transition_ablations/metrics.csv
```

## Inspect one FastWave checkpoint

```bash
python scripts/analyze_transition_dynamics.py \
  --ckpt data/runs/transition_independent_norm_fastwave_p3_d0_q2_seed42/best.pt \
  --device cpu \
  --trial 0
```

Key outputs:

```text
transition_dynamics/dynamics_trace.csv
transition_dynamics/pair_similarity.csv
transition_dynamics/key_and_retrieval_similarity.png
transition_dynamics/fast_weight_progression.png
transition_dynamics/summary.csv
```

## Hyperparameters

The scripts start with:

```text
lambda = 1.0
eta = 0.5
beta = 0.25
fast_patch_norm = l2
fast_to_site bias = disabled
```

Override beta or eta through environment variables, for example:

```bash
BETA=0.10 ETA=0.50 DEVICE=cuda ./scripts/run_independent_transition_pairs.sh
```
