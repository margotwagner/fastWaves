import torch
from torch.utils.data import Dataset


def ring_distance(grid: torch.Tensor, pos: torch.Tensor, n_space: int) -> torch.Tensor:
    """Circular distance between grid [N] and positions [B]."""
    d = torch.abs(grid[None, :] - pos[:, None])
    return torch.minimum(d, n_space - d)


def make_ring_bump_sequence(
    batch_size: int,
    seq_len: int,
    n_space: int,
    velocity: int = 1,
    sigma: float = 3.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return moving Gaussian bumps on a ring: [B, T, N]."""
    starts = torch.randint(0, n_space, (batch_size,), device=device)
    times = torch.arange(seq_len, device=device)
    positions = (starts[:, None] + velocity * times[None, :]) % n_space

    grid = torch.arange(n_space, device=device)
    frames = []
    for t in range(seq_len):
        d = ring_distance(grid, positions[:, t], n_space)
        bump = torch.exp(-(d**2) / (2 * sigma**2))
        frames.append(bump)
    return torch.stack(frames, dim=1)


class RingBumpDataset(Dataset):
    """One-step prediction dataset: input x_t, target x_{t+1}."""

    def __init__(
        self,
        n_samples: int = 1024,
        seq_len: int = 64,
        n_space: int = 64,
        velocity: int = 1,
        sigma: float = 3.0,
        seed: int = 0,
    ):
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        seq = make_ring_bump_sequence(n_samples, seq_len + 1, n_space, velocity, sigma)
        torch.random.set_rng_state(generator_state)

        self.x = seq[:, :-1, :].float()
        self.y = seq[:, 1:, :].float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class AmbiguousDirectionRingDataset(torch.utils.data.Dataset):
    """
    Ring bump sequences with randomly chosen direction per sequence.

    The same current position can imply different futures depending on
    recent motion history.

    x: [T-1, N]
    y: [T-1, N]
    """

    def __init__(
        self,
        n_samples=512,
        seq_len=40,
        n_space=32,
        velocity=1,
        sigma=2.0,
        seed=42,
    ):
        super().__init__()

        g = torch.Generator().manual_seed(seed)

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_space = n_space
        self.velocity = velocity
        self.sigma = sigma

        starts = torch.randint(0, n_space, (n_samples,), generator=g)

        # direction is randomly +1 or -1 per sequence
        directions = torch.randint(0, 2, (n_samples,), generator=g)
        directions = directions * 2 - 1  # 0/1 -> -1/+1

        t = torch.arange(seq_len)
        positions = (
            starts[:, None] + directions[:, None] * velocity * t[None, :]
        ) % n_space

        grid = torch.arange(n_space)

        seq = []
        for tt in range(seq_len):
            d = torch.abs(grid[None, :] - positions[:, tt : tt + 1])
            d = torch.minimum(d, n_space - d)
            bump = torch.exp(-(d**2) / (2 * sigma**2))
            seq.append(bump)

        seq = torch.stack(seq, dim=1).float()

        self.full_seq = seq
        self.x = seq[:, :-1, :]
        self.y = seq[:, 1:, :]
        self.directions = directions

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EightArmRadialMazeDataset(torch.utils.data.Dataset):
    """
    Symbolic eight-arm working-memory task.

    Trial structure:
      1. Forced phase: four unique arms are presented one at a time.
      2. Optional settle phase: blank/no-op timesteps after forced phase.
      3. Choice phase: all arms are available; target is unvisited arms.

    x, y are both [T, n_space]. Targets use channels 0:8.
    """

    def __init__(
        self,
        n_samples: int = 1024,
        seq_len: int = 16,
        n_space: int = 32,
        n_arms: int = 8,
        n_forced: int = 4,
        n_choice=None,
        settle_steps: int = 2,
        expose_visited_memory: bool = False,
        include_availability: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        if n_arms != 8:
            raise ValueError("This implementation assumes n_arms=8.")
        if n_forced >= n_arms:
            raise ValueError("n_forced must be smaller than n_arms.")
        if n_space < 12:
            raise ValueError(
                "n_space must be at least 12 for arm channels + phase cues."
            )
        if expose_visited_memory and n_space < 20:
            raise ValueError("expose_visited_memory=True requires n_space >= 20.")
        if include_availability and n_space < 28:
            include_availability = False

        self.n_samples = n_samples
        self.n_space = n_space
        self.n_arms = n_arms
        self.n_forced = n_forced
        self.n_choice = n_arms - n_forced if n_choice is None else n_choice
        self.settle_steps = settle_steps
        self.expose_visited_memory = expose_visited_memory
        self.include_availability = include_availability

        natural_len = n_forced + settle_steps + self.n_choice
        self.seq_len = seq_len if seq_len is not None else natural_len
        if self.seq_len < natural_len:
            raise ValueError(
                f"seq_len={self.seq_len} is too short; need at least {natural_len}."
            )

        g = torch.Generator().manual_seed(seed)
        xs = torch.zeros(n_samples, self.seq_len, n_space)
        ys = torch.zeros(n_samples, self.seq_len, n_space)
        forced_orders = torch.empty(n_samples, n_forced, dtype=torch.long)
        remaining_masks = torch.zeros(n_samples, n_arms)

        for i in range(n_samples):
            perm = torch.randperm(n_arms, generator=g)
            forced = perm[:n_forced]
            remaining = perm[n_forced:]
            forced_orders[i] = forced
            remaining_mask = torch.zeros(n_arms)
            remaining_mask[remaining] = 1.0
            remaining_masks[i] = remaining_mask

            visited_mask = torch.zeros(n_arms)
            t = 0

            for arm in forced:
                arm = int(arm.item())
                xs[i, t, arm] = 1.0
                xs[i, t, 8] = 1.0
                xs[i, t, 10] = 1.0
                if self.include_availability:
                    xs[i, t, 20 + arm] = 1.0
                if self.expose_visited_memory:
                    xs[i, t, 12:20] = visited_mask

                ys[i, t, arm] = 1.0
                visited_mask[arm] = 1.0
                t += 1

            for _ in range(settle_steps):
                xs[i, t, 11] = 1.0
                if self.expose_visited_memory:
                    xs[i, t, 12:20] = visited_mask
                t += 1

            still_remaining = remaining_mask.clone()
            for _ in range(self.n_choice):
                xs[i, t, 9] = 1.0
                if self.include_availability:
                    xs[i, t, 20:28] = 1.0
                if self.expose_visited_memory:
                    xs[i, t, 12:20] = visited_mask
                ys[i, t, :8] = still_remaining

                valid = torch.where(still_remaining > 0)[0]
                if len(valid) > 0:
                    chosen = valid[torch.randint(len(valid), (1,), generator=g).item()]
                    still_remaining[chosen] = 0.0
                    visited_mask[chosen] = 1.0
                t += 1

            while t < self.seq_len:
                xs[i, t, 11] = 1.0
                if self.expose_visited_memory:
                    xs[i, t, 12:20] = visited_mask
                t += 1

        self.x = xs.float()
        self.y = ys.float()
        self.forced_orders = forced_orders
        self.remaining_masks = remaining_masks

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EightArmTrajectoryDataset(torch.utils.data.Dataset):
    """
    Trajectory version of the eight-arm radial-maze working-memory task.

    This is closer to the experimental paradigm than the symbolic task because
    each arm visit is represented as a spatial trajectory:

        center -> inner arm -> outer arm -> reward -> outer arm -> inner arm -> center

    Trial structure:
      1. Forced phase: the animal/network is driven through four unique arms.
      2. Settle phase: it sits at the center with no phase-specific evidence.
      3. Choice phase: it must generate trajectories through the four unvisited arms.

    x and y are one-step prediction tensors with shape [T, n_space]. The target
    y[t] is the next spatial/cue state after x[t]. This is compatible with the
    existing ring-bump training loop.

    Why the task requires memory:
      The center state after the forced phase is visually identical across trials,
      but the correct next choice arm depends on which arms were already visited.
      Thus, the hidden state or fast weights must carry trial-specific memory.

    Recommended n_space:
      n_space >= 32 with arm_len=3.

    Channel layout for default n_arms=8, arm_len=3:
      0                         center position
      1:25                      arm trajectory positions, grouped by arm
      25                        forced-phase cue
      26                        choice-phase cue
      27                        reward/update cue
      28                        settle cue
      29                        outbound cue
      30                        inbound cue
      31                        center/no-arm cue
      remaining channels         unused

    Position index for arm a, depth d in {0, 1, 2}:
      1 + a * arm_len + d

    The reward location is d = arm_len - 1.
    """

    def __init__(
        self,
        n_samples: int = 1024,
        seq_len: int = 72,
        n_space: int = 32,
        n_arms: int = 8,
        n_forced: int = 4,
        arm_len: int = 3,
        settle_steps: int = 2,
        reward_hold_steps: int = 1,
        center_hold_steps: int = 0,
        choice_order: str = "random",
        seed: int = 0,
    ):
        super().__init__()
        if n_arms != 8:
            raise ValueError("This implementation assumes n_arms=8.")
        if n_forced >= n_arms:
            raise ValueError("n_forced must be smaller than n_arms.")
        if arm_len < 2:
            raise ValueError("arm_len must be at least 2.")
        if choice_order not in {"random", "ascending"}:
            raise ValueError("choice_order must be 'random' or 'ascending'.")

        self.n_samples = n_samples
        self.n_space = n_space
        self.n_arms = n_arms
        self.n_forced = n_forced
        self.n_choice = n_arms - n_forced
        self.arm_len = arm_len
        self.settle_steps = settle_steps
        self.reward_hold_steps = reward_hold_steps
        self.center_hold_steps = center_hold_steps
        self.choice_order = choice_order

        self.center_idx = 0
        self.n_pos = 1 + n_arms * arm_len
        self.cue_forced = self.n_pos
        self.cue_choice = self.n_pos + 1
        self.cue_reward = self.n_pos + 2
        self.cue_settle = self.n_pos + 3
        self.cue_outbound = self.n_pos + 4
        self.cue_inbound = self.n_pos + 5
        self.cue_center = self.n_pos + 6
        self.min_n_space = self.n_pos + 7
        if n_space < self.min_n_space:
            raise ValueError(
                f"n_space={n_space} is too small. Need at least {self.min_n_space} "
                f"for n_arms={n_arms}, arm_len={arm_len}. Use --n-space {self.min_n_space} or larger."
            )

        # Frames per arm visit. The start center is included for every visit.
        # Visit path: center, outbound depths, reward hold repeats, inbound depths, center.
        self.visit_len = 1 + arm_len + max(0, reward_hold_steps - 1) + (arm_len - 1) + 1
        natural_frames = (n_arms * self.visit_len) + settle_steps
        # Dataset returns x=frames[:-1], y=frames[1:], so it needs natural_frames - 1 examples.
        self.natural_seq_len = natural_frames - 1
        if seq_len is None:
            seq_len = self.natural_seq_len
        if seq_len < self.natural_seq_len:
            raise ValueError(
                f"seq_len={seq_len} is too short. Need at least {self.natural_seq_len}. "
                f"Try --seq-len {self.natural_seq_len}."
            )
        self.seq_len = seq_len

        g = torch.Generator().manual_seed(seed)
        xs = torch.zeros(n_samples, self.seq_len, n_space)
        ys = torch.zeros(n_samples, self.seq_len, n_space)
        forced_orders = torch.empty(n_samples, n_forced, dtype=torch.long)
        choice_orders = torch.empty(n_samples, self.n_choice, dtype=torch.long)
        full_orders = torch.empty(n_samples, n_arms, dtype=torch.long)

        for i in range(n_samples):
            perm = torch.randperm(n_arms, generator=g)
            forced = perm[:n_forced]
            remaining = perm[n_forced:]
            if choice_order == "ascending":
                remaining = torch.sort(remaining).values
            # If choice_order == random, the remaining part of the permutation is already random.

            forced_orders[i] = forced
            choice_orders[i] = remaining
            full_order = torch.cat([forced, remaining], dim=0)
            full_orders[i] = full_order

            frames = []
            for arm in forced.tolist():
                frames.extend(self._visit_frames(arm, phase="forced"))
                frames.extend(
                    self._center_hold_frames(center_hold_steps, phase="forced")
                )

            # Brief no-op/settling period at center before choice rollout.
            for _ in range(settle_steps):
                frames.append(
                    self._make_frame(
                        pos_idx=self.center_idx, phase="settle", direction="center"
                    )
                )

            for arm in remaining.tolist():
                frames.extend(self._visit_frames(arm, phase="choice"))
                frames.extend(
                    self._center_hold_frames(center_hold_steps, phase="choice")
                )

            frames = torch.stack(frames, dim=0)
            x = frames[:-1]
            y = frames[1:]
            xs[i, : x.shape[0], :] = x
            ys[i, : y.shape[0], :] = y

            # If seq_len is longer than the natural task, pad with center-settle states.
            if x.shape[0] < self.seq_len:
                pad_frame = self._make_frame(
                    pos_idx=self.center_idx, phase="settle", direction="center"
                )
                xs[i, x.shape[0] :, :] = pad_frame
                ys[i, y.shape[0] :, :] = pad_frame

        self.x = xs.float()
        self.y = ys.float()
        self.forced_orders = forced_orders
        self.choice_orders = choice_orders
        self.full_orders = full_orders

    def arm_pos_idx(self, arm: int, depth: int) -> int:
        """Return channel index for an arm-depth position."""
        return 1 + arm * self.arm_len + depth

    def _make_frame(self, pos_idx: int, phase: str, direction: str) -> torch.Tensor:
        frame = torch.zeros(self.n_space)
        frame[pos_idx] = 1.0

        if phase == "forced":
            frame[self.cue_forced] = 1.0
        elif phase == "choice":
            frame[self.cue_choice] = 1.0
        elif phase == "settle":
            frame[self.cue_settle] = 1.0
        else:
            raise ValueError(f"Unknown phase: {phase}")

        if direction == "outbound":
            frame[self.cue_outbound] = 1.0
        elif direction == "inbound":
            frame[self.cue_inbound] = 1.0
        elif direction == "center":
            frame[self.cue_center] = 1.0
        elif direction == "reward":
            frame[self.cue_reward] = 1.0
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return frame

    def _visit_frames(self, arm: int, phase: str) -> list[torch.Tensor]:
        frames = []
        frames.append(
            self._make_frame(pos_idx=self.center_idx, phase=phase, direction="center")
        )

        # Outbound path from inner arm to reward site.
        for depth in range(self.arm_len):
            direction = "reward" if depth == self.arm_len - 1 else "outbound"
            frames.append(
                self._make_frame(
                    pos_idx=self.arm_pos_idx(arm, depth),
                    phase=phase,
                    direction=direction,
                )
            )

        # Optional dwell at reward site, analogous to reward consumption/pause.
        reward_idx = self.arm_pos_idx(arm, self.arm_len - 1)
        for _ in range(max(0, self.reward_hold_steps - 1)):
            frames.append(
                self._make_frame(pos_idx=reward_idx, phase=phase, direction="reward")
            )

        # Inbound path back to center. Do not repeat reward depth.
        for depth in reversed(range(self.arm_len - 1)):
            frames.append(
                self._make_frame(
                    pos_idx=self.arm_pos_idx(arm, depth),
                    phase=phase,
                    direction="inbound",
                )
            )

        frames.append(
            self._make_frame(pos_idx=self.center_idx, phase=phase, direction="center")
        )
        return frames

    def _center_hold_frames(self, n_steps: int, phase: str) -> list[torch.Tensor]:
        return [
            self._make_frame(pos_idx=self.center_idx, phase=phase, direction="center")
            for _ in range(n_steps)
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EightArmBumpTrajectoryDataset(torch.utils.data.Dataset):
    """
    Smooth-bump trajectory version of the eight-arm radial-maze task with an
    auxiliary 8-channel arm-choice head.

    First n_pos channels are a smooth spatial bump over the radial maze graph.
    Cue channels remain binary. Extra arm-choice channels explicitly supervise
    which arm is being traversed after a center->arm departure, without leaking
    the answer at the center before departure.
    """

    def __init__(
        self,
        n_samples: int = 1024,
        seq_len: int = 72,
        n_space: int = 40,
        n_arms: int = 8,
        n_forced: int = 4,
        arm_len: int = 3,
        settle_steps: int = 2,
        reward_hold_steps: int = 1,
        center_hold_steps: int = 0,
        choice_order: str = "random",
        bump_sigma: float = 0.75,
        forced_departure_weight: float = 3.0,
        choice_departure_weight: float = 10.0,
        arm_choice_weight: float = 50.0,
        seed: int = 0,
    ):
        super().__init__()
        if n_arms != 8:
            raise ValueError("This implementation assumes n_arms=8.")
        if n_forced >= n_arms:
            raise ValueError("n_forced must be smaller than n_arms.")
        if arm_len < 2:
            raise ValueError("arm_len must be at least 2.")
        if choice_order not in {"random", "ascending"}:
            raise ValueError("choice_order must be 'random' or 'ascending'.")
        if bump_sigma <= 0:
            raise ValueError("bump_sigma must be positive.")

        self.n_samples = n_samples
        self.n_space = n_space
        self.n_arms = n_arms
        self.n_forced = n_forced
        self.n_choice = n_arms - n_forced
        self.arm_len = arm_len
        self.settle_steps = settle_steps
        self.reward_hold_steps = reward_hold_steps
        self.center_hold_steps = center_hold_steps
        self.choice_order = choice_order
        self.bump_sigma = float(bump_sigma)
        self.forced_departure_weight = float(forced_departure_weight)
        self.choice_departure_weight = float(choice_departure_weight)
        self.arm_choice_weight = float(arm_choice_weight)

        self.center_idx = 0
        self.n_pos = 1 + n_arms * arm_len
        self.cue_forced = self.n_pos
        self.cue_choice = self.n_pos + 1
        self.cue_reward = self.n_pos + 2
        self.cue_settle = self.n_pos + 3
        self.cue_outbound = self.n_pos + 4
        self.cue_inbound = self.n_pos + 5
        self.cue_center = self.n_pos + 6

        # New explicit arm-choice/action head.
        # For defaults: n_pos=25, cues=25:32, arm choice=32:40.
        self.arm_choice_start = self.n_pos + 7
        self.arm_choice_end = self.arm_choice_start + n_arms
        self.min_n_space = self.arm_choice_end
        if n_space < self.min_n_space:
            raise ValueError(
                f"n_space={n_space} is too small. Need at least {self.min_n_space} "
                f"for n_arms={n_arms}, arm_len={arm_len}. Use --n-space {self.min_n_space} or larger."
            )

        self.pos_dist = self._make_radial_graph_distance_matrix()

        self.visit_len = 1 + arm_len + max(0, reward_hold_steps - 1) + (arm_len - 1) + 1
        natural_frames = (n_arms * self.visit_len) + settle_steps
        self.natural_seq_len = natural_frames - 1

        if seq_len is None:
            seq_len = self.natural_seq_len
        if seq_len < self.natural_seq_len:
            raise ValueError(
                f"seq_len={seq_len} is too short. Need at least {self.natural_seq_len}. "
                f"Try --seq-len {self.natural_seq_len}."
            )
        self.seq_len = seq_len

        g = torch.Generator().manual_seed(seed)
        xs = torch.zeros(n_samples, self.seq_len, n_space)
        ys = torch.zeros(n_samples, self.seq_len, n_space)
        forced_orders = torch.empty(n_samples, n_forced, dtype=torch.long)
        choice_orders = torch.empty(n_samples, self.n_choice, dtype=torch.long)
        full_orders = torch.empty(n_samples, n_arms, dtype=torch.long)

        for i in range(n_samples):
            perm = torch.randperm(n_arms, generator=g)
            forced = perm[:n_forced]
            remaining = perm[n_forced:]
            if choice_order == "ascending":
                remaining = torch.sort(remaining).values

            forced_orders[i] = forced
            choice_orders[i] = remaining
            full_orders[i] = torch.cat([forced, remaining], dim=0)

            frames = []
            for arm in forced.tolist():
                frames.extend(self._visit_frames(arm, phase="forced"))
                frames.extend(self._center_hold_frames(center_hold_steps, phase="forced"))

            for _ in range(settle_steps):
                frames.append(
                    self._make_frame(
                        pos_idx=self.center_idx,
                        phase="settle",
                        direction="center",
                        arm_choice=None,
                    )
                )

            for arm in remaining.tolist():
                frames.extend(self._visit_frames(arm, phase="choice"))
                frames.extend(self._center_hold_frames(center_hold_steps, phase="choice"))

            frames = torch.stack(frames, dim=0)
            x = frames[:-1]
            y = frames[1:]
            xs[i, : x.shape[0], :] = x
            ys[i, : y.shape[0], :] = y

            if x.shape[0] < self.seq_len:
                pad_frame = self._make_frame(
                    pos_idx=self.center_idx,
                    phase="settle",
                    direction="center",
                    arm_choice=None,
                )
                xs[i, x.shape[0] :, :] = pad_frame
                ys[i, y.shape[0] :, :] = pad_frame

        self.x = xs.float()
        self.y = ys.float()
        self.loss_weights = self._make_elementwise_loss_weights()
        self.forced_orders = forced_orders
        self.choice_orders = choice_orders
        self.full_orders = full_orders

    def arm_pos_idx(self, arm: int, depth: int) -> int:
        return 1 + arm * self.arm_len + depth

    def _pos_to_arm_depth(self, pos_idx: int):
        if pos_idx == self.center_idx:
            return -1, -1
        z = pos_idx - 1
        return z // self.arm_len, z % self.arm_len

    def _make_radial_graph_distance_matrix(self) -> torch.Tensor:
        D = torch.zeros(self.n_pos, self.n_pos)
        for i in range(self.n_pos):
            ai, di = self._pos_to_arm_depth(i)
            for j in range(self.n_pos):
                aj, dj = self._pos_to_arm_depth(j)
                if i == j:
                    dist = 0
                elif i == self.center_idx:
                    dist = dj + 1
                elif j == self.center_idx:
                    dist = di + 1
                elif ai == aj:
                    dist = abs(di - dj)
                else:
                    dist = (di + 1) + (dj + 1)
                D[i, j] = float(dist)
        return D

    def _position_bump(self, pos_idx: int) -> torch.Tensor:
        d = self.pos_dist[pos_idx]
        bump = torch.exp(-0.5 * (d / self.bump_sigma) ** 2)
        return bump / bump.max().clamp_min(1e-8)

    def _make_frame(
        self,
        pos_idx: int,
        phase: str,
        direction: str,
        arm_choice: int | None = None,
    ) -> torch.Tensor:
        frame = torch.zeros(self.n_space)
        frame[: self.n_pos] = self._position_bump(pos_idx)

        if phase == "forced":
            frame[self.cue_forced] = 1.0
        elif phase == "choice":
            frame[self.cue_choice] = 1.0
        elif phase == "settle":
            frame[self.cue_settle] = 1.0
        else:
            raise ValueError(f"Unknown phase: {phase}")

        if direction == "outbound":
            frame[self.cue_outbound] = 1.0
        elif direction == "inbound":
            frame[self.cue_inbound] = 1.0
        elif direction == "center":
            frame[self.cue_center] = 1.0
        elif direction == "reward":
            frame[self.cue_reward] = 1.0
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Do not set this on the center frame before departure; that would leak
        # the correct next arm. Set it only once the trajectory is already on the selected arm.
        if arm_choice is not None:
            frame[self.arm_choice_start + int(arm_choice)] = 1.0

        return frame

    def _visit_frames(self, arm: int, phase: str) -> list[torch.Tensor]:
        frames = [
            self._make_frame(
                pos_idx=self.center_idx,
                phase=phase,
                direction="center",
                arm_choice=None,
            )
        ]

        for depth in range(self.arm_len):
            direction = "reward" if depth == self.arm_len - 1 else "outbound"
            frames.append(
                self._make_frame(
                    pos_idx=self.arm_pos_idx(arm, depth),
                    phase=phase,
                    direction=direction,
                    arm_choice=arm,
                )
            )

        reward_idx = self.arm_pos_idx(arm, self.arm_len - 1)
        for _ in range(max(0, self.reward_hold_steps - 1)):
            frames.append(
                self._make_frame(
                    pos_idx=reward_idx,
                    phase=phase,
                    direction="reward",
                    arm_choice=arm,
                )
            )

        for depth in reversed(range(self.arm_len - 1)):
            frames.append(
                self._make_frame(
                    pos_idx=self.arm_pos_idx(arm, depth),
                    phase=phase,
                    direction="inbound",
                    arm_choice=arm,
                )
            )

        frames.append(
            self._make_frame(
                pos_idx=self.center_idx,
                phase=phase,
                direction="center",
                arm_choice=None,
            )
        )
        return frames

    def _center_hold_frames(self, n_steps: int, phase: str) -> list[torch.Tensor]:
        return [
            self._make_frame(
                pos_idx=self.center_idx,
                phase=phase,
                direction="center",
                arm_choice=None,
            )
            for _ in range(n_steps)
        ]

    def _make_elementwise_loss_weights(self) -> torch.Tensor:
        weights = torch.ones_like(self.y)

        x_pos = self.x[:, :, : self.n_pos].argmax(dim=-1)
        y_pos = self.y[:, :, : self.n_pos].argmax(dim=-1)
        center_to_arm = (x_pos == self.center_idx) & (y_pos != self.center_idx)
        forced_departure = center_to_arm & (self.x[:, :, self.cue_forced] > 0.5)
        choice_departure = center_to_arm & (self.x[:, :, self.cue_choice] > 0.5)

        weights[forced_departure, :] = self.forced_departure_weight
        weights[choice_departure, :] = self.choice_departure_weight

        # The arm-choice head is the direct supervision for the branch decision.
        arm_choice_active = self.y[:, :, self.arm_choice_start : self.arm_choice_end].sum(dim=-1) > 0
        weights[arm_choice_active, self.arm_choice_start : self.arm_choice_end] = self.arm_choice_weight

        return weights

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.loss_weights[idx]


def build_dataset(args, n_samples, seed):
    if args.task == "ring":
        return RingBumpDataset(
            n_samples=n_samples,
            seq_len=args.seq_len,
            n_space=args.n_space,
            velocity=args.velocity,
            sigma=args.sigma,
            seed=seed,
        )

    if args.task == "ambiguous_ring":
        return AmbiguousDirectionRingDataset(
            n_samples=n_samples,
            seq_len=args.seq_len,
            n_space=args.n_space,
            velocity=args.velocity,
            sigma=args.sigma,
            seed=seed,
        )

    if args.task == "eight_arm":
        return EightArmRadialMazeDataset(
            n_samples=n_samples,
            seq_len=args.seq_len,
            n_space=args.n_space,
            settle_steps=getattr(args, "settle_steps", 2),
            expose_visited_memory=getattr(args, "expose_visited_memory", False),
            include_availability=getattr(args, "include_availability", True),
            seed=seed,
        )

    if args.task == "eight_arm_traj":
        return EightArmTrajectoryDataset(
            n_samples=n_samples,
            seq_len=args.seq_len,
            n_space=args.n_space,
            n_arms=getattr(args, "n_arms", 8),
            n_forced=getattr(args, "n_forced", 4),
            arm_len=getattr(args, "arm_len", 3),
            settle_steps=getattr(args, "settle_steps", 2),
            reward_hold_steps=getattr(args, "reward_hold_steps", 1),
            center_hold_steps=getattr(args, "center_hold_steps", 0),
            choice_order=getattr(args, "choice_order", "random"),
            seed=seed,
        )

    if args.task == "eight_arm_bump_traj":
        return EightArmBumpTrajectoryDataset(
            n_samples=n_samples,
            seq_len=args.seq_len,
            n_space=args.n_space,
            n_arms=getattr(args, "n_arms", 8),
            n_forced=getattr(args, "n_forced", 4),
            arm_len=getattr(args, "arm_len", 3),
            settle_steps=getattr(args, "settle_steps", 2),
            reward_hold_steps=getattr(args, "reward_hold_steps", 1),
            center_hold_steps=getattr(args, "center_hold_steps", 0),
            choice_order=getattr(args, "choice_order", "random"),
            bump_sigma=getattr(args, "bump_sigma", 0.75),
            forced_departure_weight=getattr(args, "forced_departure_weight", 3.0),
            choice_departure_weight=getattr(args, "choice_departure_weight", 10.0),
            arm_choice_weight=getattr(args, "arm_choice_weight", 50.0),
            seed=seed,
        )

    raise ValueError(f"Unknown task: {args.task}")
