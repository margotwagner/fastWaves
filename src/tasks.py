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
        action_hold_steps: int = 3,
        choice_order: str = "random",
        choice_objective: str = "commitment",
        bump_sigma: float = 0.75,
        forced_departure_weight: float = 3.0,
        choice_departure_weight: float = 10.0,
        arm_choice_weight: float = 50.0,
        routing_weight: float = 20.0,
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
        if choice_objective not in {"exact", "valid_set", "commitment"}:
            raise ValueError(
                "choice_objective must be 'exact', 'valid_set', or 'commitment'."
            )
        if action_hold_steps < 1:
            raise ValueError("action_hold_steps must be at least 1.")
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
        self.action_hold_steps = int(action_hold_steps)
        self.choice_order = choice_order
        self.choice_objective = choice_objective
        self.bump_sigma = float(bump_sigma)
        self.forced_departure_weight = float(forced_departure_weight)
        self.choice_departure_weight = float(choice_departure_weight)
        self.arm_choice_weight = float(arm_choice_weight)
        self.routing_weight = float(routing_weight)

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

        # Per arm visit frames:
        #   center(no action) -> action_hold_steps x center(action cue)
        #   -> outbound depths -> optional reward holds -> inbound depths
        #   -> center(no action)
        #
        # Multiple action-cued center frames give second-order wave dynamics time
        # to turn a selected action into directed motion before departure.
        self.visit_len = (
            1
            + self.action_hold_steps
            + arm_len
            + max(0, reward_hold_steps - 1)
            + (arm_len - 1)
            + 1
        )
        self.selection_to_routing_offset = self.action_hold_steps
        natural_frames = (n_arms * self.visit_len) + settle_steps
        self.natural_seq_len = natural_frames - 1

        # True inputs are supplied through the first choice-phase center frame.
        # The model then predicts the arm action, which is fed back on the next step.
        self.first_choice_frame = (n_forced * self.visit_len) + settle_steps
        self.rollout_prefix_len = self.first_choice_frame + 1

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

        # Supervision metadata. These masks are never included in x and therefore
        # are never shown to the model. At each choice event, ones mark every arm
        # that remains unvisited in the teacher-forced history.
        valid_choice_masks = torch.zeros(
            n_samples, self.seq_len, n_arms, dtype=torch.float32
        )

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

            # Find the four memory-dependent action-selection transitions:
            # center(no action) -> center(action). The sampled route provides one
            # teacher-forced continuation, but every currently unvisited arm is
            # accepted by the valid-set choice loss.
            x_pos = x[:, : self.n_pos].argmax(dim=-1)
            y_pos = y[:, : self.n_pos].argmax(dim=-1)
            x_action_active = (
                x[:, self.arm_choice_start : self.arm_choice_end].sum(dim=-1) > 0.5
            )
            y_action_active = (
                y[:, self.arm_choice_start : self.arm_choice_end].sum(dim=-1) > 0.5
            )
            action_selection = (
                (x_pos == self.center_idx)
                & (~x_action_active)
                & (y_pos == self.center_idx)
                & y_action_active
                & (y[:, self.cue_choice] > 0.5)
            )
            selection_times = torch.where(action_selection)[0]
            if len(selection_times) != self.n_choice:
                raise RuntimeError(
                    f"Expected {self.n_choice} choice events, found "
                    f"{len(selection_times)} for trial {i}."
                )
            for j, t in enumerate(selection_times.tolist()):
                valid_choice_masks[i, t, remaining[j:]] = 1.0

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
        self.valid_choice_masks = valid_choice_masks.float()

        # In exact mode the valid masks remain available for evaluation, but the
        # training loss uses the sampled one-hot target. In valid_set mode they
        # drive the partial-label choice loss.
        if self.choice_objective in {"valid_set", "commitment"}:
            self.choice_loss_masks = self.valid_choice_masks.clone()
        else:
            self.choice_loss_masks = torch.zeros_like(self.valid_choice_masks)

        self.forced_orders = forced_orders
        self.choice_orders = choice_orders
        self.full_orders = full_orders
        self.loss_weights = self._make_elementwise_loss_weights()

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

        # Do not set this on the first center frame before selection; that would
        # leak the answer. After selection, the chosen action is held at center
        # and then carried along the arm trajectory.
        if arm_choice is not None:
            frame[self.arm_choice_start + int(arm_choice)] = 1.0

        return frame

    def _visit_frames(self, arm: int, phase: str) -> list[torch.Tensor]:
        # The first center frame has no action cue and therefore requires the
        # network to select an arm from memory. It is followed by one or more
        # center frames carrying the selected action. Those preparation frames
        # do not reveal the valid set; they only hold the model's chosen action
        # long enough for the recurrent dynamics to prepare the departure.
        frames = [
            self._make_frame(
                pos_idx=self.center_idx,
                phase=phase,
                direction="center",
                arm_choice=None,
            )
        ]
        frames.extend(
            self._make_frame(
                pos_idx=self.center_idx,
                phase=phase,
                direction="center",
                arm_choice=arm,
            )
            for _ in range(self.action_hold_steps)
        )

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

        # Routing transition: center(with chosen-arm cue) -> first spatial position on that arm.
        # This is the step that was collapsing to center in autonomous rollout.  Upweight
        # only the spatial bump channels so the model is explicitly trained to route the
        # bump out of center once an arm-choice cue is available.
        x_arm_choice_active = (
            self.x[:, :, self.arm_choice_start : self.arm_choice_end].sum(dim=-1) > 0
        )
        routing_transition = center_to_arm & x_arm_choice_active
        weights[routing_transition, : self.n_pos] = self.routing_weight

        # The arm-choice head is supervised throughout a sampled arm trajectory.
        arm_choice_active = (
            self.y[:, :, self.arm_choice_start : self.arm_choice_end].sum(dim=-1) > 0
        )
        b_idx, t_idx = torch.where(arm_choice_active)
        weights[
            b_idx,
            t_idx,
            self.arm_choice_start : self.arm_choice_end,
        ] = self.arm_choice_weight

        if self.choice_objective in {"valid_set", "commitment"}:
            # At memory-dependent selection events, several arms are correct.
            # Remove the arbitrary sampled one-hot MSE only at those events; the
            # set-valued loss in train.py replaces it. Routing and action
            # persistence remain one-hot supervised after an action is present.
            action_selection = self.valid_choice_masks.sum(dim=-1) > 0.5
            b_idx, t_idx = torch.where(action_selection)
            weights[
                b_idx,
                t_idx,
                self.arm_choice_start : self.arm_choice_end,
            ] = 0.0

        return weights

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.loss_weights[idx],
            self.choice_loss_masks[idx],
        )



class EightArmSuccessorRecallDataset(torch.utils.data.Dataset):
    """Trial-specific successor recall on an eight-arm radial maze.

    A trial presents a random ordered sequence of distinct arms, waits through
    a delay, then presents one nonterminal arm as a query. The model must report
    the arm that followed the query in that episode.

    The query is held for at least two timesteps. This is intentional: the first
    query frame establishes a query-shaped recurrent state; on the next frame a
    transition fast-weight model can use that state to retrieve the successor.

    Channel layout matches ``EightArmBumpTrajectoryDataset``:
      0:n_pos      radial-maze spatial bump
      n_pos        encoding/forced cue (also the fast-write gate)
      n_pos + 1    query cue
      n_pos + 2    reward cue (unused here, retained for compatibility)
      n_pos + 3    delay/settle cue
      n_pos + 4    outbound cue (unused)
      n_pos + 5    inbound cue (unused)
      n_pos + 6    center cue
      n_pos + 7 : n_pos + 7 + n_arms   eight-way successor output head

    ``__getitem__`` returns five tensors:
      x, y, loss_weights, successor_target, successor_query_mask
    """

    def __init__(
        self,
        n_samples: int = 2048,
        seq_len: int | None = None,
        n_space: int = 40,
        n_arms: int = 8,
        arm_len: int = 3,
        successor_seq_length: int = 4,
        successor_delay_steps: int = 5,
        successor_query_hold_steps: int = 2,
        bump_sigma: float = 0.75,
        seed: int = 0,
    ):
        super().__init__()
        if n_arms != 8:
            raise ValueError("This implementation currently assumes n_arms=8.")
        if arm_len < 2:
            raise ValueError("arm_len must be at least 2.")
        if not 2 <= successor_seq_length <= n_arms:
            raise ValueError("successor_seq_length must be in [2, n_arms].")
        if successor_delay_steps < 0:
            raise ValueError("successor_delay_steps must be non-negative.")
        if successor_query_hold_steps < 2:
            raise ValueError(
                "successor_query_hold_steps must be at least 2 so FastWave can "
                "form a query state before reading the transition memory."
            )
        if bump_sigma <= 0:
            raise ValueError("bump_sigma must be positive.")

        self.n_samples = int(n_samples)
        self.n_space = int(n_space)
        self.n_arms = int(n_arms)
        self.arm_len = int(arm_len)
        self.successor_seq_length = int(successor_seq_length)
        self.successor_delay_steps = int(successor_delay_steps)
        self.successor_query_hold_steps = int(successor_query_hold_steps)
        self.bump_sigma = float(bump_sigma)

        self.center_idx = 0
        self.n_pos = 1 + self.n_arms * self.arm_len
        self.cue_forced = self.n_pos
        self.cue_choice = self.n_pos + 1
        self.cue_query = self.cue_choice
        self.cue_reward = self.n_pos + 2
        self.cue_settle = self.n_pos + 3
        self.cue_outbound = self.n_pos + 4
        self.cue_inbound = self.n_pos + 5
        self.cue_center = self.n_pos + 6
        self.arm_choice_start = self.n_pos + 7
        self.arm_choice_end = self.arm_choice_start + self.n_arms
        self.min_n_space = self.arm_choice_end
        if self.n_space < self.min_n_space:
            raise ValueError(
                f"n_space={self.n_space} is too small; need at least "
                f"{self.min_n_space}."
            )

        self.natural_seq_len = (
            self.successor_seq_length
            + self.successor_delay_steps
            + self.successor_query_hold_steps
        )
        if seq_len is None:
            seq_len = self.natural_seq_len
        if seq_len < self.natural_seq_len:
            raise ValueError(
                f"seq_len={seq_len} is too short; need at least "
                f"{self.natural_seq_len}."
            )
        self.seq_len = int(seq_len)
        self.query_start = self.successor_seq_length + self.successor_delay_steps
        self.successor_prediction_time = (
            self.query_start + self.successor_query_hold_steps - 1
        )

        self.pos_dist = self._make_radial_graph_distance_matrix()
        g = torch.Generator().manual_seed(seed)

        xs = torch.zeros(self.n_samples, self.seq_len, self.n_space)
        ys = torch.zeros_like(xs)
        loss_weights = torch.zeros_like(xs)
        successor_targets = torch.empty(self.n_samples, dtype=torch.long)
        query_masks = torch.zeros(self.n_samples, self.seq_len, dtype=torch.bool)
        sequences = torch.empty(
            self.n_samples, self.successor_seq_length, dtype=torch.long
        )
        query_indices = torch.empty(self.n_samples, dtype=torch.long)
        query_arms = torch.empty(self.n_samples, dtype=torch.long)

        for i in range(self.n_samples):
            sequence = torch.randperm(self.n_arms, generator=g)[
                : self.successor_seq_length
            ]
            query_index = int(
                torch.randint(
                    0,
                    self.successor_seq_length - 1,
                    (1,),
                    generator=g,
                ).item()
            )
            query_arm = int(sequence[query_index].item())
            successor = int(sequence[query_index + 1].item())

            sequences[i] = sequence
            query_indices[i] = query_index
            query_arms[i] = query_arm
            successor_targets[i] = successor

            frames = []
            for arm in sequence.tolist():
                frames.append(self._arm_frame(arm, phase="forced"))
            for _ in range(self.successor_delay_steps):
                frames.append(self._center_frame(phase="settle"))
            for _ in range(self.successor_query_hold_steps):
                frames.append(self._arm_frame(query_arm, phase="query"))

            while len(frames) < self.seq_len:
                frames.append(self._center_frame(phase="settle"))

            xs[i] = torch.stack(frames[: self.seq_len], dim=0)
            ys[i, self.successor_prediction_time, self.arm_choice_start + successor] = 1.0
            loss_weights[
                i,
                self.successor_prediction_time,
                self.arm_choice_start : self.arm_choice_end,
            ] = 1.0
            query_masks[i, self.successor_prediction_time] = True

        self.x = xs.float()
        self.y = ys.float()
        self.loss_weights = loss_weights.float()
        self.successor_targets = successor_targets
        self.successor_query_masks = query_masks
        self.sequences = sequences
        self.query_indices = query_indices
        self.query_arms = query_arms

    def arm_pos_idx(self, arm: int, depth: int) -> int:
        return 1 + int(arm) * self.arm_len + int(depth)

    def _pos_to_arm_depth(self, pos_idx: int):
        if pos_idx == self.center_idx:
            return -1, -1
        z = pos_idx - 1
        return z // self.arm_len, z % self.arm_len

    def _make_radial_graph_distance_matrix(self) -> torch.Tensor:
        distance = torch.zeros(self.n_pos, self.n_pos)
        for i in range(self.n_pos):
            ai, di = self._pos_to_arm_depth(i)
            for j in range(self.n_pos):
                aj, dj = self._pos_to_arm_depth(j)
                if i == j:
                    d = 0
                elif i == self.center_idx:
                    d = dj + 1
                elif j == self.center_idx:
                    d = di + 1
                elif ai == aj:
                    d = abs(di - dj)
                else:
                    d = (di + 1) + (dj + 1)
                distance[i, j] = float(d)
        return distance

    def _position_bump(self, pos_idx: int) -> torch.Tensor:
        d = self.pos_dist[pos_idx]
        bump = torch.exp(-0.5 * (d / self.bump_sigma) ** 2)
        return bump / bump.max().clamp_min(1e-8)

    def _arm_frame(self, arm: int, phase: str) -> torch.Tensor:
        frame = torch.zeros(self.n_space)
        reward_pos = self.arm_pos_idx(arm, self.arm_len - 1)
        frame[: self.n_pos] = self._position_bump(reward_pos)
        if phase == "forced":
            frame[self.cue_forced] = 1.0
        elif phase == "query":
            frame[self.cue_query] = 1.0
        else:
            raise ValueError(f"Unknown arm-frame phase: {phase}")
        return frame

    def _center_frame(self, phase: str) -> torch.Tensor:
        frame = torch.zeros(self.n_space)
        frame[: self.n_pos] = self._position_bump(self.center_idx)
        if phase == "settle":
            frame[self.cue_settle] = 1.0
        else:
            raise ValueError(f"Unknown center-frame phase: {phase}")
        frame[self.cue_center] = 1.0
        return frame

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.loss_weights[idx],
            self.successor_targets[idx],
            self.successor_query_masks[idx],
        )


class EightArmTransitionRecallDataset(torch.utils.data.Dataset):
    """One-shot arm-to-arm transition recall with optional activity scrubbing.

    Each episode contains ``n_pairs`` randomly generated source->target pairs.
    Every target frame carries a dedicated write cue. FastWave therefore writes
    only after a source state has been established and the target arrives:

        reset -> source A -> target B + WRITE

    For the independent-pairs condition, a hard state-reset frame is inserted
    between pairs and before the query. The reset erases neural activity in all
    recurrent models, while FastWave retains its fast synaptic matrix. The
    query arm is repeated; prediction is scored on the final query frame.

    Channel layout (default n_space=40):
      0:25    radial-maze bump channels
      25      encoding cue
      26      query cue
      27      fast-write cue (active on each target frame)
      28      hard state-reset cue
      29      source cue
      30      target cue
      31      center cue
      32:40   eight-way successor output head

    ``__getitem__`` returns:
      x, y, loss_weights, successor_target, successor_query_mask
    """

    def __init__(
        self,
        n_samples: int = 2048,
        seq_len: int | None = None,
        n_space: int = 40,
        n_arms: int = 8,
        arm_len: int = 3,
        transition_n_pairs: int = 1,
        transition_delay_steps: int = 0,
        transition_query_hold_steps: int = 2,
        transition_reset_between_pairs: bool = False,
        transition_reset_before_query: bool = False,
        bump_sigma: float = 0.75,
        seed: int = 0,
    ):
        super().__init__()
        if n_arms != 8:
            raise ValueError("This implementation currently assumes n_arms=8.")
        if arm_len < 2:
            raise ValueError("arm_len must be at least 2.")
        if not 1 <= transition_n_pairs <= n_arms // 2:
            raise ValueError("transition_n_pairs must be in [1, n_arms // 2].")
        if transition_delay_steps < 0:
            raise ValueError("transition_delay_steps must be non-negative.")
        if transition_query_hold_steps < 2:
            raise ValueError(
                "transition_query_hold_steps must be at least 2: the first "
                "query frame creates the key state and the second reads F."
            )
        if bump_sigma <= 0:
            raise ValueError("bump_sigma must be positive.")

        self.n_samples = int(n_samples)
        self.n_space = int(n_space)
        self.n_arms = int(n_arms)
        self.arm_len = int(arm_len)
        self.transition_n_pairs = int(transition_n_pairs)
        self.transition_delay_steps = int(transition_delay_steps)
        self.transition_query_hold_steps = int(transition_query_hold_steps)
        self.transition_reset_between_pairs = bool(transition_reset_between_pairs)
        self.transition_reset_before_query = bool(transition_reset_before_query)
        self.bump_sigma = float(bump_sigma)

        self.center_idx = 0
        self.n_pos = 1 + self.n_arms * self.arm_len
        self.cue_encode = self.n_pos
        self.cue_query = self.n_pos + 1
        self.cue_write = self.n_pos + 2
        self.cue_reset = self.n_pos + 3
        self.cue_source = self.n_pos + 4
        self.cue_target = self.n_pos + 5
        self.cue_center = self.n_pos + 6
        self.arm_choice_start = self.n_pos + 7
        self.arm_choice_end = self.arm_choice_start + self.n_arms
        self.min_n_space = self.arm_choice_end
        if self.n_space < self.min_n_space:
            raise ValueError(
                f"n_space={self.n_space} is too small; need at least "
                f"{self.min_n_space}."
            )

        # Generic hooks consumed by train.py/model constructors.
        self.fast_write_cue = self.cue_write
        self.cue_forced = self.cue_write  # backward-compatible alias
        self.state_reset_cue = self.cue_reset

        pair_frames = 2 * self.transition_n_pairs
        between_resets = (
            self.transition_n_pairs - 1
            if self.transition_reset_between_pairs
            else 0
        )
        query_reset = 1 if self.transition_reset_before_query else 0
        self.natural_seq_len = (
            pair_frames
            + between_resets
            + self.transition_delay_steps
            + query_reset
            + self.transition_query_hold_steps
        )
        if seq_len is None:
            seq_len = self.natural_seq_len
        if seq_len < self.natural_seq_len:
            raise ValueError(
                f"seq_len={seq_len} is too short; need at least "
                f"{self.natural_seq_len}."
            )
        self.seq_len = int(seq_len)

        self.pos_dist = self._make_radial_graph_distance_matrix()
        g = torch.Generator().manual_seed(seed)

        xs = torch.zeros(self.n_samples, self.seq_len, self.n_space)
        ys = torch.zeros_like(xs)
        loss_weights = torch.zeros_like(xs)
        successor_targets = torch.empty(self.n_samples, dtype=torch.long)
        query_masks = torch.zeros(self.n_samples, self.seq_len, dtype=torch.bool)

        pair_sources = torch.empty(
            self.n_samples, self.transition_n_pairs, dtype=torch.long
        )
        pair_targets = torch.empty_like(pair_sources)
        pair_source_times = torch.empty_like(pair_sources)
        pair_target_times = torch.empty_like(pair_sources)
        queried_pair_indices = torch.empty(self.n_samples, dtype=torch.long)
        query_arms = torch.empty(self.n_samples, dtype=torch.long)
        query_start_times = torch.empty(self.n_samples, dtype=torch.long)
        prediction_times = torch.empty(self.n_samples, dtype=torch.long)

        for i in range(self.n_samples):
            # Distinct arms make the first sanity and independent-pair tasks
            # unambiguous and prevent duplicate keys or values within a trial.
            perm = torch.randperm(self.n_arms, generator=g)[
                : 2 * self.transition_n_pairs
            ]
            sources = perm[0::2]
            targets = perm[1::2]
            query_pair = int(
                torch.randint(
                    0, self.transition_n_pairs, (1,), generator=g
                ).item()
            )
            query_arm = int(sources[query_pair].item())
            successor = int(targets[query_pair].item())

            pair_sources[i] = sources
            pair_targets[i] = targets
            queried_pair_indices[i] = query_pair
            query_arms[i] = query_arm
            successor_targets[i] = successor

            frames: list[torch.Tensor] = []
            source_times = []
            target_times = []
            for pair_idx, (source, target) in enumerate(
                zip(sources.tolist(), targets.tolist())
            ):
                if pair_idx > 0 and self.transition_reset_between_pairs:
                    frames.append(self._reset_frame())
                source_times.append(len(frames))
                frames.append(self._arm_frame(source, phase="source"))
                target_times.append(len(frames))
                frames.append(self._arm_frame(target, phase="target"))

            for _ in range(self.transition_delay_steps):
                frames.append(self._center_frame())

            if self.transition_reset_before_query:
                frames.append(self._reset_frame())

            query_start = len(frames)
            for _ in range(self.transition_query_hold_steps):
                frames.append(self._arm_frame(query_arm, phase="query"))
            prediction_time = len(frames) - 1

            while len(frames) < self.seq_len:
                frames.append(self._center_frame())

            xs[i] = torch.stack(frames[: self.seq_len], dim=0)
            ys[i, prediction_time, self.arm_choice_start + successor] = 1.0
            loss_weights[
                i,
                prediction_time,
                self.arm_choice_start : self.arm_choice_end,
            ] = 1.0
            query_masks[i, prediction_time] = True

            pair_source_times[i] = torch.tensor(source_times)
            pair_target_times[i] = torch.tensor(target_times)
            query_start_times[i] = query_start
            prediction_times[i] = prediction_time

        self.x = xs.float()
        self.y = ys.float()
        self.loss_weights = loss_weights.float()
        self.successor_targets = successor_targets
        self.successor_query_masks = query_masks
        self.pair_sources = pair_sources
        self.pair_targets = pair_targets
        self.pair_source_times = pair_source_times
        self.pair_target_times = pair_target_times
        self.queried_pair_indices = queried_pair_indices
        self.query_arms = query_arms
        self.query_start_times = query_start_times
        self.prediction_times = prediction_times

        # Scalar aliases are convenient because all trials share the same timing.
        self.query_start = int(query_start_times[0].item())
        self.successor_prediction_time = int(prediction_times[0].item())

    def arm_pos_idx(self, arm: int, depth: int) -> int:
        return 1 + int(arm) * self.arm_len + int(depth)

    def _pos_to_arm_depth(self, pos_idx: int):
        if pos_idx == self.center_idx:
            return -1, -1
        z = pos_idx - 1
        return z // self.arm_len, z % self.arm_len

    def _make_radial_graph_distance_matrix(self) -> torch.Tensor:
        distance = torch.zeros(self.n_pos, self.n_pos)
        for i in range(self.n_pos):
            ai, di = self._pos_to_arm_depth(i)
            for j in range(self.n_pos):
                aj, dj = self._pos_to_arm_depth(j)
                if i == j:
                    d = 0
                elif i == self.center_idx:
                    d = dj + 1
                elif j == self.center_idx:
                    d = di + 1
                elif ai == aj:
                    d = abs(di - dj)
                else:
                    d = (di + 1) + (dj + 1)
                distance[i, j] = float(d)
        return distance

    def _position_bump(self, pos_idx: int) -> torch.Tensor:
        d = self.pos_dist[pos_idx]
        bump = torch.exp(-0.5 * (d / self.bump_sigma) ** 2)
        return bump / bump.max().clamp_min(1e-8)

    def _arm_frame(self, arm: int, phase: str) -> torch.Tensor:
        frame = torch.zeros(self.n_space)
        reward_pos = self.arm_pos_idx(arm, self.arm_len - 1)
        frame[: self.n_pos] = self._position_bump(reward_pos)
        if phase in {"source", "query"}:
            # Source and query frames are deliberately identical. After a hard
            # state reset, the isolated source arm therefore recreates the same
            # fast-weight key at encoding and retrieval time.
            frame[self.cue_query] = 1.0
            frame[self.cue_source] = 1.0
        elif phase == "target":
            frame[self.cue_target] = 1.0
            frame[self.cue_write] = 1.0
        else:
            raise ValueError(f"Unknown arm-frame phase: {phase}")
        return frame

    def _center_frame(self) -> torch.Tensor:
        frame = torch.zeros(self.n_space)
        frame[: self.n_pos] = self._position_bump(self.center_idx)
        frame[self.cue_center] = 1.0
        return frame

    def _reset_frame(self) -> torch.Tensor:
        frame = torch.zeros(self.n_space)
        frame[self.cue_reset] = 1.0
        return frame

    def query_frame(self, arm: int) -> torch.Tensor:
        """Public helper used by diagnostic query sweeps."""
        return self._arm_frame(int(arm), phase="query")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.loss_weights[idx],
            self.successor_targets[idx],
            self.successor_query_masks[idx],
        )

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
            action_hold_steps=getattr(args, "action_hold_steps", 1),
            choice_order=getattr(args, "choice_order", "random"),
            choice_objective=getattr(args, "choice_objective", "valid_set"),
            bump_sigma=getattr(args, "bump_sigma", 0.75),
            forced_departure_weight=getattr(args, "forced_departure_weight", 3.0),
            choice_departure_weight=getattr(args, "choice_departure_weight", 10.0),
            arm_choice_weight=getattr(args, "arm_choice_weight", 50.0),
            routing_weight=getattr(args, "routing_weight", 20.0),
            seed=seed,
        )


    if args.task == "eight_arm_successor":
        return EightArmSuccessorRecallDataset(
            n_samples=n_samples,
            seq_len=args.seq_len,
            n_space=args.n_space,
            n_arms=getattr(args, "n_arms", 8),
            arm_len=getattr(args, "arm_len", 3),
            successor_seq_length=getattr(args, "successor_seq_length", 4),
            successor_delay_steps=getattr(args, "successor_delay_steps", 5),
            successor_query_hold_steps=getattr(
                args, "successor_query_hold_steps", 2
            ),
            bump_sigma=getattr(args, "bump_sigma", 0.75),
            seed=seed,
        )


    if args.task == "eight_arm_transition_recall":
        return EightArmTransitionRecallDataset(
            n_samples=n_samples,
            seq_len=args.seq_len,
            n_space=args.n_space,
            n_arms=getattr(args, "n_arms", 8),
            arm_len=getattr(args, "arm_len", 3),
            transition_n_pairs=getattr(args, "transition_n_pairs", 1),
            transition_delay_steps=getattr(args, "transition_delay_steps", 0),
            transition_query_hold_steps=getattr(
                args, "transition_query_hold_steps", 2
            ),
            transition_reset_between_pairs=getattr(
                args, "transition_reset_between_pairs", False
            ),
            transition_reset_before_query=getattr(
                args, "transition_reset_before_query", False
            ),
            bump_sigma=getattr(args, "bump_sigma", 0.75),
            seed=seed,
        )

    raise ValueError(f"Unknown task: {args.task}")
