import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """Tiny Elman RNN baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, u: torch.Tensor):
        # u: [B, T, input_dim]
        B, T, _ = u.shape
        h = torch.zeros(B, self.hidden_dim, device=u.device)
        ys, hs = [], []
        for t in range(T):
            h = torch.tanh(self.inp(u[:, t]) + self.rec(h))
            ys.append(self.out(h))
            hs.append(h)
        return torch.stack(ys, dim=1), {"hidden": torch.stack(hs, dim=1)}


class GlobalFastRNN(nn.Module):
    """Elman RNN plus global Hinton-style fast weights.

    This is the standard non-local fast-weights baseline:
        F <- lambda F + eta h_new h_old^T   (transition)
        F <- lambda F + eta h_new h_new^T   (autoassoc)

    Retrieval uses the previous fast-weight matrix before the current update:
        fast_drive = F h_old
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lam: float = 0.95,
        eta: float = 0.1,
        beta: float = 1.0,
        fast_update: str = "autoassoc",
    ):
        super().__init__()
        if fast_update not in {"autoassoc", "transition"}:
            raise ValueError("fast_update must be 'autoassoc' or 'transition'")
        self.hidden_dim = hidden_dim
        self.lam = lam
        self.eta = eta
        self.beta = beta
        self.fast_update = fast_update

        self.inp = nn.Linear(input_dim, hidden_dim)
        self.rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, u: torch.Tensor):
        # u: [B, T, input_dim]
        B, T, _ = u.shape
        H = self.hidden_dim
        h = torch.zeros(B, H, device=u.device)
        F_fast = torch.zeros(B, H, H, device=u.device)
        ys, hs, f_norms, fast_drive_norms = [], [], [], []

        for t in range(T):
            h_old = h
            fast_drive = torch.bmm(F_fast, h_old.unsqueeze(-1)).squeeze(-1)
            h_new = torch.tanh(
                self.inp(u[:, t]) + self.rec(h_old) + self.beta * fast_drive
            )

            if self.fast_update == "transition":
                # Store h_new-from-h_old mapping.
                outer = torch.einsum("bi,bj->bij", h_new, h_old)
            else:
                # Store autoassociative memory of the new hidden state.
                outer = torch.einsum("bi,bj->bij", h_new, h_new)
            F_fast = self.lam * F_fast + self.eta * outer

            ys.append(self.out(h_new))
            hs.append(h_new)
            f_norms.append(F_fast.norm(dim=(1, 2)))
            fast_drive_norms.append(fast_drive.norm(dim=1))
            h = h_new

        return torch.stack(ys, dim=1), {
            "hidden": torch.stack(hs, dim=1),
            "fast_weight_norm": torch.stack(f_norms, dim=1),
            "fast_drive_norm": torch.stack(fast_drive_norms, dim=1),
        }


class LocalFastRNN(nn.Module):
    """First-order local spatial RNN plus local fast weights over patches.

    This is the local-fast-memory baseline without second-order wave dynamics.
    It keeps the same local patch fast weights as LocalFastWaveRNN, but removes
    velocity, damping, omega, and dt.

    fast_update='autoassoc':  F_i <- lambda F_i + eta patch_new patch_new^T
    fast_update='transition': F_i <- lambda F_i + eta patch_new patch_old^T
    """

    def __init__(
        self,
        input_dim: int,
        n_space: int,
        output_dim: int,
        channels: int = 1,
        kernel_size: int = 7,
        patch_size: int = 5,
        lam: float = 0.95,
        eta: float = 0.1,
        beta: float = 1.0,
        fast_update: str = "transition",
    ):
        super().__init__()
        if fast_update not in {"autoassoc", "transition"}:
            raise ValueError("fast_update must be 'autoassoc' or 'transition'")
        self.n_space = n_space
        self.channels = channels
        self.patch_size = patch_size
        self.patch_dim = channels * patch_size
        self.lam = lam
        self.eta = eta
        self.beta = beta
        self.fast_update = fast_update
        hidden_dim = channels * n_space

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.local_conv = CircularDepthwiseConv1D(channels, kernel_size)
        self.fast_to_site = nn.Linear(self.patch_dim, channels)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, u: torch.Tensor):
        # u: [B, T, input_dim]
        B, T, _ = u.shape
        C, N, P = self.channels, self.n_space, self.patch_dim
        x = torch.zeros(B, C, N, device=u.device)
        F_fast = torch.zeros(B, N, P, P, device=u.device)
        ys, xs, f_norms, fast_drive_norms = [], [], [], []

        for t in range(T):
            patches = local_patches_1d(x, self.patch_size)  # [B,N,P]
            retrieved = torch.einsum("bnij,bnj->bni", F_fast, patches)
            fast_drive = self.fast_to_site(retrieved).transpose(1, 2)  # [B,C,N]

            input_drive = self.input_proj(u[:, t]).view(B, C, N)
            spatial_drive = self.local_conv(x)
            x_new = torch.tanh(spatial_drive + self.beta * fast_drive + input_drive)

            new_patches = local_patches_1d(x_new, self.patch_size)
            if self.fast_update == "transition":
                # Store local next-patch-from-current-patch mapping.
                outer = torch.einsum("bni,bnj->bnij", new_patches, patches)
            else:
                # Store local autoassociative memory of the new patch.
                outer = torch.einsum("bni,bnj->bnij", new_patches, new_patches)
            F_fast = self.lam * F_fast + self.eta * outer

            ys.append(self.out(x_new.reshape(B, -1)))
            xs.append(x_new)
            f_norms.append(F_fast.norm(dim=(2, 3)).mean(dim=1))
            fast_drive_norms.append(fast_drive.norm(dim=(1, 2)))
            x = x_new

        return torch.stack(ys, dim=1), {
            "local_state": torch.stack(xs, dim=1),
            "fast_weight_norm": torch.stack(f_norms, dim=1),
            "fast_drive_norm": torch.stack(fast_drive_norms, dim=1),
        }


class CircularDepthwiseConv1D(nn.Module):
    """Depthwise circular 1D convolution for local spatial wave coupling."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        pad = self.kernel_size // 2
        x_pad = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
        return self.conv(x_pad)


def local_patches_1d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Return circular local patches. x [B,C,N] -> patches [B,N,C*patch_size]."""
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")
    B, C, N = x.shape
    r = patch_size // 2
    x_pad = torch.cat([x[..., -r:], x, x[..., :r]], dim=-1)
    patches = []
    for i in range(N):
        patches.append(x_pad[..., i : i + patch_size].reshape(B, C * patch_size))
    return torch.stack(patches, dim=1)


class WaveRNN(nn.Module):
    """Damped driven wave RNN on a 1D ring.

    The diagnostics-only ``intervention`` dictionary may contain:
      - start_t: intervention onset
      - zero_velocity_once: set v=0 once at start_t

    No new trainable parameters are introduced, so old checkpoints remain
    compatible.
    """

    def __init__(
        self,
        input_dim: int,
        n_space: int,
        output_dim: int,
        channels: int = 1,
        kernel_size: int = 7,
        dt: float = 0.1,
        omega: float = 1.0,
        damping: float = 0.2,
        readout_state: str = "x",
    ):
        super().__init__()
        if readout_state not in {"x", "xv"}:
            raise ValueError("readout_state must be 'x' or 'xv'")
        self.n_space = n_space
        self.channels = channels
        self.dt = dt
        self.omega = omega
        self.damping = damping
        self.readout_state = readout_state
        hidden_dim = channels * n_space

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.wave_conv = CircularDepthwiseConv1D(channels, kernel_size)
        readout_dim = hidden_dim if readout_state == "x" else 2 * hidden_dim
        self.out = nn.Linear(readout_dim, output_dim)

    def initial_state(self, batch_size: int, reference: torch.Tensor):
        C, N = self.channels, self.n_space
        return {
            "x": reference.new_zeros(batch_size, C, N),
            "v": reference.new_zeros(batch_size, C, N),
        }

    def step(self, u_t, state, t: int = 0, intervention=None):
        intervention = {} if intervention is None else intervention
        start_t = int(intervention.get("start_t", 10**12))
        x = state["x"]
        v = state["v"]

        if t == start_t and intervention.get("zero_velocity_once", False):
            v = torch.zeros_like(v)

        x_pre, v_pre = x, v
        input_drive = self.input_proj(u_t).view(
            u_t.shape[0], self.channels, self.n_space
        )
        spatial_drive = self.wave_conv(x)
        drive = torch.tanh(spatial_drive + input_drive)

        v_new = v + self.dt * (
            -(self.omega**2) * x - self.damping * v + drive
        )
        x_new = x + self.dt * v_new

        x_flat = x_new.reshape(u_t.shape[0], -1)
        if self.readout_state == "xv":
            readout = torch.cat([x_flat, v_new.reshape(u_t.shape[0], -1)], dim=-1)
        else:
            readout = x_flat
        y_t = self.out(readout)

        new_state = {"x": x_new, "v": v_new}
        aux = {
            "x_pre": x_pre,
            "v_pre": v_pre,
            "x": x_new,
            "v": v_new,
        }
        return y_t, new_state, aux

    def forward(
        self,
        u: torch.Tensor,
        record_times=None,
        record_all: bool = False,
        intervention=None,
    ):
        B, T, _ = u.shape
        state = self.initial_state(B, u)
        selected = set(range(T)) if record_all else set(record_times or [])

        ys, xs, vs = [], [], []
        diag_times, diag_x_pre, diag_v_pre = [], [], []
        for t in range(T):
            y_t, state, step_aux = self.step(
                u[:, t], state, t=t, intervention=intervention
            )
            ys.append(y_t)
            xs.append(step_aux["x"])
            vs.append(step_aux["v"])
            if t in selected:
                diag_times.append(t)
                diag_x_pre.append(step_aux["x_pre"].detach().cpu())
                diag_v_pre.append(step_aux["v_pre"].detach().cpu())

        extras = {
            "wave_state": torch.stack(xs, dim=1),
            "wave_velocity": torch.stack(vs, dim=1),
        }
        if diag_times:
            extras.update(
                {
                    "diagnostic_times": torch.tensor(diag_times, dtype=torch.long),
                    "wave_state_pre": torch.stack(diag_x_pre, dim=1),
                    "wave_velocity_pre": torch.stack(diag_v_pre, dim=1),
                }
            )
        return torch.stack(ys, dim=1), extras


class LocalFastWaveRNN(nn.Module):
    """Wave RNN plus local Hinton-style fast weights over spatial patches.

    fast_update='autoassoc': F_i <- lambda F_i + eta patch_t patch_t^T
    fast_update='transition': F_i <- lambda F_i + eta patch_{t+1} patch_t^T

    ``record_times=[...]`` stores the complete fast state only at selected
    timesteps. ``record_all=True`` stores it at every timestep and should be
    used only with a small batch because F can be large.

    Optional ``intervention`` keys:
      - start_t: intervention onset
      - erase_fast_once: set F=0 once at onset
      - shuffle_fast_once: swap F across trials once at onset
      - zero_velocity_once: set v=0 once at onset
      - disable_fast_drive_after: set applied fast drive to zero from onset
      - disable_fast_write_after: allow decay but no new writes from onset

    No new trainable parameters are introduced, so old checkpoints load exactly.
    """

    def __init__(
        self,
        input_dim: int,
        n_space: int,
        output_dim: int,
        channels: int = 1,
        kernel_size: int = 7,
        patch_size: int = 5,
        dt: float = 0.1,
        omega: float = 1.0,
        damping: float = 0.2,
        readout_state: str = "x",
        lam: float = 0.95,
        eta: float = 0.1,
        beta: float = 1.0,
        fast_update: str = "transition",
    ):
        super().__init__()
        if fast_update not in {"autoassoc", "transition"}:
            raise ValueError("fast_update must be 'autoassoc' or 'transition'")
        if readout_state not in {"x", "xv"}:
            raise ValueError("readout_state must be 'x' or 'xv'")
        self.n_space = n_space
        self.channels = channels
        self.patch_size = patch_size
        self.patch_dim = channels * patch_size
        self.dt = dt
        self.omega = omega
        self.damping = damping
        self.readout_state = readout_state
        self.lam = lam
        self.eta = eta
        self.beta = beta
        self.fast_update = fast_update
        hidden_dim = channels * n_space

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.wave_conv = CircularDepthwiseConv1D(channels, kernel_size)
        self.fast_to_site = nn.Linear(self.patch_dim, channels)
        readout_dim = hidden_dim if readout_state == "x" else 2 * hidden_dim
        self.out = nn.Linear(readout_dim, output_dim)

    def initial_state(self, batch_size: int, reference: torch.Tensor):
        C, N, P = self.channels, self.n_space, self.patch_dim
        return {
            "x": reference.new_zeros(batch_size, C, N),
            "v": reference.new_zeros(batch_size, C, N),
            "F": reference.new_zeros(batch_size, N, P, P),
        }

    def step(self, u_t, state, t: int = 0, intervention=None):
        intervention = {} if intervention is None else intervention
        start_t = int(intervention.get("start_t", 10**12))
        x = state["x"]
        v = state["v"]
        F_read = state["F"]

        # One-time interventions are applied before F_t is queried.
        if t == start_t:
            if intervention.get("zero_velocity_once", False):
                v = torch.zeros_like(v)
            if intervention.get("erase_fast_once", False):
                F_read = torch.zeros_like(F_read)
            if intervention.get("shuffle_fast_once", False):
                # Deterministic cross-trial shuffle; batch size one has no effect.
                F_read = F_read.flip(0)

        x_pre, v_pre = x, v
        patches = local_patches_1d(x, self.patch_size)
        retrieved = torch.einsum("bnij,bnj->bni", F_read, patches)
        fast_drive_raw = self.fast_to_site(retrieved).transpose(1, 2)

        if t >= start_t and intervention.get("disable_fast_drive_after", False):
            fast_drive = torch.zeros_like(fast_drive_raw)
        else:
            fast_drive = fast_drive_raw

        input_drive = self.input_proj(u_t).view(
            u_t.shape[0], self.channels, self.n_space
        )
        spatial_drive = self.wave_conv(x)
        drive = torch.tanh(spatial_drive + self.beta * fast_drive + input_drive)

        v_new = v + self.dt * (
            -(self.omega**2) * x - self.damping * v + drive
        )
        x_new = x + self.dt * v_new

        new_patches = local_patches_1d(x_new, self.patch_size)
        if t >= start_t and intervention.get("disable_fast_write_after", False):
            F_next = self.lam * F_read
        else:
            if self.fast_update == "transition":
                outer = torch.einsum("bni,bnj->bnij", new_patches, patches)
            else:
                outer = torch.einsum("bni,bnj->bnij", new_patches, new_patches)
            F_next = self.lam * F_read + self.eta * outer

        x_flat = x_new.reshape(u_t.shape[0], -1)
        if self.readout_state == "xv":
            readout = torch.cat([x_flat, v_new.reshape(u_t.shape[0], -1)], dim=-1)
        else:
            readout = x_flat
        y_t = self.out(readout)

        new_state = {"x": x_new, "v": v_new, "F": F_next}
        aux = {
            "x_pre": x_pre,
            "v_pre": v_pre,
            "x": x_new,
            "v": v_new,
            "F_read": F_read,
            "F_post": F_next,
            "query": patches,
            "retrieved": retrieved,
            "fast_drive_raw": fast_drive_raw,
            "fast_drive_applied": fast_drive,
        }
        return y_t, new_state, aux

    def forward(
        self,
        u: torch.Tensor,
        record_times=None,
        record_all: bool = False,
        intervention=None,
    ):
        B, T, _ = u.shape
        state = self.initial_state(B, u)
        selected = set(range(T)) if record_all else set(record_times or [])

        ys, xs, vs, f_norms, fast_drive_norms = [], [], [], [], []
        diag_times = []
        diagnostic = {
            "wave_state_pre": [],
            "wave_velocity_pre": [],
            "fast_memory_read": [],
            "fast_memory_post": [],
            "fast_query": [],
            "fast_retrieved": [],
            "fast_drive_raw": [],
            "fast_drive_applied": [],
        }

        for t in range(T):
            y_t, state, step_aux = self.step(
                u[:, t], state, t=t, intervention=intervention
            )
            ys.append(y_t)
            xs.append(step_aux["x"])
            vs.append(step_aux["v"])
            f_norms.append(step_aux["F_post"].norm(dim=(2, 3)).mean(dim=1))
            fast_drive_norms.append(
                step_aux["fast_drive_applied"].norm(dim=(1, 2))
            )

            if t in selected:
                diag_times.append(t)
                values = {
                    "wave_state_pre": step_aux["x_pre"],
                    "wave_velocity_pre": step_aux["v_pre"],
                    "fast_memory_read": step_aux["F_read"],
                    "fast_memory_post": step_aux["F_post"],
                    "fast_query": step_aux["query"],
                    "fast_retrieved": step_aux["retrieved"],
                    "fast_drive_raw": step_aux["fast_drive_raw"],
                    "fast_drive_applied": step_aux["fast_drive_applied"],
                }
                for key, value in values.items():
                    diagnostic[key].append(value.detach().cpu())

        extras = {
            "wave_state": torch.stack(xs, dim=1),
            "wave_velocity": torch.stack(vs, dim=1),
            "fast_weight_norm": torch.stack(f_norms, dim=1),
            "fast_drive_norm": torch.stack(fast_drive_norms, dim=1),
        }
        if diag_times:
            extras["diagnostic_times"] = torch.tensor(diag_times, dtype=torch.long)
            for key, values in diagnostic.items():
                extras[key] = torch.stack(values, dim=1)
        return torch.stack(ys, dim=1), extras

