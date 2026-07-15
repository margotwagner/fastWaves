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
    """Damped driven wave RNN on a 1D ring."""

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

    def forward(self, u: torch.Tensor):
        # u: [B, T, input_dim]
        B, T, _ = u.shape
        C, N = self.channels, self.n_space
        x = torch.zeros(B, C, N, device=u.device)
        v = torch.zeros(B, C, N, device=u.device)
        ys, xs, vs = [], [], []

        for t in range(T):
            input_drive = self.input_proj(u[:, t]).view(B, C, N)
            spatial_drive = self.wave_conv(x)
            drive = torch.tanh(spatial_drive + input_drive)

            v = v + self.dt * (-(self.omega**2) * x - self.damping * v + drive)
            x = x + self.dt * v

            x_flat = x.reshape(B, -1)
            if self.readout_state == "xv":
                readout = torch.cat([x_flat, v.reshape(B, -1)], dim=-1)
            else:
                readout = x_flat

            ys.append(self.out(readout))
            xs.append(x)
            vs.append(v)

        return torch.stack(ys, dim=1), {
            "wave_state": torch.stack(xs, dim=1),
            "wave_velocity": torch.stack(vs, dim=1),
        }


class LocalFastWaveRNN(nn.Module):
    """Wave RNN plus local Hinton-style fast weights over spatial patches.

    fast_update='autoassoc': F_i <- lambda F_i + eta patch_t patch_t^T
    fast_update='transition': F_i <- lambda F_i + eta patch_{t+1} patch_t^T
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
        fast_write_phase: str = "all",
        fast_nonwrite_mode: str = "decay",
        fast_write_cue_index: int | None = None,
    ):
        super().__init__()
        if fast_update not in {"autoassoc", "transition"}:
            raise ValueError("fast_update must be 'autoassoc' or 'transition'")
        if readout_state not in {"x", "xv"}:
            raise ValueError("readout_state must be 'x' or 'xv'")
        if fast_write_phase not in {"all", "forced"}:
            raise ValueError("fast_write_phase must be 'all' or 'forced'")
        if fast_nonwrite_mode not in {"decay", "hold"}:
            raise ValueError("fast_nonwrite_mode must be 'decay' or 'hold'")
        if fast_write_phase == "forced" and fast_write_cue_index is None:
            raise ValueError(
                "fast_write_cue_index is required when fast_write_phase='forced'"
            )
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
        self.fast_write_phase = fast_write_phase
        self.fast_nonwrite_mode = fast_nonwrite_mode
        self.fast_write_cue_index = fast_write_cue_index
        hidden_dim = channels * n_space

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.wave_conv = CircularDepthwiseConv1D(channels, kernel_size)
        self.fast_to_site = nn.Linear(self.patch_dim, channels)
        readout_dim = hidden_dim if readout_state == "x" else 2 * hidden_dim
        self.out = nn.Linear(readout_dim, output_dim)

    def _fast_write_gate(self, u_t: torch.Tensor) -> torch.Tensor:
        """Return a per-trial gate with shape [B, 1, 1, 1]."""
        batch_size = u_t.shape[0]
        if self.fast_write_phase == "all":
            return u_t.new_ones(batch_size, 1, 1, 1)

        if self.fast_write_cue_index is None:
            raise RuntimeError("fast_write_cue_index was not configured")
        if not 0 <= self.fast_write_cue_index < u_t.shape[-1]:
            raise IndexError(
                f"fast_write_cue_index={self.fast_write_cue_index} is outside "
                f"input dimension {u_t.shape[-1]}"
            )

        forced = u_t[:, self.fast_write_cue_index] > 0.5
        return forced.to(dtype=u_t.dtype).view(batch_size, 1, 1, 1)

    def forward(self, u: torch.Tensor):
        B, T, _ = u.shape
        C, N, P = self.channels, self.n_space, self.patch_dim
        x = torch.zeros(B, C, N, device=u.device)
        v = torch.zeros(B, C, N, device=u.device)
        F = torch.zeros(B, N, P, P, device=u.device)
        ys, xs, vs, f_norms, fast_drive_norms, write_gates = [], [], [], [], [], []

        for t in range(T):
            patches = local_patches_1d(x, self.patch_size)  # [B,N,P]
            retrieved = torch.einsum("bnij,bnj->bni", F, patches)
            fast_drive = self.fast_to_site(retrieved).transpose(1, 2)  # [B,C,N]

            input_drive = self.input_proj(u[:, t]).view(B, C, N)
            spatial_drive = self.wave_conv(x)
            drive = torch.tanh(spatial_drive + self.beta * fast_drive + input_drive)

            v = v + self.dt * (-(self.omega**2) * x - self.damping * v + drive)
            x_new = x + self.dt * v

            new_patches = local_patches_1d(x_new, self.patch_size)
            if self.fast_update == "transition":
                # Store local next-patch-from-current-patch mapping.
                outer = torch.einsum("bni,bnj->bnij", new_patches, patches)
            else:
                # Store local autoassociative memory of current/new patch.
                outer = torch.einsum("bni,bnj->bnij", new_patches, new_patches)
            write_gate = self._fast_write_gate(u[:, t])
            F_decayed = self.lam * F

            if self.fast_nonwrite_mode == "decay":
                # Outside the permitted write phase, apply only F <- lambda F.
                F = F_decayed + write_gate * self.eta * outer
            elif self.fast_nonwrite_mode == "hold":
                # Outside the permitted write phase, preserve F exactly.
                F_written = F_decayed + self.eta * outer
                F = write_gate * F_written + (1.0 - write_gate) * F
            else:
                raise RuntimeError(
                    f"Unexpected fast_nonwrite_mode: {self.fast_nonwrite_mode}"
                )

            x_flat = x_new.reshape(B, -1)
            if self.readout_state == "xv":
                readout = torch.cat([x_flat, v.reshape(B, -1)], dim=-1)
            else:
                readout = x_flat

            ys.append(self.out(readout))
            xs.append(x_new)
            vs.append(v)
            f_norms.append(F.norm(dim=(2, 3)).mean(dim=1))
            fast_drive_norms.append(fast_drive.norm(dim=(1, 2)))
            write_gates.append(write_gate[:, 0, 0, 0])
            x = x_new

        return torch.stack(ys, dim=1), {
            "wave_state": torch.stack(xs, dim=1),
            "wave_velocity": torch.stack(vs, dim=1),
            "fast_weight_norm": torch.stack(f_norms, dim=1),
            "fast_drive_norm": torch.stack(fast_drive_norms, dim=1),
            "fast_write_gate": torch.stack(write_gates, dim=1),
        }
