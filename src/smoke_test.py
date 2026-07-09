import torch

from src.models import VanillaRNN, WaveRNN, LocalFastWaveRNN, local_patches_1d
from src.tasks import make_ring_bump_sequence


def check_shapes():
    B, T, N = 4, 12, 32
    u = make_ring_bump_sequence(B, T, N)
    assert u.shape == (B, T, N)

    patches = local_patches_1d(u[:, 0].unsqueeze(1), patch_size=5)
    assert patches.shape == (B, N, 5)

    models = [
        VanillaRNN(N, 64, N),
        WaveRNN(N, N, N),
        LocalFastWaveRNN(N, N, N, patch_size=5, fast_update="transition"),
        LocalFastWaveRNN(N, N, N, patch_size=5, fast_update="autoassoc"),
    ]
    for model in models:
        yhat, extras = model(u)
        assert yhat.shape == (B, T, N), (type(model).__name__, yhat.shape)
        loss = torch.nn.functional.mse_loss(yhat, u)
        loss.backward()
        print(f"OK: {type(model).__name__} | yhat {tuple(yhat.shape)} | loss {loss.item():.4f}")


if __name__ == "__main__":
    check_shapes()
    print("All smoke tests passed.")
