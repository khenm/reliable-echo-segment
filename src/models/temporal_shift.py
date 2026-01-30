import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger()


class TemporalShiftModule(nn.Module):
    """
    TSM: Temporal Shift Module (Lin et al., 2019).
    Shifts a fraction of channels along time dimension for temporal context.
    Zero additional parameters, zero additional FLOPs.
    """

    def __init__(self, shift_fraction: float = 0.125):
        super().__init__()
        self.shift_fraction = shift_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) - feature tensor with temporal dimension
        Returns:
            (B, T, C, H, W) - same shape with shifted channels
        """
        B, T, C, H, W = x.shape
        shift_size = int(C * self.shift_fraction)

        if shift_size == 0 or T <= 1:
            return x

        out = x.clone()

        # Shift first shift_size channels backward (past info -> current)
        out[:, 1:, :shift_size] = x[:, :-1, :shift_size]
        out[:, 0, :shift_size] = 0

        # Shift next shift_size channels forward (future info -> current)
        out[:, :-1, shift_size:2*shift_size] = x[:, 1:, shift_size:2*shift_size]
        out[:, -1, shift_size:2*shift_size] = 0

        return out


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell for spatial-temporal feature learning.
    Maintains spatial structure while learning temporal dynamics.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x: torch.Tensor, state: tuple = None) -> tuple:
        """
        Args:
            x: (B, C, H, W) - input at single timestep
            state: tuple of (h, c) each (B, hidden_dim, H, W)
        Returns:
            h_next: (B, hidden_dim, H, W)
            (h_next, c_next): new state tuple
        """
        B, C, H, W = x.shape

        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class ConvLSTM(nn.Module):
    """
    Full ConvLSTM layer that processes a sequence.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            output: (B, T, hidden_dim, H, W)
        """
        B, T, C, H, W = x.shape
        outputs = []
        state = None

        for t in range(T):
            h, state = self.cell(x[:, t], state)
            outputs.append(h)

        return torch.stack(outputs, dim=1)
