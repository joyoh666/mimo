import torch

if __package__:
    from .codebook import DFTCodebook
else:
    from codebook import DFTCodebook


class BeamSweeper:
    """Exhaustively select a DFT beam for each ``[batch, K]`` channel."""

    def __init__(self, codebook: DFTCodebook):
        self.codebook_obj = codebook
        self.codebook = codebook.get_codebook()

    def compute_beam_power(self, h: torch.Tensor) -> torch.Tensor:
        """Compute beam power from ``h[batch, K, receive_ant, transmit_ant]``.

        Returns a tensor with shape ``[batch, K, num_beams]``. Receive-antenna
        powers are non-coherently summed.
        """
        self._validate_channel(h)
        h, codebook = self._align_codebook(h)

        effective_channels = torch.einsum(
            "qkrt,bt->qkrb", h, codebook
        )
        return torch.sum(torch.abs(effective_channels) ** 2, dim=-2)

    def select_beam(self, h: torch.Tensor):
        """Return ``(best_beam, best_power, best_index)`` for each batch/K."""
        beam_powers = self.compute_beam_power(h)
        best_power, best_index = torch.max(beam_powers, dim=-1)

        _, codebook = self._align_codebook(h)
        best_beam = codebook[best_index]
        return best_beam, best_power, best_index

    def _validate_channel(self, h: torch.Tensor) -> None:
        if not torch.is_tensor(h):
            raise TypeError("h must be a torch.Tensor.")
        if h.ndim != 4:
            raise ValueError(
                "h must have shape [batch, K, receive_ant, transmit_ant]."
            )
        if any(size < 1 for size in h.shape):
            raise ValueError("Every channel dimension must be greater than zero.")
        if not (h.is_floating_point() or h.is_complex()):
            raise TypeError("h must have a real or complex floating-point dtype.")
        if h.shape[-1] != self.codebook.shape[-1]:
            raise ValueError(
                f"Channel has {h.shape[-1]} transmit antennas, but the "
                f"codebook has {self.codebook.shape[-1]}."
            )

    def _align_codebook(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common_dtype = torch.promote_types(h.dtype, self.codebook.dtype)
        return (
            h.to(dtype=common_dtype),
            self.codebook.to(device=h.device, dtype=common_dtype),
        )
