import math
import operator

import torch
from sionna.phy.mimo import grid_of_beams_dft, grid_of_beams_dft_ula


class DFTCodebook:
    """Create a ULA or UPA DFT codebook with shape ``[beams, antennas]``."""

    def __init__(
        self,
        array_type: str,
        num_ant: int | None = None,
        num_ant_v: int | None = None,
        num_ant_h: int | None = None,
        oversampling: int = 1,
        oversampling_v: int = 1,
        oversampling_h: int = 1,
        normalize: bool = True,
        device: str | torch.device = "cpu",
    ):
        if not isinstance(array_type, str):
            raise TypeError("array_type must be a string.")

        self.array_type = array_type.upper()
        self.device = torch.device(device)
        self.normalize = bool(normalize)

        if self.array_type == "ULA":
            num_ant = self._positive_integer("num_ant", num_ant)
            oversampling = self._positive_integer(
                "oversampling", oversampling
            )
            codebook = grid_of_beams_dft_ula(num_ant, oversampling)
            self.num_ant = num_ant

        elif self.array_type == "UPA":
            num_ant_v = self._positive_integer("num_ant_v", num_ant_v)
            num_ant_h = self._positive_integer("num_ant_h", num_ant_h)
            oversampling_v = self._positive_integer(
                "oversampling_v", oversampling_v
            )
            oversampling_h = self._positive_integer(
                "oversampling_h", oversampling_h
            )
            codebook = grid_of_beams_dft(
                num_ant_v,
                num_ant_h,
                oversampling_v,
                oversampling_h,
            )
            self.num_ant = num_ant_v * num_ant_h

            # Sionna returns [vertical_beams, horizontal_beams, antennas].
            codebook = codebook.reshape(-1, self.num_ant)

        else:
            raise ValueError("array_type must be ULA or UPA.")

        # Sionna always returns unit-norm beams. Restore the unnormalized DFT
        # coefficients when normalization is explicitly disabled.
        if not self.normalize:
            codebook = codebook * math.sqrt(self.num_ant)

        self.codebook = codebook.to(self.device)

    @staticmethod
    def _positive_integer(name: str, value: int | None) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer greater than zero.")
        try:
            value = operator.index(value)
        except TypeError as error:
            raise ValueError(
                f"{name} must be an integer greater than zero."
            ) from error
        if value < 1:
            raise ValueError(f"{name} must be an integer greater than zero.")
        return value

    @property
    def num_beams(self) -> int:
        return self.codebook.shape[0]

    def get_codebook(self) -> torch.Tensor:
        return self.codebook

    def get_beam(self, beam_index) -> torch.Tensor:
        return self.codebook[beam_index]
