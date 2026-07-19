# src/receiver.py

import numpy as np
import torch

from sionna.phy.ofdm import LSChannelEstimator
from sionna.phy.ofdm import LMMSEEqualizer, ZFEqualizer, MFEqualizer
from sionna.phy.mimo import StreamManagement


class OFDMReceiver:
    def __init__(
        self,
        resource_grid,
        num_rx=1,
        num_tx=1,
        num_streams_per_tx=1,
        equalizer_type="lmmse",
        interpolation_type="lin",
        perfect_csi=False,
        device=None,
    ):
        self.resource_grid = resource_grid
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_streams_per_tx = num_streams_per_tx
        self.equalizer_type = equalizer_type.lower()
        self.interpolation_type = interpolation_type
        self.perfect_csi = perfect_csi
        self.device = device

        rx_tx_association = np.ones([num_rx, num_tx], dtype=int)

        self.stream_management = StreamManagement(
            rx_tx_association,
            num_streams_per_tx
        )

        if not perfect_csi:
            self.channel_estimator = LSChannelEstimator(
                resource_grid,
                interpolation_type=interpolation_type,
                device=device
            )
        else:
            self.channel_estimator = None

        if self.equalizer_type == "lmmse":
            self.equalizer = LMMSEEqualizer(
                resource_grid,
                self.stream_management,
                device=device
            )

        elif self.equalizer_type == "zf":
            self.equalizer = ZFEqualizer(
                resource_grid,
                self.stream_management,
                device=device
            )

        elif self.equalizer_type == "mf":
            self.equalizer = MFEqualizer(
                resource_grid,
                self.stream_management,
                device=device
            )

        else:
            raise ValueError(
                "equalizer_type must be one of: 'lmmse', 'zf', 'mf'"
            )

    def _keep_effective_subcarriers(self, h_freq):
        """
        GenerateOFDMChannel이 만든 h_freq는 보통 fft_size 전체 subcarrier를 가진다.

        하지만 Sionna OFDM equalizer는 guard carrier와 DC carrier를 제거한
        effective subcarrier만 사용한다.

        따라서 perfect CSI를 equalizer에 넣기 전에 마지막 주파수 차원을
        effective_subcarrier_ind로 잘라줘야 한다.
        """

        eff_ind = torch.as_tensor(
            self.resource_grid.effective_subcarrier_ind,
            dtype=torch.long,
            device=h_freq.device
        )

        num_effective_subcarriers = eff_ind.numel()

        # 이미 effective subcarrier만 남아 있는 경우
        if h_freq.shape[-1] == num_effective_subcarriers:
            return h_freq

        # 전체 fft_size를 가진 경우
        if h_freq.shape[-1] >= int(torch.max(eff_ind).item()) + 1:
            return torch.index_select(
                h_freq,
                dim=-1,
                index=eff_ind
            )

        raise ValueError(
            f"Invalid h_freq last dimension: {h_freq.shape[-1]}. "
            f"Expected either fft_size or num_effective_subcarriers="
            f"{num_effective_subcarriers}."
        )

    def __call__(self, y_rg, no, h_freq=None):
        if self.perfect_csi:
            if h_freq is None:
                raise ValueError(
                    "h_freq must be provided when perfect_csi=True."
                )

            h_hat = self._keep_effective_subcarriers(h_freq)

            # perfect CSI이므로 channel estimation error variance는 0
            err_var = torch.zeros_like(h_hat.real)

        else:
            h_hat, err_var = self.channel_estimator(y_rg, no)

        x_hat, no_eff = self.equalizer(
            y_rg,
            h_hat,
            err_var,
            no
        )

        return x_hat, no_eff, h_hat, err_var