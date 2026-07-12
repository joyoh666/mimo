import torch
from sionna.phy.mimo import (
    grid_of_beams_dft_ula,
    grid_of_beams_dft,
)


class DFTCodebook:
    """
    Sionna PHY의 DFT Grid-of-Beams 함수를 이용해서
    ULA 또는 URA/UPA 코드북을 생성하는 모듈.

    최종 codebook shape:
        [num_beams, num_tx_ant]

    각 row가 하나의 beamforming vector.
    """

    def __init__(
        self,
        array_type: str,
        num_ant: int = None,
        num_ant_v: int = None,
        num_ant_h: int = None,
        oversampling: int = 1,
        oversampling_v: int = 1,
        oversampling_h: int = 1,
        normalize: bool = True,
        device: str = "cpu",
        precision=None,
    ):
        self.array_type = array_type.lower()
        self.device = device
        self.normalize = normalize
        self.precision = precision

        if self.array_type == "ula":
            if num_ant is None:
                raise ValueError("For ULA, num_ant must be provided.")

            self.num_ant = num_ant
            self.num_ant_v = None
            self.num_ant_h = None
            self.oversampling = oversampling

            codebook = grid_of_beams_dft_ula(
                num_ant=num_ant,
                oversmpl=oversampling,
                precision=precision,
            )

            # Sionna output:
            # [num_ant * oversampling, num_ant]
            self.codebook = codebook.to(device)

        elif self.array_type in ["ura", "upa"]:
            if num_ant_v is None or num_ant_h is None:
                raise ValueError(
                    "For URA/UPA, num_ant_v and num_ant_h must be provided."
                )

            self.num_ant_v = num_ant_v
            self.num_ant_h = num_ant_h
            self.num_ant = num_ant_v * num_ant_h
            self.oversampling_v = oversampling_v
            self.oversampling_h = oversampling_h

            codebook = grid_of_beams_dft(
                num_ant_v=num_ant_v,
                num_ant_h=num_ant_h,
                oversmpl_v=oversampling_v,
                oversmpl_h=oversampling_h,
                precision=precision,
            )

            # Sionna output:
            # [num_ant_v * oversampling_v,
            #  num_ant_h * oversampling_h,
            #  num_ant_v * num_ant_h]
            #
            # Flatten beam grid:
            # [num_beams, num_ant]
            self.codebook = codebook.reshape(
                -1,
                self.num_ant,
            ).to(device)

        else:
            raise ValueError(
                "array_type must be one of ['ula', 'ura', 'upa']."
            )

        if self.normalize:
            self.codebook = self._normalize_codebook(self.codebook)

    @staticmethod
    def _normalize_codebook(codebook):
        """
        각 beam vector의 송신 전력이 1이 되도록 정규화.

        input:
            codebook: [num_beams, num_tx_ant]

        output:
            codebook: [num_beams, num_tx_ant]
        """

        norm = torch.sqrt(
            torch.sum(torch.abs(codebook) ** 2, dim=-1, keepdim=True)
            + 1e-12
        )

        return codebook / norm

    @property
    def num_beams(self):
        return self.codebook.shape[0]

    @property
    def num_tx_ant(self):
        return self.codebook.shape[1]

    def get_codebook(self):
        return self.codebook

    def get_beam(self, beam_index):
        """
        beam_index에 해당하는 beam vector 반환.

        beam_index:
            scalar int 또는 tensor

        output:
            beam:
                scalar index이면 [num_tx_ant]
                tensor index이면 [..., num_tx_ant]
        """

        return self.codebook[beam_index]
    
class DFTBeamSweeper:
    """
    DFT codebook 기반 exhaustive beam sweeping.

    Canonical channel shape:
        h: [..., K, M]

    Codebook shape:
        W: [B, M]

    where:
        B = number of beams
        K = receive dimension / stream dimension
        M = number of transmit antennas

    Sweeping metric:
        power_m = || H w_m ||^2

    Output:
        best_beam_index: [...]
        best_beam: [..., M]
        best_power: [...]
    """

    def __init__(self, codebook:DFTCodebook):
        self.codebook_obj = codebook
        self.codebook = codebook.get_codebook()

    def compute_beam_power(self, h):
        self._check_channel_shape(h)

        if h.shape[-1] != self.codebook.shape[-1]:
           raise ValueError(
                f"Channel has num_tx_ant={h.shape[-1]}, "
                f"but codebook has num_tx_ant={self.codebook.shape[-1]}."
            )

        effective_channels = torch.einsum("...km,bm->...kb", h, self.codebook)

        beam_powers = torch.sum(torch.abs(effective_channels)**2, dim=-2)

        return beam_powers
    
    def select_beam(self, h):
        beam_powers = self.compute_beam_power(h)

        best_power, best_index = torch.max(beam_powers, dim=-1)

        best_beam = self.codebook[best_index]

        return best_beam, best_index, best_power
    
    @staticmethod
    def _check_channel_shape(h):
        if h.ndim < 2:
            raise ValueError(
                "h must have shape [..., K, M]."
            )

        if h.shape[-1] < 1:
            raise ValueError(
                "Last dimension of h must be num_tx_ant."
            )

        if h.shape[-2] < 1:
            raise ValueError(
                "Second last dimension of h must be K."
            )
        
class CodebookPrecoder:
    def __init__(self, beam_sweeper:DFTBeamSweeper):
        self.beam_sweeper = beam_sweeper

    def compute_matrix(self,h):
        best_beam, _, _ = self.beam_sweeper.select_beam(h) 
        return best_beam.unsqueeze(-1)
    
    def apply(self, x, h):
       best_beam, _, _ = self.beam_sweeper.select_beam(h)

       x_precoded = best_beam * x.unsqueeze(-1)

       return x_precoded
    
class DFTBeamTracker:
    def __init__(self,
                 codebook: DFTCodebook,
                 neighbor_radius: int=1,
                 full_sweep_period: int=20,
                 power_drop_threshold_db: float=3.0
                 ):
        self.codebook_obj = codebook
        self.codebook = codebook.get_codebook()
        self.sweeper = DFTBeamSweeper(codebook)

        self.neighbor_radius = neighbor_radius
        self.full_sweep_period = full_sweep_period
        self.power_drop_threshold_db = power_drop_threshold_db

        self.current_beam_index = None
        self.current_power = None
        self.frame_count = 0

    def initialize(self, h):
        """
        최초 beam acquisition.
        전체 코드북을 sweeping해서 초기 beam 선택.
        """

        beam_powers = self.sweeper.compute_beam_power(h)
        beam_powers = self._average_non_beam_dims(beam_powers)

        best_power, best_index = torch.max(beam_powers, dim=-1)

        self.current_beam_index = best_index
        self.current_power = best_power

        best_beam = self.codebook[best_index]

        return best_beam, best_index, best_power

    def update(self, h):
        """
        매 시간/frame마다 호출되는 beam tracking 함수.

        h:
            현재 시간의 channel

        return:
            best_beam
            best_index
            best_power
            mode: "init", "track", or "full_sweep"
        """
        if self.current_beam_index is None:
            best_beam, best_index, best_power = self.initialize(h)
            return best_beam, best_index, best_power, "init"
        
        self.frame_count += 1

        #주기적으로 full sweep 수행
        if self.frame_count % self.full_sweep_period == 0:
            best_beam, best_index, best_power = self.initialize(h)
            return best_beam, best_index, best_power, "full-sweep"
        
        #이전 beam 주변만 탐색
        candidate_indices = self._get_neighbor_indices(
            self.current_beam_index
        )

        candidate_powers = self._compute_candidate_powers(
            h,
            candidate_indices
        )

        best_local_power, best_local_pos = torch.max(
            candidate_powers,
            dim=-1
        )

        best_index = candidate_indices[
            torch.arange(candidate_indices.shape[0], device=h.device),
            best_local_pos
        ]

        best_beam = self.codebook[best_index]

        power_drop_db = 10.0 * torch.log10(
            (self.current_power + 1e-12) / (best_local_power + 1e-12)
        )

        need_full_sweep = power_drop_db > self.power_drop_threshold_db

        if torch.any(need_full_sweep):
            full_beam_powers = self.sweeper.compute_beam_power(h)
            full_beam_powers = self._average_non_beam_dims(full_beam_powers)

            full_power, full_index = torch.max(full_beam_powers, dim=-1)
            full_beam = self.codebook[full_index]

            best_index = torch.where(need_full_sweep, full_index, best_index)
            best_power = torch.where(need_full_sweep, full_power, best_local_power)
            best_beam = self.codebook[best_index]

            mode = "full_sweep"
        else:
            best_power = best_local_power
            mode = "track"

        self.current_beam_index = best_index
        self.current_power = best_power

        return best_beam, best_index, best_power, mode

    def _average_non_beam_dims(self, beam_powers):
        """
        beam_powers shape:
            [batch, ..., num_beams]

        OFDM이면 symbol/subcarrier 축 평균.
        Flat이면 그대로 사용.
        """

        if beam_powers.ndim == 2:
            return beam_powers

        reduce_dims = tuple(range(1, beam_powers.ndim - 1))

        return torch.mean(
            beam_powers,
            dim=reduce_dims
        )
    
    def _compute_candidate_powers(self, h, candidate_indices):
        """
        후보 beam들에 대해서만 power 계산.

        h:
            [..., K, M]

        candidate_indices:
            [batch, num_candidates]

        output:
            [batch, num_candidates]
        """

        batch_size = h.shape[0]
        powers = []

        for b in range(batch_size):
            h_b = h[b:b+1]
            cb_b = self.codebook[candidate_indices[b]]

            effective = torch.einsum(
                "...km,bm->...kb",
                h_b,
                cb_b
            )

            power = torch.sum(
                torch.abs(effective) ** 2,
                dim=-2
            )

            power = self._average_non_beam_dims(power)
            power = power.squeeze(0)

            powers.append(power)

        return torch.stack(powers, dim=0)

    def _get_neighbor_indices(self, beam_index):
        """
        현재 beam index 주변 후보 beam index 반환.

        현재 구현은 ULA 또는 flatten된 코드북 기준.
        우선 단순하게 index 주변 beam을 후보로 사용한다.
        """

        device = beam_index.device
        num_beams = self.codebook.shape[0]

        offsets = torch.arange(
            -self.neighbor_radius,
            self.neighbor_radius + 1,
            device=device
        )

        candidate_indices = beam_index.unsqueeze(-1) + offsets.unsqueeze(0)

        candidate_indices = torch.clamp(
            candidate_indices,
            min=0,
            max=num_beams - 1
        )

        return candidate_indices