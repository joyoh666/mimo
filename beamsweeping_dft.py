import math
import operator
from numbers import Real

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
        if not isinstance(array_type, str):
            raise TypeError("array_type must be a string.")

        self.array_type = array_type.lower()
        self.device = device
        self.normalize = normalize
        self.precision = precision

        if self.array_type == "ula":
            if num_ant is None:
                raise ValueError("For ULA, num_ant must be provided.")

            self.num_ant = self._validate_integer("num_ant", num_ant, minimum=1)
            self.num_ant_v = None
            self.num_ant_h = None
            self.oversampling = self._validate_integer(
                "oversampling", oversampling, minimum=1
            )
            self.oversampling_v = None
            self.oversampling_h = None

            codebook = grid_of_beams_dft_ula(
                num_ant=self.num_ant,
                oversmpl=self.oversampling,
                precision=precision,
            )

            self._beam_grid_shape = (self.num_ant * self.oversampling,)

            # Sionna output:
            # [num_ant * oversampling, num_ant]
            self.codebook = codebook.to(device)

        elif self.array_type in ["ura", "upa"]:
            if num_ant_v is None or num_ant_h is None:
                raise ValueError(
                    "For URA/UPA, num_ant_v and num_ant_h must be provided."
                )

            self.num_ant_v = self._validate_integer(
                "num_ant_v", num_ant_v, minimum=1
            )
            self.num_ant_h = self._validate_integer(
                "num_ant_h", num_ant_h, minimum=1
            )
            self.num_ant = self.num_ant_v * self.num_ant_h
            self.oversampling = None
            self.oversampling_v = self._validate_integer(
                "oversampling_v", oversampling_v, minimum=1
            )
            self.oversampling_h = self._validate_integer(
                "oversampling_h", oversampling_h, minimum=1
            )

            codebook = grid_of_beams_dft(
                num_ant_v=self.num_ant_v,
                num_ant_h=self.num_ant_h,
                oversmpl_v=self.oversampling_v,
                oversmpl_h=self.oversampling_h,
                precision=precision,
            )

            self._beam_grid_shape = (
                self.num_ant_v * self.oversampling_v,
                self.num_ant_h * self.oversampling_h,
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
    def _validate_integer(name, value, minimum):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer >= {minimum}.")

        try:
            value = operator.index(value)
        except TypeError as error:
            raise ValueError(
                f"{name} must be an integer >= {minimum}."
            ) from error

        if value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

        return value

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

    @property
    def beam_grid_shape(self):
        """Number of DFT beams along each physical array axis."""

        return self._beam_grid_shape

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
        best_beam: [..., M]
        best_beam_index: [...]
        best_power: [...]
    """

    def __init__(self, codebook:DFTCodebook):
        self.codebook_obj = codebook
        self.codebook = codebook.get_codebook()

    def compute_beam_power(self, h):
        self._check_channel_compatibility(h)

        h, codebook = self._align_operands(h, self.codebook)
        effective_channels = torch.einsum("...km,bm->...kb", h, codebook)

        beam_powers = torch.sum(torch.abs(effective_channels)**2, dim=-2)

        return beam_powers
    
    def select_beam(self, h):
        beam_powers = self.compute_beam_power(h)

        best_power, best_index = torch.max(beam_powers, dim=-1)

        best_beam = self._beam_for_channel(best_index, h)

        return best_beam, best_index, best_power

    def _beam_for_channel(self, beam_index, h):
        _, codebook = self._align_operands(h, self.codebook)
        return codebook[beam_index]

    def _check_channel_compatibility(self, h):
        self._check_channel_shape(h)

        if h.shape[-1] != self.codebook.shape[-1]:
            raise ValueError(
                f"Channel has num_tx_ant={h.shape[-1]}, "
                f"but codebook has num_tx_ant={self.codebook.shape[-1]}."
            )

        if h.device != self.codebook.device:
            raise ValueError(
                f"h is on {h.device}, but the codebook is on "
                f"{self.codebook.device}. Create the codebook on the channel "
                "device or move h first."
            )
    
    @staticmethod
    def _check_channel_shape(h):
        if not torch.is_tensor(h):
            raise TypeError("h must be a torch.Tensor.")

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

        if not (h.is_floating_point() or h.is_complex()):
            raise TypeError("h must have a real or complex floating-point dtype.")

    @staticmethod
    def _align_operands(h, codebook):
        """Align channel/codebook dtype while keeping device errors explicit."""

        if h.device != codebook.device:
            raise ValueError(
                f"h is on {h.device}, but the codebook is on {codebook.device}. "
                "Create the codebook on the channel device or move h first."
            )

        common_dtype = torch.promote_types(h.dtype, codebook.dtype)
        return h.to(dtype=common_dtype), codebook.to(dtype=common_dtype)
        
class CodebookPrecoder:
    def __init__(self, beam_sweeper:DFTBeamSweeper):
        self.beam_sweeper = beam_sweeper

    def compute_matrix(self,h):
        best_beam, _, _ = self.beam_sweeper.select_beam(h) 
        return best_beam.unsqueeze(-1)
    
    def apply(self, x, h):
        """
        Apply the selected beam to one or more symbols.

        ``x`` must start with the channel's beam-selection dimensions. Any
        additional trailing dimensions are treated as symbol/resource axes and
        use the same selected beam.
        """

        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor.")

        best_beam, _, _ = self.beam_sweeper.select_beam(h)

        if x.device != best_beam.device:
            raise ValueError(
                f"x is on {x.device}, but the selected beam is on "
                f"{best_beam.device}."
            )

        beam_selection_shape = best_beam.shape[:-1]
        if x.ndim < len(beam_selection_shape):
            raise ValueError(
                "x has fewer dimensions than the beam-selection dimensions "
                f"{tuple(beam_selection_shape)}."
            )

        for x_size, beam_size in zip(x.shape, beam_selection_shape):
            if x_size != beam_size and x_size != 1 and beam_size != 1:
                raise ValueError(
                    "The leading dimensions of x must be broadcast-compatible "
                    f"with {tuple(beam_selection_shape)}, but got {tuple(x.shape)}."
                )

        for _ in range(x.ndim - len(beam_selection_shape)):
            best_beam = best_beam.unsqueeze(-2)

        x_precoded = best_beam * x.unsqueeze(-1)

        return x_precoded
    
class DFTBeamTracker:
    """
    Track a DFT beam for batched or unbatched channels.

    Batched channels have shape ``[batch, ..., K, M]``. Dimensions between the
    batch and receive dimensions are averaged so that each batch element keeps
    one analog beam. An unbatched ``[K, M]`` channel is handled as batch size 1.

    ``frame_count`` is the number of tracking frames since the most recent full
    sweep. A periodic or power-drop-triggered full sweep resets it to zero.
    Because the tracker exposes one mode and one counter, a power-drop trigger
    in any batch element starts a batch-wide full sweep.
    """

    def __init__(self,
                 codebook: DFTCodebook,
                 neighbor_radius: int=1,
                 full_sweep_period: int=20,
                 power_drop_threshold_db: float=3.0
                 ):
        self.codebook_obj = codebook
        self.codebook = codebook.get_codebook()
        self.sweeper = DFTBeamSweeper(codebook)

        self.neighbor_radius = DFTCodebook._validate_integer(
            "neighbor_radius", neighbor_radius, minimum=0
        )
        self.full_sweep_period = DFTCodebook._validate_integer(
            "full_sweep_period", full_sweep_period, minimum=1
        )

        if (
            isinstance(power_drop_threshold_db, bool)
            or not isinstance(power_drop_threshold_db, Real)
            or not math.isfinite(power_drop_threshold_db)
            or power_drop_threshold_db < 0
        ):
            raise ValueError(
                "power_drop_threshold_db must be a finite non-negative number."
            )
        self.power_drop_threshold_db = float(power_drop_threshold_db)

        self.current_beam_index = None
        self.current_power = None
        self.frame_count = 0
        self._batch_size = None
        self.last_full_sweep_mask = None
        self.last_power_drop_mask = None

    def initialize(self, h):
        """
        최초 beam acquisition.
        전체 코드북을 sweeping해서 초기 beam 선택.
        """

        h, was_unbatched = self._prepare_tracking_channel(h)
        best_beam, best_index, best_power = self._perform_full_sweep(
            h, prefer_current_on_tie=False
        )

        return self._restore_unbatched(
            best_beam, best_index, best_power, was_unbatched
        )

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

        h, was_unbatched = self._prepare_tracking_channel(h)
        self.sweeper._check_channel_compatibility(h)
        if h.shape[0] != self._batch_size:
            raise ValueError(
                "The channel batch size changed from "
                f"{self._batch_size} to {h.shape[0]}. Call initialize() to "
                "start tracking the new batch."
            )

        self.frame_count += 1

        # 주기적으로 full sweep 수행
        if self.frame_count >= self.full_sweep_period:
            best_beam, best_index, best_power = self._perform_full_sweep(
                h, prefer_current_on_tie=True
            )
            best_beam, best_index, best_power = self._restore_unbatched(
                best_beam, best_index, best_power, was_unbatched
            )
            return best_beam, best_index, best_power, "full_sweep"

        # 이전 beam 주변만 탐색
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

        best_beam = self.sweeper._beam_for_channel(best_index, h)

        power_dtype = torch.promote_types(
            self.current_power.dtype, best_local_power.dtype
        )
        previous_power = self.current_power.to(dtype=power_dtype)
        local_power = best_local_power.to(dtype=power_dtype)
        power_floor = torch.finfo(power_dtype).tiny
        power_drop_db = 10.0 * torch.log10(
            torch.clamp(previous_power, min=power_floor)
            / torch.clamp(local_power, min=power_floor)
        )

        need_full_sweep = power_drop_db > self.power_drop_threshold_db

        if torch.any(need_full_sweep):
            best_beam, best_index, best_power = self._perform_full_sweep(
                h,
                prefer_current_on_tie=True,
                power_drop_mask=need_full_sweep,
            )
            mode = "full_sweep"
        else:
            best_power = best_local_power
            self.last_full_sweep_mask = torch.zeros_like(
                best_index, dtype=torch.bool
            )
            self.last_power_drop_mask = torch.zeros_like(
                best_index, dtype=torch.bool
            )
            mode = "track"

        self.current_beam_index = best_index
        self.current_power = best_power

        best_beam, best_index, best_power = self._restore_unbatched(
            best_beam, best_index, best_power, was_unbatched
        )

        return best_beam, best_index, best_power, mode

    def reset(self):
        """Forget all tracking state so that the next update performs a sweep."""

        self.current_beam_index = None
        self.current_power = None
        self.frame_count = 0
        self._batch_size = None
        self.last_full_sweep_mask = None
        self.last_power_drop_mask = None

    def _perform_full_sweep(
        self,
        h,
        prefer_current_on_tie,
        power_drop_mask=None,
    ):
        beam_powers = self.sweeper.compute_beam_power(h)
        beam_powers = self._average_non_beam_dims(beam_powers)

        best_power, best_index = torch.max(beam_powers, dim=-1)

        if prefer_current_on_tie and self.current_beam_index is not None:
            current_power = torch.gather(
                beam_powers,
                dim=-1,
                index=self.current_beam_index.unsqueeze(-1),
            ).squeeze(-1)
            tolerance = (
                8.0
                * torch.finfo(best_power.dtype).eps
                * best_power.abs()
            )
            keep_current = current_power >= best_power - tolerance
            best_index = torch.where(
                keep_current, self.current_beam_index, best_index
            )
            best_power = torch.where(keep_current, current_power, best_power)

        self.current_beam_index = best_index
        self.current_power = best_power
        self.frame_count = 0
        self._batch_size = h.shape[0]
        self.last_full_sweep_mask = torch.ones_like(best_index, dtype=torch.bool)
        if power_drop_mask is None:
            power_drop_mask = torch.zeros_like(best_index, dtype=torch.bool)
        self.last_power_drop_mask = power_drop_mask.clone()

        best_beam = self.sweeper._beam_for_channel(best_index, h)
        return best_beam, best_index, best_power

    @staticmethod
    def _prepare_tracking_channel(h):
        DFTBeamSweeper._check_channel_shape(h)

        if h.ndim == 2:
            return h.unsqueeze(0), True

        if h.shape[0] < 1:
            raise ValueError("The channel batch dimension must be non-empty.")

        return h, False

    @staticmethod
    def _restore_unbatched(best_beam, best_index, best_power, was_unbatched):
        if was_unbatched:
            return best_beam[0], best_index[0], best_power[0]

        return best_beam, best_index, best_power

    def _average_non_beam_dims(self, beam_powers):
        """
        beam_powers shape:
            [batch, ..., num_beams]

        OFDM이면 symbol/subcarrier 축 평균.
        Flat이면 그대로 사용.
        """

        if beam_powers.ndim < 2:
            raise ValueError(
                "beam_powers must include batch and beam dimensions."
            )

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

        if h.ndim < 3:
            raise ValueError("h must have shape [batch, ..., K, M].")

        if candidate_indices.ndim != 2 or candidate_indices.shape[0] != h.shape[0]:
            raise ValueError(
                "candidate_indices must have shape [batch, num_candidates]."
            )

        candidate_codebooks = self.codebook[candidate_indices]
        h, candidate_codebooks = self.sweeper._align_operands(
            h, candidate_codebooks
        )

        effective = torch.einsum(
            "b...km,bcm->b...kc",
            h,
            candidate_codebooks,
        )
        power = torch.sum(torch.abs(effective) ** 2, dim=-2)

        return self._average_non_beam_dims(power)

    def _get_neighbor_indices(self, beam_index):
        """
        현재 beam index 주변 후보 beam index 반환.

        DFT beam grid is periodic. ULA neighbors wrap around the single beam
        axis, while URA/UPA neighbors are formed over both physical grid axes.
        """

        device = beam_index.device
        if beam_index.ndim != 1:
            raise ValueError("beam_index must have shape [batch].")

        grid_shape = self.codebook_obj.beam_grid_shape

        if len(grid_shape) == 1:
            num_beams = grid_shape[0]
            offsets = self._axis_offsets(num_beams, device)
            return torch.remainder(
                beam_index.unsqueeze(-1) + offsets.unsqueeze(0),
                num_beams,
            )

        num_beams_v, num_beams_h = grid_shape
        offsets_v = self._axis_offsets(num_beams_v, device)
        offsets_h = self._axis_offsets(num_beams_h, device)

        current_v = torch.div(beam_index, num_beams_h, rounding_mode="floor")
        current_h = torch.remainder(beam_index, num_beams_h)

        candidate_v = torch.remainder(
            current_v[:, None, None] + offsets_v[None, :, None],
            num_beams_v,
        )
        candidate_h = torch.remainder(
            current_h[:, None, None] + offsets_h[None, None, :],
            num_beams_h,
        )

        candidate_indices = candidate_v * num_beams_h + candidate_h
        return candidate_indices.flatten(start_dim=1)

    def _axis_offsets(self, axis_size, device):
        if 2 * self.neighbor_radius + 1 >= axis_size:
            return torch.arange(axis_size, device=device)

        steps = torch.arange(1, self.neighbor_radius + 1, device=device)
        signed_steps = torch.stack((-steps, steps), dim=-1).flatten()
        return torch.cat((torch.zeros(1, dtype=torch.long, device=device), signed_steps))
