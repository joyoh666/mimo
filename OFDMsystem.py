import sionna as sn
import sionna.phy

from dataclasses import dataclass
import numpy as np

from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    LMMSEEqualizer,
    LSChannelEstimator,
)
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import CDL, Antenna, AntennaArray
from sionna.phy.mimo import StreamManagement
from sionna.phy.mapping import BinarySource, Mapper, Demapper


@dataclass
class OFDMConfig:
    # Link setup
    num_ut: int = 1
    num_bs: int = 1
    num_ut_ant: int = 1
    num_bs_ant: int = 4

    # Direction
    # "downlink": BS -> UT
    # "uplink": UT -> BS
    direction: str = "downlink"

    # Number of streams per transmitter
    # Beamforming이면 보통 1 stream으로 시작하는 것이 자연스러움
    num_streams_per_tx: int = 1

    # OFDM resource grid
    num_ofdm_symbols: int = 14
    fft_size: int = 76
    subcarrier_spacing: float = 30e3
    cyclic_prefix_length: int = 6
    pilot_pattern: str = "kronecker"
    pilot_ofdm_symbol_indices: tuple = (2, 11)

    # Carrier / channel
    carrier_frequency: float = 2.6e9
    delay_spread: float = 100e-9
    cdl_model: str = "C"
    speed: float = 10.0

    # Antenna configuration
    ut_polarization: str = "single"
    ut_polarization_type: str = "V"

    bs_polarization: str = "dual"
    bs_polarization_type: str = "cross"

    # BS array shape
    # num_bs_ant가 4이고 dual polarization이면
    # 실제 배열은 rows x cols x 2 = 4가 되도록 cols를 자동 계산
    bs_num_rows: int = 1

    antenna_pattern: str = "38.901"

    # Modulation
    num_bits_per_symbol: int = 2  # QPSK
    coderate: float = 1.0

    # Channel option
    add_awgn: bool = True
    normalize_channel: bool = True
    return_channel: bool = True


def make_rx_tx_association(cfg: OFDMConfig):
    """
    RX-TX association matrix 생성.

    downlink:
        TX = BS
        RX = UT

    uplink:
        TX = UT
        RX = BS
    """

    if cfg.direction == "downlink":
        num_rx = cfg.num_ut
        num_tx = cfg.num_bs

    elif cfg.direction == "uplink":
        num_rx = cfg.num_bs
        num_tx = cfg.num_ut

    else:
        raise ValueError("direction must be either 'downlink' or 'uplink'.")

    # 기본값: 모든 RX가 모든 TX로부터 stream을 받는다고 가정
    rx_tx_association = np.ones((num_rx, num_tx), dtype=int)

    return rx_tx_association, num_rx, num_tx


def make_bs_array(cfg: OFDMConfig):
    """
    BS antenna array 생성.

    dual polarization이면 실제 antenna port 수는
        num_rows * num_cols * 2

    single polarization이면
        num_rows * num_cols
    """

    if cfg.bs_polarization == "dual":
        pol_factor = 2
    else:
        pol_factor = 1

    if cfg.num_bs_ant % (cfg.bs_num_rows * pol_factor) != 0:
        raise ValueError(
            "num_bs_ant must be divisible by bs_num_rows * polarization_factor. "
            "For dual polarization, polarization_factor=2."
        )

    bs_num_cols = cfg.num_bs_ant // (cfg.bs_num_rows * pol_factor)

    bs_array = AntennaArray(
        num_rows=cfg.bs_num_rows,
        num_cols=bs_num_cols,
        polarization=cfg.bs_polarization,
        polarization_type=cfg.bs_polarization_type,
        antenna_pattern=cfg.antenna_pattern,
        carrier_frequency=cfg.carrier_frequency,
    )

    return bs_array


def make_ut_array(cfg: OFDMConfig):
    """
    UT antenna 생성.

    지금은 단일 UT antenna를 기본으로 둠.
    UT도 배열로 확장하고 싶으면 AntennaArray로 바꾸면 됨.
    """

    if cfg.num_ut_ant == 1:
        ut_array = Antenna(
            polarization=cfg.ut_polarization,
            polarization_type=cfg.ut_polarization_type,
            antenna_pattern=cfg.antenna_pattern,
            carrier_frequency=cfg.carrier_frequency,
        )
    else:
        # UT도 여러 안테나를 쓰고 싶을 때
        # 여기서는 간단히 1 x num_ut_ant 배열로 설정
        ut_array = AntennaArray(
            num_rows=1,
            num_cols=cfg.num_ut_ant,
            polarization=cfg.ut_polarization,
            polarization_type=cfg.ut_polarization_type,
            antenna_pattern=cfg.antenna_pattern,
            carrier_frequency=cfg.carrier_frequency,
        )

    return ut_array


class OFDMSystem(sn.phy.Block):
    def __init__(self, cfg: OFDMConfig, perfect_csi: bool = False):
        super().__init__()

        self.cfg = cfg
        self.perfect_csi = perfect_csi

        # --------------------------------------------------
        # 1. RX-TX association / Stream management
        # --------------------------------------------------
        rx_tx_association, num_rx, num_tx = make_rx_tx_association(cfg)

        self.rx_tx_association = rx_tx_association
        self.num_rx = num_rx
        self.num_tx = num_tx

        self.stream_management = StreamManagement(
            rx_tx_association,
            cfg.num_streams_per_tx,
        )

        # --------------------------------------------------
        # 2. Resource grid
        # --------------------------------------------------
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=cfg.num_ofdm_symbols,
            fft_size=cfg.fft_size,
            subcarrier_spacing=cfg.subcarrier_spacing,
            num_tx=num_tx,
            num_streams_per_tx=cfg.num_streams_per_tx,
            cyclic_prefix_length=cfg.cyclic_prefix_length,
            pilot_pattern=cfg.pilot_pattern,
            pilot_ofdm_symbol_indices=list(cfg.pilot_ofdm_symbol_indices),
        )

        # --------------------------------------------------
        # 3. Antennas
        # --------------------------------------------------
        self.ut_array = make_ut_array(cfg)
        self.bs_array = make_bs_array(cfg)

        # --------------------------------------------------
        # 4. CDL channel model
        # --------------------------------------------------
        self.cdl = CDL(
            cfg.cdl_model,
            cfg.delay_spread,
            cfg.carrier_frequency,
            self.ut_array,
            self.bs_array,
            cfg.direction,
            min_speed=cfg.speed,
        )

        # --------------------------------------------------
        # 5. Transmitter blocks
        # --------------------------------------------------
        self.binary_source = BinarySource()

        self.mapper = Mapper(
            "qam",
            cfg.num_bits_per_symbol,
        )

        self.rg_mapper = ResourceGridMapper(
            self.resource_grid,
        )

        # --------------------------------------------------
        # 6. OFDM channel
        # --------------------------------------------------
        self.channel = OFDMChannel(
            self.cdl,
            self.resource_grid,
            add_awgn=cfg.add_awgn,
            normalize_channel=cfg.normalize_channel,
            return_channel=cfg.return_channel,
        )

        # --------------------------------------------------
        # 7. Receiver blocks
        # --------------------------------------------------
        self.ls_est = LSChannelEstimator(
            self.resource_grid,
            interpolation_type="nn",
        )

        self.lmmse_equ = LMMSEEqualizer(
            self.resource_grid,
            self.stream_management,
        )

        self.demapper = Demapper(
            "app",
            "qam",
            cfg.num_bits_per_symbol,
        )