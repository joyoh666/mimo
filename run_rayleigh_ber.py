import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

import sionna.phy
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.channel import GenerateOFDMChannel, ApplyOFDMChannel
from sionna.phy.channel.tr38901 import TDL

# src 폴더를 쓰는 경우
# sys.path.append("../")

from phy.modem import Modem
from phy.ofdm import OFDMResourceGrid
from phy.receiver import OFDMReceiver


# ============================================================
# 0. Basic settings
# ============================================================

sionna.phy.config.seed = 42

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

num_bits_per_symbol = 2  # QPSK
coderate = 1.0

batch_size = 1000
ebno_dbs = np.arange(-5, 21, 2)

carrier_frequency = 7e9

# Wideband OFDM setting
num_ofdm_symbols = 14
fft_size = 256
subcarrier_spacing = 120e3
cyclic_prefix_length = 20

# Frequency-selective Rayleigh-like channel
delay_spread = 300e-9


# ============================================================
# 1. Link simulation function
# ============================================================

def simulate_ofdm_link(
    ebno_db,
    num_tx_ant,
    num_rx_ant,
    num_streams_per_tx,
    label="link",
):
    """
    OFDM link simulation using:
        Modem
        OFDMResourceGrid
        OFDMReceiver

    Cases:
        SISO:
            num_tx_ant = 1
            num_rx_ant = 1
            num_streams_per_tx = 1

        2x2 MIMO spatial multiplexing:
            num_tx_ant = 2
            num_rx_ant = 2
            num_streams_per_tx = 2
    """

    num_tx = 1
    num_rx = 1

    # --------------------------------------------------------
    # Modem
    # --------------------------------------------------------
    modem = Modem(
        num_bits_per_symbol=num_bits_per_symbol,
        device=device,
    )

    # --------------------------------------------------------
    # OFDM resource grid
    # --------------------------------------------------------
    ofdm = OFDMResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=subcarrier_spacing,
        num_tx=num_tx,
        num_streams_per_tx=num_streams_per_tx,
        cyclic_prefix_length=cyclic_prefix_length,
        num_guard_carriers=[12, 11],
        dc_null=True,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11],
        device=device,
    )

    rg = ofdm.resource_grid

    # --------------------------------------------------------
    # Wideband Rayleigh fading channel
    # --------------------------------------------------------
    # TDL-A를 사용해서 delay spread가 있는 frequency-selective fading 생성
    # normalize_channel=True로 평균 채널 파워를 정규화
    channel_model = TDL(
        model="A",
        delay_spread=delay_spread,
        carrier_frequency=carrier_frequency,
        num_tx_ant=num_tx_ant,
        num_rx_ant=num_rx_ant,
        min_speed=0.0,
        max_speed=0.0,
    )

    gen_channel = GenerateOFDMChannel(
        channel_model=channel_model,
        resource_grid=rg,
        normalize_channel=True,
        device=device,
    )

    apply_channel = ApplyOFDMChannel(device=device)

    # --------------------------------------------------------
    # Receiver
    # --------------------------------------------------------
    # 먼저 모듈 연결 확인이 목적이므로 perfect_csi=True로 둠.
    # 나중에 LS channel estimation을 테스트하려면 perfect_csi=False로 변경.
    receiver = OFDMReceiver(
        resource_grid=rg,
        num_rx=num_rx,
        num_tx=num_tx,
        num_streams_per_tx=num_streams_per_tx,
        equalizer_type="lmmse",
        perfect_csi=True,
        device=device,
    )

    # --------------------------------------------------------
    # Noise variance
    # --------------------------------------------------------
    no = ebnodb2no(
        ebno_db=float(ebno_db),
        num_bits_per_symbol=num_bits_per_symbol,
        coderate=coderate,
        resource_grid=rg,
        device=device,
    )

    # --------------------------------------------------------
    # Generate bits
    # --------------------------------------------------------
    num_data_symbols = ofdm.num_data_symbols()

    bits = modem.generate_bits([
        batch_size,
        num_tx,
        num_streams_per_tx,
        num_data_symbols,
        num_bits_per_symbol,
    ])

    # bits:
    # [batch, num_tx, num_streams_per_tx, num_data_symbols, bits_per_symbol]

    # --------------------------------------------------------
    # QAM modulation
    # --------------------------------------------------------
    data_symbols = modem.modulate(bits)

    # data_symbols:
    # [batch, num_tx, num_streams_per_tx, num_data_symbols]

    # --------------------------------------------------------
    # Map QAM symbols to OFDM resource grid
    # --------------------------------------------------------
    x_rg = ofdm.map_data_symbols(data_symbols)
    x_rg = x_rg / torch.sqrt(
    torch.tensor(num_streams_per_tx, dtype=torch.float32, device=device))
    # x_rg:
    # [batch, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
    #
    # 여기서는 num_streams_per_tx == num_tx_ant로 설정했으므로
    # ApplyOFDMChannel의 antenna dimension으로 바로 사용 가능.

    # --------------------------------------------------------
    # Generate channel and transmit
    # --------------------------------------------------------
    h_freq = gen_channel(batch_size)

    y_rg = apply_channel(
        x_rg,
        h_freq,
        no,
    )

    # --------------------------------------------------------
    # Equalization
    # --------------------------------------------------------
    x_hat, no_eff, h_hat, err_var = receiver(
        y_rg=y_rg,
        no=no,
        h_freq=h_freq,
    )

    # x_hat:
    # [batch, num_tx, num_streams_per_tx, num_data_symbols]
    #
    # no_eff:
    # [batch, num_tx, num_streams_per_tx, num_data_symbols]

    # --------------------------------------------------------
    # Demodulation
    # --------------------------------------------------------
    bits_hat = modem.demodulate(
        symbols=x_hat,
        no_eff=no_eff,
    )

    # --------------------------------------------------------
    # BER
    # --------------------------------------------------------
    bits_ref = torch.reshape(bits, [-1])
    bits_est = torch.reshape(bits_hat, [-1])

    ber = compute_ber(bits_ref, bits_est).item()

    print(
        f"{label:20s} | Eb/N0 = {ebno_db:5.1f} dB | BER = {ber:.4e}"
    )

    return ber


# ============================================================
# 2. Run BER curves
# ============================================================

ber_siso = []
ber_mimo_2x2 = []

for ebno_db in ebno_dbs:
    ber = simulate_ofdm_link(
        ebno_db=ebno_db,
        num_tx_ant=1,
        num_rx_ant=1,
        num_streams_per_tx=1,
        label="SISO OFDM",
    )
    ber_siso.append(max(ber, 1e-7))

    ber = simulate_ofdm_link(
        ebno_db=ebno_db,
        num_tx_ant=2,
        num_rx_ant=2,
        num_streams_per_tx=2,
        label="2x2 MIMO-OFDM",
    )
    ber_mimo_2x2.append(max(ber, 1e-7))


# ============================================================
# 3. Plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.semilogy(
    ebno_dbs,
    ber_siso,
    "o-",
    label="SISO OFDM baseline",
)

plt.semilogy(
    ebno_dbs,
    ber_mimo_2x2,
    "s-",
    label="2x2 MIMO-OFDM, LMMSE equalizer",
)

plt.grid(True, which="both")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BER")
plt.title("Wideband Rayleigh OFDM BER: SISO vs 2x2 MIMO")
plt.legend()
plt.ylim(1e-7, 1)
plt.tight_layout()
plt.show()