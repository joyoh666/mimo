import numpy as np
import torch
import matplotlib.pyplot as plt

import sionna.phy
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, CDL


# ============================================================
# 1. 기본 설정
# ============================================================
sionna.phy.config.seed = 42
torch.manual_seed(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)

# OFDM parameters
subcarrier_spacing = 120e3
fft_size = 1024
num_ofdm_symbols = 14
sampling_frequency = fft_size * subcarrier_spacing  # 122.88 MHz

# Carrier frequency
carrier_frequency = 7e9

# MISO beamforming setting
num_tx_ant = 8   # 필요하면 2, 4, 8, 16으로 바꿔서 비교 가능
num_rx_ant = 1

# Modulation
num_bits_per_symbol = 2  # QPSK
coderate = 1.0

# CDL setting
cdl_model = "A"          # A: NLOS 성격, D/E: LOS 성격
delay_spread = 300e-9    # wideband 효과를 보기 위해 300 ns 사용
min_speed = 0.0
max_speed = 0.0

# Simulation setting
ebno_dbs = np.arange(0, 18, 2)
batch_size = 128          # OFDM frame 개수
num_mc_batches = 5        # 느리면 1~2로 줄이고, 정확도를 높이려면 늘리기


# ============================================================
# 2. Sionna blocks
# ============================================================
binary_source = BinarySource(device=device)
mapper = Mapper("qam", num_bits_per_symbol, device=device)

demapper = Demapper(
    "app",
    "qam",
    num_bits_per_symbol,
    hard_out=True,
    device=device
)


# ============================================================
# 3. CDL channel 생성
# ============================================================
def make_cdl_channel():
    """
    BS: num_tx_ant개의 ULA 형태 안테나
    UT: 단일 안테나

    여기서는 안테나 pattern 자체보다 wideband/narrowband 차이를 보는 것이 목적이므로
    omni single-polarization antenna를 사용한다.
    """
    # Use AntennaArray with element specifications (TR 38.901 API)
    bs_array = AntennaArray(
        num_rows=1,
        num_cols=num_tx_ant,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
        device=device,
    )

    ut_array = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
        device=device,
    )

    cdl = CDL(
        model=cdl_model,
        delay_spread=delay_spread,
        carrier_frequency=carrier_frequency,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        min_speed=min_speed,
        max_speed=max_speed,
        device=device,
    )

    return cdl


cdl = make_cdl_channel()


# ============================================================
# 4. CDL CIR -> wideband OFDM channel 변환
# ============================================================
def generate_wideband_channel():
    """
    Output:
        h_wb: [batch_size, num_ofdm_symbols, fft_size, num_tx_ant]

    h_wb[b, s, k, :]는 b번째 frame, s번째 OFDM symbol,
    k번째 subcarrier에서의 MISO channel vector.
    """

    # CDL CIR 생성
    a, tau = cdl(
        batch_size=batch_size,
        num_time_steps=num_ofdm_symbols,
        sampling_frequency=sampling_frequency
    )

    # OFDM subcarrier frequency 생성
    frequencies = subcarrier_frequencies(
        fft_size,
        subcarrier_spacing
    ).to(device)

    # CIR -> OFDM frequency response
    h_f = cir_to_ofdm_channel(
        frequencies,
        a,
        tau,
        normalize=False
    )

    # h_f shape:
    # [B, num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant, num_ofdm_symbols, fft_size]
    h_wb = h_f[:, 0, 0, 0, :, :, :]      # [B, Nt, S, K]
    h_wb = h_wb.permute(0, 2, 3, 1)      # [B, S, K, Nt]

    return h_wb, frequencies


# ============================================================
# 5. Narrowband channel 생성
# ============================================================
def wideband_to_narrowband(h_wb, frequencies):
    """
    narrowband 가정:
        전체 대역에서 채널이 flat하다고 보고,
        center subcarrier의 channel을 모든 subcarrier에 복사한다.

    Output:
        h_nb: [B, S, K, Nt]
    """

    center_idx = torch.argmin(torch.abs(frequencies)).item()

    # center subcarrier channel
    h_center = h_wb[:, :, center_idx, :]      # [B, S, Nt]

    # 모든 subcarrier에 같은 channel을 적용
    h_nb = h_center.unsqueeze(2).expand(
        -1, -1, fft_size, -1
    )  # [B, S, K, Nt]

    return h_nb


# ============================================================
# 6. MRT beamforming + QPSK BER 계산
# ============================================================
def simulate_given_channel(h, ebno_db):
    """
    h shape:
        [B, S, K, Nt]

    각 subcarrier마다 MRT beamforming을 적용한다.

    w[k] = h[k]^H / ||h[k]||

    단일 사용자 MISO이므로 effective channel은
        h_eff[k] = h[k] w[k]
    이다.
    """

    no = ebnodb2no(
        ebno_db=float(ebno_db),
        num_bits_per_symbol=num_bits_per_symbol,
        coderate=coderate,
        device=device
    )

    if not torch.is_tensor(no):
        no = torch.tensor(no, dtype=torch.float32, device=device)
    else:
        no = no.to(device)

    # Random bits
    b = binary_source([
        batch_size,
        num_ofdm_symbols,
        fft_size,
        num_bits_per_symbol
    ])

    # QPSK mapping
    x = mapper(b)  # [B, S, K]

    # MRT / CBF precoding
    h_norm = torch.linalg.norm(h, dim=-1, keepdim=True)
    h_norm = torch.clamp(h_norm, min=1e-12)

    w = torch.conj(h) / h_norm  # [B, S, K, Nt]

    # Effective channel
    h_eff = torch.sum(h * w, dim=-1)  # [B, S, K]

    # Received signal
    y_clean = h_eff * x

    noise = torch.sqrt(no / 2.0) * (
        torch.randn_like(y_clean.real) +
        1j * torch.randn_like(y_clean.real)
    )

    y = y_clean + noise

    # Perfect CSI equalization
    y_eq = y / h_eff

    # Equalized noise variance
    no_eff = no / torch.clamp(torch.abs(h_eff) ** 2, min=1e-12)

    # Hard demapping
    b_hat = demapper(y_eq, no_eff)

    ber = compute_ber(b, b_hat).item()

    return ber


# ============================================================
# 7. Eb/N0 sweep
# ============================================================
ber_narrowband = []
ber_wideband = []

for ebno_db in ebno_dbs:
    err_nb = []
    err_wb = []

    for _ in range(num_mc_batches):

        # 같은 CDL realization에서 narrowband/wideband 비교
        h_wb, frequencies = generate_wideband_channel()
        h_nb = wideband_to_narrowband(h_wb, frequencies)

        ber_nb = simulate_given_channel(h_nb, ebno_db)
        ber_wb = simulate_given_channel(h_wb, ebno_db)

        err_nb.append(ber_nb)
        err_wb.append(ber_wb)

    avg_ber_nb = np.mean(err_nb)
    avg_ber_wb = np.mean(err_wb)

    ber_narrowband.append(avg_ber_nb)
    ber_wideband.append(avg_ber_wb)

    print(
        f"Eb/N0 = {ebno_db:>2} dB | "
        f"Narrowband BER = {avg_ber_nb:.3e} | "
        f"Wideband OFDM BER = {avg_ber_wb:.3e}"
    )


# ============================================================
# 8. Plot: narrowband vs wideband BER
# ============================================================
plt.figure(figsize=(8, 5))

plt.semilogy(
    ebno_dbs,
    ber_narrowband,
    "o-",
    label="Narrowband MRT"
)

plt.semilogy(
    ebno_dbs,
    ber_wideband,
    "s-",
    label="Wideband OFDM MRT"
)

plt.grid(True, which="both")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BER")
plt.title(
    f"Narrowband vs Wideband Beamforming BER\n"
    f"fc = 7 GHz, SCS = 120 kHz, FFT = 1024, Nt = {num_tx_ant}"
)
plt.legend()
plt.tight_layout()
plt.show()