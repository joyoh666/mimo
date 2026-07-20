import numpy as np
import torch
import matplotlib.pyplot as plt

import sionna.phy
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.channel.tr38901 import AntennaArray, CDL


# ============================================================
# 1. 기본 설정
# ============================================================
sionna.phy.config.seed = 42

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)

# 비교할 Tx 안테나 수
tx_ant_list = [2, 4, 8, 16]

# FR3 후보 주파수
freq_list = [7e9, 15e9, 24e9]

# 거리 sweep
distance_list = np.arange(20, 221, 20)  # 20 m ~ 220 m

# QPSK
num_bits_per_symbol = 2
coderate = 1.0

# Monte Carlo 심볼 수
# 너무 느리면 10000 정도로 낮춰도 됨
batch_size = 30000

# CDL 설정
cdl_model = "D"          # D/E는 LOS 성분 포함, A/B/C는 NLOS 성격
delay_spread = 100e-9    # 100 ns
sampling_frequency = 15.36e6

# coverage 판단 기준
target_ber = 1e-3


# ============================================================
# 실제 링크 버짓 파라미터
# ============================================================
# ref_rx_ebno_db 같은 기준값을 쓰지 않고,
# 아래 파라미터로 수신 SNR을 계산한다.
#
# tx_power_dbm은 전체 송신 전력이다.
# 안테나 수가 늘어나도 총 송신 전력은 고정된다.
# MRT weight를 ||w||=1로 정규화하므로 array gain은 h_eff에서 자연스럽게 반영된다.
tx_power_dbm = 10.0             # total Tx power [dBm]
tx_element_gain_dbi = 0.0       # 개별 Tx antenna element gain [dBi], array gain 아님
rx_antenna_gain_dbi = 0.0       # Rx antenna gain [dBi]
noise_figure_db = 7.0           # receiver noise figure [dB]
implementation_loss_db = 0.0    # optional implementation loss [dB]

# 잡음전력을 계산할 시스템 대역폭
# 현재 코드는 narrowband single-carrier 형태이므로 sampling_frequency와 동일하게 둔다.
# OFDM 120 kHz x 1024를 쓰려면 system_bandwidth = 122.88e6 으로 바꾸면 된다.
system_bandwidth = sampling_frequency

# path-loss exponent
# LOS에 가깝게 보려면 2.0~2.2
# NLOS에 가깝게 보려면 3.0 이상
path_loss_exponent = 2.2


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
# 3. Link budget / Path loss model
# ============================================================
def ci_path_loss_db(f_hz, d_m, n=2.2):
    """
    Close-In free-space reference distance path-loss model.

    PL(dB) = FSPL(f, 1m) + 10*n*log10(d)
    FSPL(f, 1m) = 32.4 + 20*log10(f_GHz)

    d_m은 1m 이상이라고 가정.
    """
    f_ghz = f_hz / 1e9
    d_m = np.maximum(d_m, 1.0)
    pl_1m = 32.4 + 20.0 * np.log10(f_ghz)
    return pl_1m + 10.0 * n * np.log10(d_m)


def thermal_noise_power_dbm(bandwidth_hz, noise_figure_db):
    """
    Thermal noise power.

    N[dBm] = -174 dBm/Hz + 10log10(B) + NF
    """
    return -174.0 + 10.0 * np.log10(bandwidth_hz) + noise_figure_db


def rx_snr_from_link_budget(f_hz, d_m):
    """
    실제 링크 버짓 기반 pre-beamforming 수신 SNR 계산.

    P_rx[dBm] = P_tx[dBm] + G_tx_elem[dBi] + G_rx[dBi] - PL[dB] - L_impl[dB]
    N[dBm]    = -174 + 10log10(B) + NF
    SNR[dB]   = P_rx[dBm] - N[dBm]

    여기서 계산되는 SNR은 안테나 배열 이득이 들어가기 전의 single-antenna 기준 SNR이다.
    MRT 빔포밍 이득은 시뮬레이션의 h_eff = h w에서 자연스럽게 반영된다.
    """
    path_loss_db = ci_path_loss_db(f_hz, d_m, path_loss_exponent)

    rx_power_dbm = (
        tx_power_dbm
        + tx_element_gain_dbi
        + rx_antenna_gain_dbi
        - path_loss_db
        - implementation_loss_db
    )

    noise_power_dbm = thermal_noise_power_dbm(system_bandwidth, noise_figure_db)
    rx_snr_db = rx_power_dbm - noise_power_dbm

    return rx_snr_db, rx_power_dbm, noise_power_dbm, path_loss_db


def snr_db_to_ebno_db(snr_db):
    """
    Sionna의 ebnodb2no()는 Eb/N0를 입력으로 받는다.
    링크 버짓에서 구한 SNR은 unit-average QPSK symbol 기준 Es/N0로 해석한다.

    Es/N0 = Eb/N0 * bits_per_symbol * coderate
    따라서 Eb/N0[dB] = SNR[dB] - 10log10(bits_per_symbol*coderate)
    """
    return snr_db - 10.0 * np.log10(num_bits_per_symbol * coderate)


# ============================================================
# 4. CDL 채널 생성 함수
# ============================================================
def make_cdl_channel(f_hz, num_tx_ant):
    """
    Tx: ULA 형태로 num_tx_ant개
    Rx: 단일 안테나

    여기서는 안테나 element pattern 효과보다
    안테나 개수에 따른 빔포밍 이득을 보는 것이 목적이라
    omni single-polarization 안테나를 사용.
    """
    bs_array = AntennaArray(
        num_rows=1,
        num_cols=num_tx_ant,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=f_hz,
        device=device
    )

    ut_array = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=f_hz,
        device=device
    )

    cdl = CDL(
        model=cdl_model,
        delay_spread=delay_spread,
        carrier_frequency=f_hz,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        min_speed=0.0,
        max_speed=0.0,
        device=device
    )

    return cdl


def generate_narrowband_cdl_channel(cdl, batch_size, num_tx_ant):
    """
    CDL은 multipath CIR a, tau를 반환한다.

    여기서는 간단한 single-carrier/narrowband 실험으로 만들기 위해
    모든 path coefficient를 합쳐서 등가 MISO 채널 h를 만든다.

    h shape: [batch_size, num_tx_ant]
    """
    a, _ = cdl(
        batch_size=batch_size,
        num_time_steps=1,
        sampling_frequency=sampling_frequency
    )

    # a shape:
    # [B, num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant, num_paths, num_time_steps]
    h = torch.sum(a[:, 0, 0, 0, :, :, 0], dim=-1)

    # 혹시 모를 shape 안정화
    h = torch.reshape(h, [batch_size, num_tx_ant])

    return h


# ============================================================
# 5. MRT / CBF beamforming BER 계산
# ============================================================
def simulate_ber_mrt_cdl(f_hz, d_m, num_tx_ant):
    """
    perfect CSI 기반 MRT/CBF 빔포밍.

    송신 벡터:
        x_vec = w * x

    MRT weight:
        w = h^H / ||h||

    등가 채널:
        h_eff = h w = ||h||

    total transmit power는 안테나 개수와 무관하게 1로 정규화된다.
    따라서 안테나 수가 늘어나면 array gain이 생긴다.
    """

    # 실제 링크 버짓에서 pre-beamforming 수신 SNR 계산
    rx_snr_db, rx_power_dbm, noise_power_dbm, path_loss_db = rx_snr_from_link_budget(
        f_hz=f_hz,
        d_m=d_m
    )

    # 링크 버짓 SNR(Es/N0)을 Sionna가 사용하는 Eb/N0로 변환
    rx_ebno_db = snr_db_to_ebno_db(rx_snr_db)

    no = ebnodb2no(
        ebno_db=float(rx_ebno_db),
        num_bits_per_symbol=num_bits_per_symbol,
        coderate=coderate,
        device=device
    )

    # bits and QPSK symbols
    b = binary_source([batch_size, num_bits_per_symbol])
    x = mapper(b)
    x = torch.squeeze(x, dim=-1)  # [B]

    # CDL channel
    cdl = make_cdl_channel(f_hz, num_tx_ant)
    h = generate_narrowband_cdl_channel(cdl, batch_size, num_tx_ant)

    # MRT / CBF weight
    h_norm = torch.linalg.norm(h, dim=-1, keepdim=True)
    h_norm = torch.clamp(h_norm, min=1e-12)
    w = torch.conj(h) / h_norm  # [B, Nt]

    # effective channel
    h_eff = torch.sum(h * w, dim=-1)  # [B]

    # received signal
    y_clean = h_eff * x

    noise = torch.sqrt(no / 2.0) * (
        torch.randn_like(y_clean.real) + 1j * torch.randn_like(y_clean.real)
    )

    y = y_clean + noise

    # perfect CSI equalization
    y_eq = y / h_eff

    # equalization 후 noise variance
    no_eff = no / torch.clamp(torch.abs(h_eff) ** 2, min=1e-12)

    # hard demapping
    b_hat = demapper(
        y_eq.unsqueeze(-1),
        no_eff.unsqueeze(-1)
    )

    b_hat = torch.reshape(b_hat, b.shape)

    ber = compute_ber(b, b_hat).item()

    return {
        "ber": ber,
        "rx_snr_db": rx_snr_db,
        "rx_ebno_db": rx_ebno_db,
        "rx_power_dbm": rx_power_dbm,
        "noise_power_dbm": noise_power_dbm,
        "path_loss_db": path_loss_db,
    }


# ============================================================
# 6. 전체 sweep
# ============================================================
results = {}

print("\n========== Link budget parameters ==========")
print(f"Tx power              = {tx_power_dbm:.2f} dBm total")
print(f"System bandwidth      = {system_bandwidth/1e6:.2f} MHz")
print(f"Noise figure          = {noise_figure_db:.2f} dB")
print(f"Path-loss exponent    = {path_loss_exponent:.2f}")
print(f"Tx element gain       = {tx_element_gain_dbi:.2f} dBi")
print(f"Rx antenna gain       = {rx_antenna_gain_dbi:.2f} dBi")
print(f"Implementation loss   = {implementation_loss_db:.2f} dB")

for f_hz in freq_list:
    f_ghz = f_hz / 1e9
    results[f_ghz] = {}

    print(f"\n========== Carrier frequency = {f_ghz:.1f} GHz ==========")

    for num_tx_ant in tx_ant_list:
        ber_list = []
        rx_snr_list = []
        rx_ebno_list = []
        rx_power_list = []
        noise_power_list = []
        path_loss_list = []

        print(f"\nTx antennas = {num_tx_ant}")

        for d_m in distance_list:
            out = simulate_ber_mrt_cdl(
                f_hz=f_hz,
                d_m=float(d_m),
                num_tx_ant=num_tx_ant
            )

            ber_list.append(out["ber"])
            rx_snr_list.append(out["rx_snr_db"])
            rx_ebno_list.append(out["rx_ebno_db"])
            rx_power_list.append(out["rx_power_dbm"])
            noise_power_list.append(out["noise_power_dbm"])
            path_loss_list.append(out["path_loss_db"])

            print(
                f"d = {d_m:>3} m | "
                f"PL = {out['path_loss_db']:>6.2f} dB | "
                f"P_rx = {out['rx_power_dbm']:>7.2f} dBm | "
                f"N = {out['noise_power_dbm']:>7.2f} dBm | "
                f"SNR before BF = {out['rx_snr_db']:>6.2f} dB | "
                f"Eb/N0 before BF = {out['rx_ebno_db']:>6.2f} dB | "
                f"BER = {out['ber']:.3e}"
            )

        results[f_ghz][num_tx_ant] = {
            "ber": np.array(ber_list),
            "rx_snr_db": np.array(rx_snr_list),
            "rx_ebno_db": np.array(rx_ebno_list),
            "rx_power_dbm": np.array(rx_power_list),
            "noise_power_dbm": np.array(noise_power_list),
            "path_loss_db": np.array(path_loss_list),
        }


# ============================================================
# 7. Plot 1: 15 GHz에서 안테나 수에 따른 BER vs distance
# ============================================================
plot_freq_ghz = 15.0

plt.figure(figsize=(8, 5))

for num_tx_ant in tx_ant_list:
    plt.semilogy(
        distance_list,
        results[plot_freq_ghz][num_tx_ant]["ber"],
        marker="o",
        label=f"Nt = {num_tx_ant}"
    )

plt.axhline(target_ber, linestyle="--", label=f"Target BER = {target_ber}")
plt.grid(True, which="both")
plt.xlabel("Distance [m]")
plt.ylabel("BER")
plt.title(
    f"Beamforming Gain vs Number of Tx Antennas at {plot_freq_ghz:.0f} GHz\n"
    f"Link budget: Pt={tx_power_dbm:.0f} dBm, B={system_bandwidth/1e6:.2f} MHz, NF={noise_figure_db:.0f} dB"
)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 8. Plot 2: 주파수별, 안테나 수별 coverage distance
# ============================================================
coverage = {}

for f_ghz in results:
    coverage[f_ghz] = []

    for num_tx_ant in tx_ant_list:
        ber_arr = results[f_ghz][num_tx_ant]["ber"]

        valid_distances = distance_list[ber_arr <= target_ber]

        if len(valid_distances) == 0:
            cov_dist = 0.0
        else:
            cov_dist = np.max(valid_distances)

        coverage[f_ghz].append(cov_dist)


plt.figure(figsize=(8, 5))

for f_ghz in sorted(coverage.keys()):
    plt.plot(
        tx_ant_list,
        coverage[f_ghz],
        marker="o",
        label=f"{f_ghz:.0f} GHz"
    )

plt.grid(True)
plt.xlabel("Number of Tx Antennas")
plt.ylabel(f"Coverage distance [m] for BER ≤ {target_ber}")
plt.title("Coverage Recovery by Beamforming in FR3 with Actual Link Budget")
plt.xticks(tx_ant_list)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 9. Plot 3: 24 GHz에서 안테나 수에 따른 BER vs distance
# ============================================================
plot_freq_ghz = 24.0

plt.figure(figsize=(8, 5))

for num_tx_ant in tx_ant_list:
    plt.semilogy(
        distance_list,
        results[plot_freq_ghz][num_tx_ant]["ber"],
        marker="o",
        label=f"Nt = {num_tx_ant}"
    )

plt.axhline(target_ber, linestyle="--", label=f"Target BER = {target_ber}")
plt.grid(True, which="both")
plt.xlabel("Distance [m]")
plt.ylabel("BER")
plt.title(
    f"Beamforming Gain vs Number of Tx Antennas at {plot_freq_ghz:.0f} GHz\n"
    f"Link budget: Pt={tx_power_dbm:.0f} dBm, B={system_bandwidth/1e6:.2f} MHz, NF={noise_figure_db:.0f} dB"
)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. Plot 4: Pre-beamforming SNR vs distance
# ============================================================
# 안테나 수와 무관한 링크 버짓 기반 SNR이므로 Nt=2 결과만 대표로 사용
plt.figure(figsize=(8, 5))

representative_nt = tx_ant_list[0]

for f_ghz in sorted(results.keys()):
    plt.plot(
        distance_list,
        results[f_ghz][representative_nt]["rx_snr_db"],
        marker="o",
        label=f"{f_ghz:.0f} GHz"
    )

plt.grid(True)
plt.xlabel("Distance [m]")
plt.ylabel("Pre-beamforming SNR [dB]")
plt.title("Actual Link Budget SNR before Beamforming")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 11. Coverage 결과 출력
# ============================================================
print("\n========== Coverage distance summary ==========")
print(f"Target BER = {target_ber}")
print(f"Tx power   = {tx_power_dbm:.2f} dBm total")
print(f"Bandwidth  = {system_bandwidth/1e6:.2f} MHz")
print(f"NF         = {noise_figure_db:.2f} dB")

for f_ghz in sorted(coverage.keys()):
    print(f"\nFrequency = {f_ghz:.0f} GHz")
    for nt, cov_d in zip(tx_ant_list, coverage[f_ghz]):
        print(f"  Nt = {nt:>2} | coverage ≈ {cov_d:>5.1f} m")