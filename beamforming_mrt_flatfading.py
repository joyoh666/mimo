import numpy as np
import torch
import matplotlib.pyplot as plt

import sionna.phy
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.channel import GenerateFlatFadingChannel, ApplyFlatFadingChannel
from sionna.phy.mimo import cbf_precoding_matrix
from sionna.phy.utils import ebnodb2no, compute_ber

# =========================
# 1. 기본 설정
# =========================
sionna.phy.config.seed = 42

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)

num_tx_ant = 4       # 2x2 Tx array를 단순히 안테나 4개로 모델링
num_rx_ant = 1       # Rx 안테나 1개
num_bits_per_symbol = 4  # QAM
coderate = 1.0       # 채널코딩 없음
batch_size = 200_000 # Eb/N0 하나당 전송할 QPSK 심볼 수

ebno_dbs = np.arange(-8, 9, 2)

# =========================
# 2. Sionna 블록 생성
# =========================
binary_source = BinarySource(device=device)
mapper = Mapper("qam", num_bits_per_symbol, device=device)

# hard_out=True이면 LLR 대신 바로 hard bit decision 출력
demapper = Demapper(
    "app",
    "qam",
    num_bits_per_symbol,
    hard_out=True,
    device=device
)

gen_channel = GenerateFlatFadingChannel(
    num_tx_ant=num_tx_ant,
    num_rx_ant=num_rx_ant,
    device=device
)

apply_channel = ApplyFlatFadingChannel(device=device)


# =========================
# 3. 한 Eb/N0에서 BER 계산하는 함수
# =========================
def simulate_one_ebno(ebno_db):
    """
    비교 대상:
    1) SISO baseline:
       Tx 4개 중 첫 번째 안테나만 사용

    2) CBF/MRT beamforming:
       perfect CSI를 안다고 가정하고 h^H 방향으로 송신
    """

    # 랜덤 비트 생성
    b = binary_source([batch_size, num_bits_per_symbol])  # [B, 4]

    # QAM mapping
    x = mapper(b)        # [B, 1]
    x = x.squeeze(-1)    # [B]

    # noise variance
    no = ebnodb2no(
        ebno_db=float(ebno_db),
        num_bits_per_symbol=num_bits_per_symbol,
        coderate=coderate,
        device=device
    )

    # 같은 채널 realization으로 SISO와 beamforming 비교
    h = gen_channel(batch_size)  # [B, 1, 4]

    # =========================
    # Case A. SISO baseline
    # =========================
    x_siso = torch.zeros(
        [batch_size, num_tx_ant],
        dtype=torch.complex64,
        device=device
    )
    x_siso[:, 0] = x  # 첫 번째 Tx 안테나만 사용

    y_siso = apply_channel(x_siso, h, no)  # [B, 1]

    # SISO effective channel: 첫 번째 안테나 채널
    h_eff_siso = h[:, 0, 0]  # [B]

    # perfect CSI equalization
    y_eq_siso = y_siso.squeeze(-1) / h_eff_siso

    # equalization 후 noise variance는 no / |h_eff|^2
    no_eff_siso = no / torch.clamp(torch.abs(h_eff_siso) ** 2, min=1e-12)

    b_hat_siso = demapper(
        y_eq_siso.unsqueeze(-1),
        no_eff_siso.unsqueeze(-1)
    )

    ber_siso = compute_ber(b, b_hat_siso).item()

    # =========================
    # Case B. CBF/MRT beamforming
    # =========================
    # h: [B, 1, 4]
    # g: [B, 4, 1]
    # 단일 stream에서는 g = h^H / ||h||
    g = cbf_precoding_matrix(h)

    # transmit vector x_bf = g * x
    x_bf = g[:, :, 0] * x[:, None]  # [B, 4]

    y_bf = apply_channel(x_bf, h, no)  # [B, 1]

    # effective channel h_eff = h @ g
    h_eff_bf = torch.matmul(h, g).squeeze(-1).squeeze(-1)  # [B]

    # perfect CSI equalization
    y_eq_bf = y_bf.squeeze(-1) / h_eff_bf
    no_eff_bf = no / torch.clamp(torch.abs(h_eff_bf) ** 2, min=1e-12)

    b_hat_bf = demapper(
        y_eq_bf.unsqueeze(-1),
        no_eff_bf.unsqueeze(-1)
    )

    ber_bf = compute_ber(b, b_hat_bf).item()

    return ber_siso, ber_bf


# =========================
# 4. Eb/N0 sweep
# =========================
ber_siso_list = []
ber_bf_list = []

for ebno_db in ebno_dbs:
    ber_siso, ber_bf = simulate_one_ebno(ebno_db)

    ber_siso_list.append(ber_siso)
    ber_bf_list.append(ber_bf)

    print(
        f"Eb/N0 = {ebno_db:>2} dB | "
        f"SISO BER = {ber_siso:.3e} | "
        f"CBF/MRT BER = {ber_bf:.3e}"
    )


# =========================
# 5. BER 그래프
# =========================
plt.figure(figsize=(7, 5))
plt.semilogy(ebno_dbs, ber_siso_list, "o-", label="SISO baseline")
plt.semilogy(ebno_dbs, ber_bf_list, "s-", label="4-Tx CBF/MRT beamforming")

plt.grid(True, which="both")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BER")
plt.title("Simple Sionna PHY Beamforming Simulation")
plt.legend()
plt.tight_layout()
plt.show()