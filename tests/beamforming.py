import numpy as np
import torch
import matplotlib.pyplot as plt 

import sionna.phy
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.channel.tr38901 import AntennaArray
from sionna.phy.channel import GenerateFlatFadingChannel, ApplyFlatFadingChannel
from sionna.phy.mimo import cbf_precoding_matrix

sionna.phy.config.seed = 42

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1. 기본 설정

# 비교할 안테나 수
num_tx_ant_list = [2, 4, 8, 16]
num_rx_ant = 1

# FR3 후보 주파수
freq_list = [7e9, 15e9, 24e9]

#QPSK
num_bits_per_symbol = 2
coderate = 1.0

#Monte Carlo 심볼 수
batch_size = int(1e7)

#============================================================
# AWGN 생성
def complex_awgn(shape,no,device):
    n_real = torch.randn(shape, device=device)
    n_imag = torch.randn(shape, device=device)

    return (n_real + 1j * n_imag) * np.sqrt(no/2)

#===========================================================
# free space pathloss 계산
def FSPL(f_hz, d_m):
    c = 3e8
    lamda = c / f_hz
    gain = (lamda / (4 * np.pi * d_m)) ** 2
    return gain

#===========================================================
# Sionna 블록 생성
binary_source = BinarySource(device=device)
mapper = Mapper("qam", num_bits_per_symbol, device=device)

demapper = Demapper(
    "app",
    "qam",
    num_bits_per_symbol,
    hard_out=True,
    device=device
)

ebno_dbs = np.arange(-10, 9, 2)

channels = []
apply_channels = []
for num_tx_ant in num_tx_ant_list:
    channel = GenerateFlatFadingChannel(
        num_tx_ant=num_tx_ant,
        num_rx_ant=num_rx_ant,
        device=device
    )
    channels.append(channel)
    apply_channel = ApplyFlatFadingChannel(device=device)
    apply_channels.append(apply_channel)

def simulate(ebno_db, f_hz, channel, apply_channel):
    b = binary_source([batch_size, num_bits_per_symbol])  # [B, 2]
    x = mapper(b)  # [B, 1]
    x = x.squeeze(-1)  # [B]

    no = ebnodb2no(float(ebno_db), num_bits_per_symbol, coderate, device=device)   

    fspl = FSPL(f_hz, 1.0)  # 1m 기준 FSPL

    h = channel(batch_size)  # [B, 1, Nt]

    g = cbf_precoding_matrix(h)  # [B, Nt, 1]

    x_bf = (g[:, :, 0] * x[:, None]).unsqueeze(1)  # [B, 1, Nt]
    y_bf = fspl * apply_channel(x_bf[:, 0, :], h, no)  # [B, 1]

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

    return ber_bf

# =========================
# 4. 안테나 개수 sweep
# =========================
ber_bf_list_2 = []
ber_bf_list_4 = []
ber_bf_list_8 = []
ber_bf_list_16 = []
f = 7e9  # 7GHz 기준으로 시뮬레이션

for num_tx_ant in num_tx_ant_list:
    channel = channels[num_tx_ant_list.index(num_tx_ant)]
    apply_channel = apply_channels[num_tx_ant_list.index(num_tx_ant)]
    for ebno_db in ebno_dbs:
        ber_bf = simulate(ebno_db, f, channel, apply_channel)
        if num_tx_ant == 2:
            ber_bf_list_2.append(ber_bf)
        elif num_tx_ant == 4:
            ber_bf_list_4.append(ber_bf)
        elif num_tx_ant == 8:
            ber_bf_list_8.append(ber_bf)
        elif num_tx_ant == 16:
            ber_bf_list_16.append(ber_bf)

# =========================
# 5. BER 그래프
# =========================
plt.figure(figsize=(7, 5))
plt.semilogy(ebno_dbs, ber_bf_list_2, "s-", label="2-Tx CBF/MRT beamforming")
plt.semilogy(ebno_dbs, ber_bf_list_4, "s-", label="4-Tx CBF/MRT beamforming")
plt.semilogy(ebno_dbs, ber_bf_list_8, "s-", label="8-Tx CBF/MRT beamforming")
plt.semilogy(ebno_dbs, ber_bf_list_16, "s-", label="16-Tx CBF/MRT beamforming")

plt.grid(True, which="both")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BER")
plt.title("Simple Sionna PHY Beamforming Simulation")
plt.legend()
plt.tight_layout()
plt.show()

#============================================================
# 5. 주파수 sweep
ber_bf_list_7 = []
ber_bf_list_15 = []
ber_bf_list_24 = []

num_tx = 16
for f_hz in freq_list:
    channel = channels[num_tx_ant_list.index(num_tx)]
    apply_channel = apply_channels[num_tx_ant_list.index(num_tx)]
    for ebno_db in ebno_dbs:
        ber_bf = simulate(ebno_db, f_hz, channel, apply_channel)
        if f_hz == 7e9:
            ber_bf_list_7.append(ber_bf)
        elif f_hz == 15e9:
            ber_bf_list_15.append(ber_bf)
        elif f_hz == 24e9:
            ber_bf_list_24.append(ber_bf)

#============================================================
# 6. 주파수별 BER 그래프    
plt.figure(figsize=(7, 5))
plt.semilogy(ebno_dbs, ber_bf_list_7, "s-", label="7GHz")
plt.semilogy(ebno_dbs, ber_bf_list_15, "s-", label="15GHz")
plt.semilogy(ebno_dbs, ber_bf_list_24, "s-", label="24GHz")

plt.grid(True, which="both")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BER")
plt.title("Simple Sionna PHY Beamforming Simulation")
plt.legend()
plt.tight_layout()
plt.show()