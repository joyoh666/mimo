import numpy as np
import torch
import matplotlib.pyplot as plt 

import sionna.phy
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.ofdm import ResourceGrid, LSChannelEstimator, ResourceGridMapper
from sionna.phy.channel import GenerateOFDMChannel, ApplyOFDMChannel
from sionna.phy.channel.tr38901 import TDL

sionna.phy.config.seed = 42

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#============================================================
num_tx_ant = 4
num_rx_ant = 1
carrier_frequency = 7e9 

num_bits_per_symbol = 2
coderate = 1.0

batch_size = 10000

#===========================================================
binary_source = BinarySource(device=device)
mapper = Mapper("qam", num_bits_per_symbol, device=device)

demapper = Demapper(
    "app",
    "qam",
    num_bits_per_symbol,
    hard_out=True,
    device=device
)

#===========================================================
num_ofdm_symbols = 14
fft_size = 64
subcarrier_spacing = 30e3
cyclic_prefix_length = 16

rg = ResourceGrid(
    num_ofdm_symbols=num_ofdm_symbols,
    fft_size=fft_size,
    subcarrier_spacing=subcarrier_spacing,
    num_tx=1,
    num_streams_per_tx=1,
    cyclic_prefix_length=cyclic_prefix_length,
    num_guard_carriers=[6, 5],
    dc_null=True,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[2, 11],
    device=device
)

tdl = TDL(
    model="D",  
    delay_spread=100e-9,
    carrier_frequency=carrier_frequency,
    num_tx_ant=num_tx_ant,
    num_rx_ant=num_rx_ant,
    min_speed=0.0,
    max_speed=0.0
)

gen_ofdm_channel = GenerateOFDMChannel(
    channel_model=tdl,
    resource_grid=rg,
    normalize_channel=True,
    device=device
)

#============================================================
ls_est = LSChannelEstimator(
    resource_grid=rg,
    interpolation_type="lin",
    device=device
)

rg_mapper = ResourceGridMapper(resource_grid=rg, device=device)

apply_ofdm_channel = ApplyOFDMChannel(device=device)

#===========================================================

def make_mrt_precoder(h_est):
    norm = torch.sqrt(
        torch.sum(torch.abs(h_est)**2, dim=1, keepdim=True) + 1e-12
    )
    w = torch.conj(h_est) / norm
    return w


#===========================================================
def get_data_mask_from_rg(rg, device):
    """
    ResourceGrid에서 data RE 위치만 추출.
    type_grid 값:
        0: data
        1: pilot
        2: guard
        3: DC
    """
    type_grid = rg.build_type_grid()
    type_grid = torch.as_tensor(type_grid, device=device)

    # shape: [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
    # 여기서는 첫 번째 tx, 첫 번째 stream의 data 위치를 사용
    data_mask = type_grid[0, 0] == 0

    return data_mask


def generate_single_stream_data(data_mask):
    """
    단일 data stream용 QPSK symbol 생성.

    return:
        b       : [batch, num_data_re, num_bits_per_symbol]
        s_grid  : [batch, num_ofdm_symbols, fft_size]
    """
    num_data_re = int(torch.sum(data_mask).item())

    b = binary_source([
        batch_size,
        num_data_re,
        num_bits_per_symbol
    ])

    s = mapper(b)
    s = s.squeeze(-1)

    # s shape: [batch, num_data_re]

    s_grid = torch.zeros(
        [batch_size, num_ofdm_symbols, fft_size],
        dtype=torch.complex64,
        device=device
    )

    s_grid[:, data_mask] = s

    return b, s_grid


def transmit_with_beamforming(h, w, s_grid, no):
    """
    MRT precoder를 data grid에 적용하고 OFDM 채널 통과.

    h:
        [batch, 1, 1, 1, num_tx_ant, num_ofdm_symbols, fft_size]

    w:
        [batch, num_tx_ant, num_ofdm_symbols, fft_size]

    s_grid:
        [batch, num_ofdm_symbols, fft_size]

    return:
        y:
            [batch, 1, 1, num_ofdm_symbols, fft_size]
    """

    # x_i[s,k] = w_i[s,k] * s[s,k]
    x_bf = w * s_grid.unsqueeze(1)

    # x_bf shape:
    # [batch, num_tx_ant, num_ofdm_symbols, fft_size]

    x_bf = x_bf.unsqueeze(1)

    # x_bf shape:
    # [batch, 1, num_tx_ant, num_ofdm_symbols, fft_size]

    y = apply_ofdm_channel(x_bf, h, no)

    return y


def equalize_and_compute_ber(y, h_for_eq, w, b, data_mask, no):
    """
    수신 신호 equalization 후 BER 계산.

    h_for_eq:
        equalization에 사용할 채널.
        perfect CSI case에서는 실제 h_true 사용.
        estimated CSI case에서는 h_hat 사용.
        shape = [batch, num_tx_ant, num_ofdm_symbols, fft_size]
    """

    # effective channel
    h_eff = torch.sum(h_for_eq * w, dim=1)

    # h_eff shape:
    # [batch, num_ofdm_symbols, fft_size]

    y_grid = y[:, 0, 0, :, :]

    x_hat_grid = y_grid / (h_eff + 1e-12)

    # data RE만 추출
    x_hat_data = x_hat_grid[:, data_mask]

    # effective noise variance
    no_eff_grid = no / (torch.abs(h_eff)**2 + 1e-12)
    no_eff_data = no_eff_grid[:, data_mask]

    b_hat = demapper(x_hat_data, no_eff_data)

    b_ref = torch.reshape(b, [b.shape[0], -1])
    b_hat = torch.reshape(b_hat, [b_hat.shape[0], -1])

    ber = compute_ber(b_ref, b_hat)

    return ber

def simulate(ebno_db):

    # =========================================================
    # 1. Noise variance
    # =========================================================
    no = ebnodb2no(
        ebno_db=float(ebno_db),
        num_bits_per_symbol=num_bits_per_symbol,
        coderate=coderate,
        resource_grid=rg,
        device=device
    )

    # =========================================================
    # 2. LS channel estimation용 pilot grid 생성
    # =========================================================
    # 데이터 RE는 0으로 둬도 됨.
    # ResourceGridMapper가 pilot 위치에는 자동으로 pilot을 넣어줌.
    x_ce_data = torch.zeros(
        [batch_size, 1, 1, rg.num_data_symbols],
        dtype=torch.complex64,
        device=device
    )

    x_ce_rg = rg_mapper(x_ce_data)

    # =========================================================
    # 3. 실제 OFDM 채널 생성
    # =========================================================
    h = gen_ofdm_channel(batch_size)

    # h shape:
    # [batch, 1, 1, 1, 4, num_ofdm_symbols, fft_size]

    # =========================================================
    # 4. Pilot 전송 후 LS channel estimation
    # =========================================================
    y_ce = apply_ofdm_channel(x_ce_rg, h, no)

    h_hat, _ = ls_est(y_ce, no)

    # h_hat shape:
    # [batch, 1, 1, 1, 4, num_ofdm_symbols, fft_size]

    h_true_miso = h[:, 0, 0, 0, :, :, :]
    h_hat_miso = h_hat[:, 0, 0, 0, :, :, :]
    h_hat_miso_eff = h_hat[:, 0, 0, 0, :, :, :]

    eff_ind = torch.as_tensor(
    rg.effective_subcarrier_ind,
    dtype=torch.long,
    device=device
    )

    h_hat_miso = torch.zeros_like(h_true_miso)
    h_hat_miso[:, :, :, eff_ind] = h_hat_miso_eff

    # shape:
    # [batch, 4, num_ofdm_symbols, fft_size]

    # =========================================================
    # 5. Perfect CSI MRT / Estimated CSI MRT precoder
    # =========================================================
    w_perfect = make_mrt_precoder(h_true_miso)
    w_est = make_mrt_precoder(h_hat_miso)

    # =========================================================
    # 6. 단일 data stream 생성
    # =========================================================
    data_mask = get_data_mask_from_rg(rg, device)
    b, s_grid = generate_single_stream_data(data_mask)

    # =========================================================
    # 7. Perfect CSI MRT 전송
    # =========================================================
    y_perfect = transmit_with_beamforming(
        h=h,
        w=w_perfect,
        s_grid=s_grid,
        no=no
    )

    ber_perfect = equalize_and_compute_ber(
        y=y_perfect,
        h_for_eq=h_true_miso,
        w=w_perfect,
        b=b,
        data_mask=data_mask,
        no=no
    )

    # =========================================================
    # 8. Estimated CSI MRT 전송
    # =========================================================
    y_est = transmit_with_beamforming(
        h=h,
        w=w_est,
        s_grid=s_grid,
        no=no
    )

    ber_est = equalize_and_compute_ber(
        y=y_est,
        h_for_eq=h_hat_miso,
        w=w_est,
        b=b,
        data_mask=data_mask,
        no=no
    )

    return ber_perfect.item(), ber_est.item()

ebno_dbs = np.arange(-10, 8, 2)

ber_perfect_list = []
ber_est_list = []

for ebno_db in ebno_dbs:
    print(f"Simulating Eb/N0 = {ebno_db} dB")

    ber_perfect, ber_est = simulate(ebno_db)

    ber_perfect_list.append(ber_perfect)
    ber_est_list.append(ber_est)

    print(f"  Perfect CSI MRT BER  : {ber_perfect:.4e}")
    print(f"  Estimated CSI MRT BER: {ber_est:.4e}")

plt.figure(figsize=(8, 5))
plt.semilogy(ebno_dbs, ber_perfect_list, "o-", label="Perfect CSI MRT")
plt.semilogy(ebno_dbs, ber_est_list, "s-", label="Estimated CSI MRT")

plt.grid(True, which="both")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BER")
plt.title("OFDM MRT Beamforming: Perfect CSI vs Estimated CSI")
plt.legend()
plt.ylim(1e-7, 1)
plt.show()