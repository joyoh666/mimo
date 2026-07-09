import numpy as np
import torch
import matplotlib.pyplot as plt
import sionna.phy

from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.mimo import grid_of_beams_dft_ula
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import GenerateOFDMChannel, ApplyOFDMChannel

sionna.phy.config.seed = 42

device = "cuda:0" if torch.cuda.is_available() else "cpu"

num_tx_ant = 16
num_rx_ant = 1
num_bits_per_symbol = 2
coderate = 1.0
batch_size = 10000

num_time_steps = 200      # slot 또는 frame index로 생각
symbols_per_step = 1000   # 각 time step에서 전송할 QPSK 심볼 수
num_beams = 32

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
carrier_frequency = 7e9
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
    min_speed=1.0,
    max_speed=10.0
)

gen_ofdm_channel = GenerateOFDMChannel(
    channel_model=tdl,
    resource_grid=rg,
    normalize_channel=True,
    device=device
)

apply_ofdm_channel = ApplyOFDMChannel(device=device)

#===========================================================
def make_codebook(num_tx_ant, num_beams):
    codebook = grid_of_beams_dft_ula(num_tx_ant, (num_beams/num_tx_ant), device=device)
    return codebook

def beam_sweep(h, codebook):
    gain = torch.abs(torch.matmul(h, codebook.T))
    power = gain ** 2

    best_power = best_index = torch.max(power, dim=1)

    return best_power, best_index

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

def transmit_with_beamsweep(h, codebook, s_grid, no):
    """
    Beam sweep를 통해 best beam을 선택하고,
    선택된 beam을 precoding weight로 사용하여 data grid를 전송.

    h: [B, S, K, Nt]
    codebook: [num_beams, Nt]
    s_grid: [B, S, K]
    no: [B, S, K]
    """

    # 각 subcarrier마다 best beam 선택
    _ , best_index = beam_sweep(h[:, :, :, :], codebook)

    # best beam index를 precoding weight로 변환
    w = codebook[best_index]  # [B, S, Nt]

    # precoding 적용
    s_grid_precoded = torch.einsum("bsk,bskn->bsk", s_grid.unsqueeze(-1), w)

    # 채널 통과
    y_grid = apply_ofdm_channel(s_grid_precoded, h, no)

    return y_grid, w

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