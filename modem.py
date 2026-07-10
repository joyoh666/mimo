import torch
from sionna.phy.mapping import Mapper, Demapper, BinarySource


class Modem:
    def __init__(self, num_bits_per_symbol, device):
        self.num_bits_per_symbol = num_bits_per_symbol

        self.binary_source = BinarySource(device=device)

        self.mapper = Mapper("qam", num_bits_per_symbol, device=device)

        self.demapper = Demapper(
            "app",
            "qam",
            num_bits_per_symbol,
            hard_out=True,
            device=device
        )
    
    def generate_bits(self, batch_size):
        return self.binary_source(batch_size)
    
    def modulate(self, bits):
        symbols = self.mapper(bits)
        return symbols.squeeze(-1)

    def demodulate(self, symbols, no_eff):
        bits_hat = self.demapper(symbols, no_eff)
        return bits_hat