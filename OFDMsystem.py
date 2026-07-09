import sionna as sn
import sionna.phy

from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LMMSEEqualizer, LSChannelEstimator
from sionna.phy.channel import GenerateOFDMChannel, ApplyOFDMChannel, OFDMChannel
from sionna.phy.channel.tr38901 import CDL, Antenna, AntennaArray
from sionna.phy.mimo import StreamManagement
from sionna.phy.mapping import BinarySource, Mapper, Demapper

import torch
import numpy as np

# Define the number of UT and BS antennas
NUM_UT = 1
NUM_BS = 1
NUM_UT_ANT = 1
NUM_BS_ANT = 4

# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
NUM_STREAMS_PER_TX = NUM_UT_ANT

# Create an RX-TX association matrix.
# RX_TX_ASSOCIATION[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change.
# For example, considering a system with 2 RX and 4 TX, the RX-TX
# association matrix could be
# [ [1 , 1, 0, 0],
#   [0 , 0, 1, 1] ]
# which indicates that the RX 0 receives from TX 0 and 1, and RX 1 receives from
# TX 2 and 3.
#
# In this notebook, as we have only a single transmitter and receiver,
# the RX-TX association matrix is simply:
RX_TX_ASSOCIATION = np.array([[1]])

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly straightforward. However,
# in setups with multiple transmitters and receivers, the stream management
# is more involved.
STREAM_MANAGEMENT = StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)

RESOURCE_GRID = ResourceGrid(num_ofdm_symbols=14,
                             fft_size=76,
                             subcarrier_spacing=30e3,
                             num_tx=NUM_UT,
                             num_streams_per_tx=NUM_STREAMS_PER_TX,
                             cyclic_prefix_length=6,
                             pilot_pattern="kronecker",
                             pilot_ofdm_symbol_indices=[2, 11])

CARRIER_FREQUENCY = 2.6e9 # Carrier frequency in Hz.
                          # This is needed here to define the antenna element spacing.

UT_ARRAY = Antenna(polarization="single",
                    polarization_type="V",
                    antenna_pattern="38.901",
                    carrier_frequency=CARRIER_FREQUENCY)

BS_ARRAY = AntennaArray(num_rows=1,
                        num_cols=int(NUM_BS_ANT/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=CARRIER_FREQUENCY)

DELAY_SPREAD = 100e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                      # about how to choose this value.

DIRECTION = "downlink"  # The `direction` determines if the UT or BS is transmitting.
                      # In the `uplink`, the UT is transmitting.

CDL_MODEL = "C"       # Suitable values are ["A", "B", "C", "D", "E"]

SPEED = 10.0          # UT speed [m/s]. BSs are always assumed to be fixed.
                      # The direction of travel will chosen randomly.

cdl = CDL(CDL_MODEL,
         DELAY_SPREAD,
         CARRIER_FREQUENCY,
         UT_ARRAY,
         BS_ARRAY,
         DIRECTION,
         min_speed=SPEED)

BATCH_SIZE = 10000 # How many examples are processed by Sionna in parallel

a, tau = cdl(batch_size=BATCH_SIZE,
             num_time_steps=RESOURCE_GRID.num_ofdm_symbols,
             sampling_frequency=1/RESOURCE_GRID.ofdm_symbol_duration)

NUM_BITS_PER_SYMBOL = 2 # QPSK
CODERATE = 1.0

class OFDMSystem(sn.phy.Block):
    """
    Complete OFDM system for link-level simulations.

    This class encapsulates the entire transmitter-channel-receiver chain
    for an OFDM system with LDPC coding over a CDL channel.
    """

    def __init__(self, perfect_csi):
        super().__init__()

        self.perfect_csi = perfect_csi

        # The binary source will create batches of information bits
        self.binary_source = BinarySource()

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = Mapper("qam", NUM_BITS_PER_SYMBOL)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = ResourceGridMapper(RESOURCE_GRID)

        # Frequency domain channel
        self.channel = OFDMChannel(cdl, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=True)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_est = LSChannelEstimator(RESOURCE_GRID, interpolation_type="nn")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(RESOURCE_GRID, STREAM_MANAGEMENT)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("app", "qam", NUM_BITS_PER_SYMBOL)