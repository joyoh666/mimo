import torch
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper

class OFDMResourceGrid:
    """
    QAM data symbols를 OFDM ResourceGrid에 배치하는 모듈.

    역할:
        [batch, num_data_symbols]
        -> [batch, num_tx, num_streams, num_ofdm_symbols, fft_size]

    Sionna ResourceGridMapper는 pilot 위치에 pilot을 자동으로 삽입한다.
    따라서 사용자는 data symbol만 넣으면 된다.
    """

    def __init__(self,
                 num_ofdm_symbols:int,
                 fft_size:int,
                 subcarrier_spacing:float,
                 num_tx:int,
                 num_streams_per_tx:int,
                 cyclic_prefix_length: int,
                num_guard_carriers,
                dc_null: bool,
                pilot_pattern: str,
                pilot_ofdm_symbol_indices,
                device: str):
        
        self.device = device

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=cyclic_prefix_length,
            num_guard_carriers=num_guard_carriers,
            dc_null=dc_null,
            pilot_pattern=pilot_pattern,
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            device=device
        )
        
        self.OFDMmapper = ResourceGridMapper(resource_grid=self.resource_grid,device=device)

    #RE에 들어간 데이터 심볼 개수(pilot 등은 제외, 실제 데이터 개수)
    def num_data_symbols(self):
        return self.resource_grid.num_data_symbols
    
    #OFDM frame 하나를 채우기 위해 필요한 bit 수
    def num_required_bits(self,num_bits_per_symbol:int):
        return self.num_data_symbols() * num_bits_per_symbol
    
    #심볼을 OFDM RE에 배치
    def map_data_symbols(self, data_symbols):
        return self.OFDMmapper(data_symbols)
    
    def build_data_mask(self):
        """
        OFDM grid에서 data RE 위치를 boolean mask로 반환한다.

        output:
            data_mask:
                [num_ofdm_symbols, fft_size]
        """
        type_grid = self.resource_grid.build_type_grid()
        type_grid = torch.as_tensor(type_grid, device=self.device)

        # type_grid shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #
        # 값의 의미:
        # 0: data
        # 1: pilot
        # 2: guard
        # 3: DC

        data_mask = type_grid == 0

        return data_mask