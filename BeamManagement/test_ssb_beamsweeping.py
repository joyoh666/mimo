import unittest

import torch

from Beamsweeping import BeamSweeper
from codebook import DFTCodebook


class SSBBeamSweepingTest(unittest.TestCase):
    @staticmethod
    def channel_for_beam(codebook, beam_index, dtype=torch.complex64):
        return (
            codebook.get_beam(beam_index)
            .conj()
            .reshape(1, 1, 1, -1)
            .to(dtype)
        )

    def test_ula_oversampling_selects_expected_beam(self):
        codebook = DFTCodebook("ULA", num_ant=4, oversampling=2)
        sweeper = BeamSweeper(codebook)

        beam, power, index = sweeper.select_beam(
            self.channel_for_beam(codebook, 6)
        )

        self.assertEqual(tuple(codebook.get_codebook().shape), (8, 4))
        self.assertEqual(index.item(), 6)
        self.assertEqual(tuple(beam.shape), (1, 1, 4))
        torch.testing.assert_close(beam.squeeze(), codebook.get_beam(6))
        torch.testing.assert_close(power, torch.ones_like(power))

    def test_upa_is_flattened_and_selects_expected_beam(self):
        codebook = DFTCodebook(
            "UPA",
            num_ant_v=2,
            num_ant_h=3,
            oversampling_v=2,
            oversampling_h=1,
        )
        sweeper = BeamSweeper(codebook)

        beam, _, index = sweeper.select_beam(
            self.channel_for_beam(codebook, 10)
        )

        self.assertEqual(tuple(codebook.get_codebook().shape), (12, 6))
        self.assertEqual(index.item(), 10)
        self.assertEqual(tuple(beam.shape), (1, 1, 6))

    def test_dtype_is_promoted_and_output_shape_is_documented(self):
        codebook = DFTCodebook("ULA", num_ant=4)
        sweeper = BeamSweeper(codebook)
        h = self.channel_for_beam(codebook, 2, torch.complex128).expand(
            2, 3, 5, 4
        )

        beam, power, index = sweeper.select_beam(h)

        self.assertEqual(beam.dtype, torch.complex128)
        self.assertEqual(power.dtype, torch.float64)
        self.assertEqual(tuple(power.shape), (2, 3))
        self.assertEqual(tuple(index.shape), (2, 3))
        self.assertTrue(torch.all(index == 2))

    def test_device_and_normalize_arguments_are_applied(self):
        normalized = DFTCodebook("ULA", num_ant=4)
        unnormalized = DFTCodebook("ULA", num_ant=4, normalize=False)

        self.assertEqual(normalized.get_codebook().device, torch.device("cpu"))
        torch.testing.assert_close(
            torch.linalg.vector_norm(normalized.get_beam(0)),
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(unnormalized.get_beam(0)),
            torch.tensor(2.0),
        )

    def test_invalid_channel_shape_and_antenna_count_are_rejected(self):
        sweeper = BeamSweeper(DFTCodebook("ULA", num_ant=4))

        with self.assertRaisesRegex(ValueError, "must have shape"):
            sweeper.select_beam(torch.ones(1, 1, 4))
        with self.assertRaisesRegex(ValueError, "transmit antennas"):
            sweeper.select_beam(torch.ones(1, 1, 1, 3))


if __name__ == "__main__":
    unittest.main()
