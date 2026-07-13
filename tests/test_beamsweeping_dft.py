import unittest

import torch

from beamsweeping_dft import (
    CodebookPrecoder,
    DFTBeamSweeper,
    DFTBeamTracker,
    DFTCodebook,
)


class DFTBeamSweepingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codebook = DFTCodebook("ula", num_ant=4)

    def channel_for_beam(self, beam_index, batched=True):
        channel = self.codebook.codebook[beam_index].conj().reshape(1, -1)
        if batched:
            channel = channel.unsqueeze(0)
        return channel

    def test_codebook_shape_norm_and_parameter_validation(self):
        self.assertEqual(tuple(self.codebook.codebook.shape), (4, 4))
        norms = torch.linalg.vector_norm(self.codebook.codebook, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))

        with self.assertRaises(ValueError):
            DFTCodebook("ula", num_ant=0)
        with self.assertRaises(ValueError):
            DFTCodebook("ula", num_ant=4, oversampling=0)
        with self.assertRaises(ValueError):
            DFTCodebook("ura", num_ant_v=2, num_ant_h=-1)

    def test_exhaustive_sweep_matches_dft_oracle_and_promotes_dtype(self):
        sweeper = DFTBeamSweeper(self.codebook)
        channels = self.codebook.codebook.conj().unsqueeze(1)

        _, best_index, best_power = sweeper.select_beam(channels)
        torch.testing.assert_close(best_index, torch.arange(4))
        torch.testing.assert_close(best_power, torch.ones(4))

        real_channel = torch.ones(1, 1, 4, dtype=torch.float32)
        _, real_index, _ = sweeper.select_beam(real_channel)
        self.assertEqual(real_index.item(), 0)

        double_channel = self.channel_for_beam(2).to(torch.complex128)
        _, double_index, double_power = sweeper.select_beam(double_channel)
        self.assertEqual(double_index.item(), 2)
        self.assertEqual(double_power.dtype, torch.float64)

        double_matrix = CodebookPrecoder(sweeper).compute_matrix(double_channel)
        self.assertEqual(double_matrix.dtype, torch.complex128)

    def test_unbatched_tracker_selects_and_tracks_correct_beam(self):
        tracker = DFTBeamTracker(
            self.codebook,
            full_sweep_period=10,
            power_drop_threshold_db=1000.0,
        )

        _, initial_index, _, initial_mode = tracker.update(
            self.channel_for_beam(2, batched=False)
        )
        _, tracked_index, _, tracked_mode = tracker.update(
            self.channel_for_beam(3, batched=False)
        )

        self.assertEqual(initial_index.ndim, 0)
        self.assertEqual(initial_index.item(), 2)
        self.assertEqual(initial_mode, "init")
        self.assertEqual(tracked_index.item(), 3)
        self.assertEqual(tracked_mode, "track")

    def test_periodic_sweep_uses_consistent_mode_and_resets_counter(self):
        tracker = DFTBeamTracker(
            self.codebook,
            full_sweep_period=2,
            power_drop_threshold_db=1000.0,
        )

        initial = tracker.update(self.channel_for_beam(0))
        tracked = tracker.update(self.channel_for_beam(0))
        swept = tracker.update(self.channel_for_beam(2))

        self.assertEqual([initial[-1], tracked[-1], swept[-1]], [
            "init",
            "track",
            "full_sweep",
        ])
        self.assertEqual(swept[1].item(), 2)
        self.assertEqual(tracker.frame_count, 0)

    def test_ula_neighbors_wrap_at_dft_grid_boundary(self):
        tracker = DFTBeamTracker(
            self.codebook,
            neighbor_radius=1,
            full_sweep_period=10,
            power_drop_threshold_db=1000.0,
        )
        neighbors = tracker._get_neighbor_indices(torch.tensor([0]))
        self.assertEqual(neighbors.tolist(), [[0, 3, 1]])

        tracker.update(self.channel_for_beam(0))
        _, best_index, _, mode = tracker.update(self.channel_for_beam(3))
        self.assertEqual(best_index.item(), 3)
        self.assertEqual(mode, "track")

    def test_tracking_keeps_current_beam_when_all_candidates_tie(self):
        for codebook in (
            self.codebook,
            DFTCodebook("ura", num_ant_v=4, num_ant_h=5),
        ):
            tracker = DFTBeamTracker(
                codebook,
                neighbor_radius=1,
                full_sweep_period=10,
            )
            zero_channel = torch.zeros(
                1, 1, codebook.num_tx_ant, dtype=codebook.codebook.dtype
            )

            indices = [tracker.update(zero_channel)[1].item() for _ in range(4)]

            self.assertEqual(indices, [0, 0, 0, 0])

    def test_full_sweep_tie_preserves_current_beam(self):
        periodic_tracker = DFTBeamTracker(
            self.codebook,
            full_sweep_period=2,
            power_drop_threshold_db=1000.0,
        )
        periodic_tracker.update(self.channel_for_beam(2))
        periodic_tracker.update(self.channel_for_beam(2))
        zero_channel = torch.zeros(1, 1, 4, dtype=torch.complex64)

        _, outage_index, outage_power, outage_mode = periodic_tracker.update(
            zero_channel
        )
        _, recovered_index, _, recovered_mode = periodic_tracker.update(
            self.channel_for_beam(2)
        )

        self.assertEqual(outage_index.item(), 2)
        self.assertEqual(outage_power.item(), 0.0)
        self.assertEqual(outage_mode, "full_sweep")
        self.assertEqual(recovered_index.item(), 2)
        self.assertEqual(recovered_mode, "track")

        drop_tracker = DFTBeamTracker(
            self.codebook,
            full_sweep_period=10,
            power_drop_threshold_db=0.1,
        )
        drop_tracker.update(self.channel_for_beam(2))
        _, drop_index, _, drop_mode = drop_tracker.update(zero_channel)
        self.assertEqual(drop_index.item(), 2)
        self.assertEqual(drop_mode, "full_sweep")

    def test_weak_channel_full_sweep_does_not_hide_a_better_beam(self):
        for amplitude in (1e-4, 1e-8):
            tracker = DFTBeamTracker(
                self.codebook,
                full_sweep_period=10,
                power_drop_threshold_db=0.1,
            )
            tracker.update(amplitude * self.channel_for_beam(0))

            _, best_index, best_power, mode = tracker.update(
                amplitude * self.channel_for_beam(2)
            )

            self.assertEqual(best_index.item(), 2)
            torch.testing.assert_close(
                best_power,
                torch.tensor([amplitude**2], dtype=best_power.dtype),
                rtol=1e-5,
                atol=0.0,
            )
            self.assertEqual(mode, "full_sweep")

    def test_ura_neighbors_follow_both_periodic_grid_axes(self):
        codebook = DFTCodebook("ura", num_ant_v=4, num_ant_h=5)
        tracker = DFTBeamTracker(
            codebook,
            neighbor_radius=1,
            full_sweep_period=10,
            power_drop_threshold_db=1000.0,
        )

        neighbors = tracker._get_neighbor_indices(torch.tensor([0]))[0]
        expected = {0, 1, 4, 5, 6, 9, 15, 16, 19}
        self.assertEqual(set(neighbors.tolist()), expected)

        initial = codebook.codebook[0].conj().reshape(1, 1, -1)
        vertical_neighbor = codebook.codebook[5].conj().reshape(1, 1, -1)
        tracker.update(initial)
        _, best_index, _, mode = tracker.update(vertical_neighbor)
        self.assertEqual(best_index.item(), 5)
        self.assertEqual(mode, "track")

    def test_power_drop_sweep_restarts_period(self):
        tracker = DFTBeamTracker(
            self.codebook,
            neighbor_radius=1,
            full_sweep_period=3,
            power_drop_threshold_db=0.1,
        )

        tracker.update(self.channel_for_beam(0))
        tracker.update(self.channel_for_beam(0))
        _, recovered_index, _, recovered_mode = tracker.update(
            self.channel_for_beam(2)
        )
        _, next_index, _, next_mode = tracker.update(self.channel_for_beam(2))

        self.assertEqual(recovered_index.item(), 2)
        self.assertEqual(recovered_mode, "full_sweep")
        self.assertEqual(next_index.item(), 2)
        self.assertEqual(next_mode, "track")
        self.assertEqual(tracker.frame_count, 1)

    def test_power_drop_trigger_starts_consistent_batch_wide_sweep(self):
        tracker = DFTBeamTracker(
            self.codebook,
            neighbor_radius=1,
            full_sweep_period=10,
            power_drop_threshold_db=0.1,
        )
        initial = torch.cat(
            [self.channel_for_beam(0), self.channel_for_beam(0)], dim=0
        )
        tracker.update(initial)

        trigger_channel = self.codebook.codebook[2].conj()
        non_trigger_channel = (
            self.codebook.codebook[0].conj()
            + 2.0 * self.codebook.codebook[2].conj()
        )
        next_channels = torch.stack(
            [trigger_channel, non_trigger_channel]
        ).unsqueeze(1)

        _, best_index, best_power, mode = tracker.update(next_channels)

        torch.testing.assert_close(best_index, torch.tensor([2, 2]))
        torch.testing.assert_close(best_power, torch.tensor([1.0, 4.0]))
        self.assertEqual(mode, "full_sweep")
        self.assertEqual(tracker.last_full_sweep_mask.tolist(), [True, True])
        self.assertEqual(tracker.last_power_drop_mask.tolist(), [True, False])
        self.assertEqual(tracker.frame_count, 0)

    def test_tracker_validates_configuration_and_batch_continuity(self):
        with self.assertRaises(ValueError):
            DFTBeamTracker(self.codebook, neighbor_radius=-1)
        with self.assertRaises(ValueError):
            DFTBeamTracker(self.codebook, full_sweep_period=0)
        with self.assertRaises(ValueError):
            DFTBeamTracker(self.codebook, power_drop_threshold_db=float("nan"))

        tracker = DFTBeamTracker(self.codebook)
        batch_two = torch.cat(
            [self.channel_for_beam(0), self.channel_for_beam(1)], dim=0
        )
        tracker.update(batch_two)

        with self.assertRaisesRegex(ValueError, "batch size changed"):
            tracker.update(self.channel_for_beam(0))

        _, index, _ = tracker.initialize(self.channel_for_beam(0))
        self.assertEqual(tuple(index.shape), (1,))

    def test_invalid_update_does_not_advance_sweep_period(self):
        tracker = DFTBeamTracker(
            self.codebook,
            full_sweep_period=2,
            power_drop_threshold_db=1000.0,
        )
        tracker.update(self.channel_for_beam(0))

        invalid_channel = torch.zeros(1, 1, 3, dtype=torch.complex64)
        with self.assertRaisesRegex(ValueError, "num_tx_ant=3"):
            tracker.update(invalid_channel)

        self.assertEqual(tracker.frame_count, 0)
        self.assertEqual(tracker.update(self.channel_for_beam(0))[-1], "track")
        self.assertEqual(tracker.frame_count, 1)

    def test_precoder_broadcasts_batch_beam_over_symbol_axis(self):
        sweeper = DFTBeamSweeper(self.codebook)
        precoder = CodebookPrecoder(sweeper)
        channel = torch.cat(
            [self.channel_for_beam(0), self.channel_for_beam(1)], dim=0
        )
        symbols = torch.arange(10, dtype=torch.float32).reshape(2, 5).to(
            torch.complex64
        )

        precoded = precoder.apply(symbols, channel)

        self.assertEqual(tuple(precoded.shape), (2, 5, 4))
        torch.testing.assert_close(
            precoded[0], symbols[0, :, None] * self.codebook.codebook[0]
        )
        torch.testing.assert_close(
            precoded[1], symbols[1, :, None] * self.codebook.codebook[1]
        )

    def test_ofdm_tracking_averages_resource_dimensions_per_batch(self):
        channels = torch.stack(
            [
                self.codebook.codebook[0].conj(),
                self.codebook.codebook[1].conj(),
            ]
        )
        channels = channels[:, None, None, None, :].expand(-1, 2, 3, 1, -1)
        tracker = DFTBeamTracker(
            self.codebook,
            full_sweep_period=10,
            power_drop_threshold_db=1000.0,
        )

        _, initial_index, _, initial_mode = tracker.update(channels)
        _, tracked_index, _, tracked_mode = tracker.update(channels)

        torch.testing.assert_close(initial_index, torch.tensor([0, 1]))
        torch.testing.assert_close(tracked_index, torch.tensor([0, 1]))
        self.assertEqual(initial_mode, "init")
        self.assertEqual(tracked_mode, "track")


if __name__ == "__main__":
    unittest.main()
