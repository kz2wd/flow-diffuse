import unittest
from unittest.mock import patch

import numpy as np
import torch
import zarr

from networks.full import train_channel as trainer


class TrainingSourceTest(unittest.TestCase):
    @staticmethod
    def store(length, marker):
        arr = zarr.zeros(shape=(length, 5, 2, 3, 2), chunks=(16, 5, 2, 3, 2), dtype='f4')
        ix = np.arange(length, dtype=np.float32)
        for ch in range(5):
            arr[:, ch, :, :, :] = np.broadcast_to((marker + ix + 1000 * ch)[:, None, None, None], (length, 2, 3, 2))
        return arr

    def setUp(self):
        self.a1 = self.store(4001, 0)
        self.holdout = self.store(205, 10000)
        self.grid = {'x': np.arange(2), 'y': np.arange(3), 'z': np.arange(2)}

    def test_loads_distinct_stores_and_fits_normalization_on_training_alone(self):
        sources = {'flow_export/A1': self.a1, 'validation/re200-sr1etot': self.holdout}
        with (
            patch.object(trainer, '_open_data_zarr', side_effect=lambda store_path='flow_export/A1': sources[store_path]) as opening,
            patch.object(trainer, '_open_channel_axis', create=True, side_effect=lambda store_path, name: self.grid[name]),
            patch.object(trainer, '_s3fs', side_effect=AssertionError('unexpected S3 client')),
        ):
            data = trainer.load_training_data(max_samples=3, task='wall_pressure_to_velocity', half_y=True)
        self.assertEqual([c.args for c in opening.call_args_list], [('flow_export/A1',), ('validation/re200-sr1etot',)])
        torch.testing.assert_close(data.train[0][:, 0, 0, 0, 0], torch.tensor([3200., 3201., 3202.]))
        torch.testing.assert_close(data.val[0][:, 0, 0, 0, 0], torch.tensor([13200., 13201., 13202.]))
        torch.testing.assert_close(data.val[1][:, 0, 0, 0, 0], torch.tensor([10200., 10201., 10202.]))
        self.assertAlmostEqual(data.x_mean.item(), 3201.)
        self.assertAlmostEqual(data.y_mean[0].item(), 201.)
        self.assertEqual(data.selection['train']['slice'], [200, 203, 1])
        self.assertEqual(data.selection['validation']['slice'], [200, 203, 1])

    def test_cutoff_preserves_a1_training_end_and_test_tail(self):
        selected = trainer.select_training_ranges(4001, 1001)
        self.assertEqual(selected['train'], range(200, 3196))
        self.assertEqual(selected['validation'], range(200, 1001))
        self.assertEqual(selected['test'], trainer.compute_splits(4001)['test'])
        with self.assertRaisesRegex(ValueError, 'spin-up'):
            trainer.select_training_ranges(4001, 200)

    def test_grid_mismatch_stops_before_loading_fields(self):
        with (
            patch.object(trainer, '_open_data_zarr', side_effect=[self.a1, self.holdout]),
            patch.object(trainer, '_open_channel_axis', create=True, side_effect=lambda store, name: self.grid[name] + (0.5 if name == 'y' and store.startswith('validation') else 0)),
            patch.object(trainer, 'load_into_ram', side_effect=AssertionError('unexpected field read')),
        ):
            with self.assertRaisesRegex(ValueError, 'grid.*y'):
                trainer.load_training_data(max_samples=1)


if __name__ == '__main__':
    unittest.main()
