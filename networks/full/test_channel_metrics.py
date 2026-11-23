import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import trackio

from networks.full import train_channel
from networks.full.channel_metrics import SpectralValidation


class SpectralValidationTest(unittest.TestCase):
    def test_bands_sum_to_validation_mse_for_uneven_batches(self):
        rng = torch.Generator().manual_seed(1)
        for nz in (1, 7, 8):
            with self.subTest(nz=nz):
                spectral = SpectralValidation()
                batch_mses = []
                for b in (4, 1):
                    target = torch.randn(b, 3, 5, 17, nz, generator=rng)
                    prediction = torch.randn(target.shape, generator=rng)
                    spectral.update(prediction, target)
                    batch_mses.append((prediction - target).square().mean().item())
                self.assertAlmostEqual(sum(spectral.scalars().values()), sum(batch_mses) / 2,
                                       places=6)

    def test_dc_and_z_nyquist_go_to_separate_bands(self):
        z = torch.arange(8)
        error = (2.0 + torch.cos(torch.pi * z)).expand(1, 1, 3, 2, 8)
        spectral = SpectralValidation()
        spectral.update(error, torch.zeros_like(error))
        metrics = spectral.scalars()
        self.assertAlmostEqual(metrics["val/spectral_mse/dc"], 4.0, places=6)
        self.assertAlmostEqual(metrics["val/spectral_mse/k_4_to_8"], 1.0, places=6)
        self.assertAlmostEqual(sum(metrics.values()), 5.0, places=6)

    def test_images_compare_mid_y_and_keep_first_samples_across_batches(self):
        target = torch.zeros(1, 1, 5, 3, 5)
        prediction = target.clone()
        prediction[:, :, :, 1, :] = 2.0
        prediction[:, :, :, 0, :] = 100.0  # not the selected y plane
        spectral = SpectralValidation(image_samples=2)
        spectral.update(prediction, target)
        spectral.update(target, prediction)
        spectral.update(prediction, prediction)
        self.assertEqual(spectral.y_index, 1)
        self.assertEqual(len(spectral.slices), 2)
        first, second = spectral.slices
        self.assertAlmostEqual(first[0, 0, 2, 2].item(), 100.0, places=4)
        self.assertEqual(first[1].count_nonzero().item(), 0)
        torch.testing.assert_close(first.flip(0), second)

    def test_tracked_trial_logs_bands_and_images_on_schedule(self):
        rng = torch.Generator().manual_seed(2)
        x = torch.randn(5, 1, 4, 1, 4, generator=rng)
        y = torch.randn(5, 3, 4, 2, 4, generator=rng)
        data = train_channel.TrainingData(
            train=(x, y), val=(x, y), x_mean=torch.zeros(1), x_std=torch.ones(1),
            y_mean=torch.zeros(3), y_std=torch.ones(3),
        )
        with (
            patch.object(train_channel, "NUM_WORKERS", 0),
            patch.object(train_channel, "save_model_to_s3"),
            patch.object(train_channel, "delete_s3"),
            patch.object(trackio, "init", return_value=SimpleNamespace(project="test", name="run")),
            patch.object(trackio, "log") as log,
            patch.object(trackio, "finish"),
        ):
            result = train_channel.run_trial({"c": 2, "n_blocks": 1}, data=data,
                                             device="cpu", epochs=3, trackio_project="test")
        steps = [call.args[0] for call in log.call_args_list]
        self.assertEqual(len(steps), 3)
        for epoch, step in enumerate(steps):
            mse = sum(value for key, value in step.items() if key.startswith("val/spectral_mse/"))
            self.assertAlmostEqual(mse, step["val/mse"], places=6)
            image_keys = [key for key in step if key.startswith("val/spectral_xz/")]
            self.assertEqual(len(image_keys), 0 if epoch == 1 else 2)
            for key in image_keys:
                self.assertIsInstance(step[key], trackio.Image)
        self.assertEqual(result["val_mse"], min(step["val/mse"] for step in steps))
        self.assertEqual(steps[-1]["val/best_so_far"], result["val_mse"])
        self.assertEqual(tuple(result["model"](x[:1]).shape), (1, 3, 4, 2, 4))


if __name__ == "__main__":
    unittest.main()
