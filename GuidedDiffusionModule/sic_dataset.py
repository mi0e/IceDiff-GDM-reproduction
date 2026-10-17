import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset


class SICCropDataset(Dataset):
    """Random 256x256 crops from daily SIC maps at native 25 km (optionally upsampled by `upscale` first).

    Returns (x, {}) with x in [-1, 1], shape [3, size, size] (SIC replicated to RGB, as the
    released UNet/sampler expect 3 channels).
    """

    def __init__(self, npz_path, ocean_mask_path, size=256, upscale=1, min_ocean=0.25,
                 random_flip=True, samples_per_epoch=None):
        data = np.load(npz_path)
        self.sic = data["arr_0"]
        self.size = size
        self.upscale = upscale
        self.random_flip = random_flip
        self.samples_per_epoch = samples_per_epoch or len(self.sic)

        h, w = self.sic.shape[1] * upscale, self.sic.shape[2] * upscale
        assert h >= size and w >= size, f"upscaled map {h}x{w} smaller than crop {size}"
        ocean = th.from_numpy(np.load(ocean_mask_path).astype(np.float32))[None, None]
        ocean = F.interpolate(ocean, scale_factor=upscale, mode="nearest")
        frac = F.avg_pool2d(ocean, size, stride=8)[0, 0].numpy()
        ys, xs = np.nonzero(frac >= min_ocean)
        self.corners = np.stack([ys * 8, xs * 8], 1)
        assert len(self.corners) > 0, "no crop location satisfies min_ocean"

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        day = rng.integers(len(self.sic))
        x = th.from_numpy(self.sic[day].astype(np.float32) / 127.5 - 1.0)[None, None]
        if self.upscale != 1:
            x = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)
        cy, cx = self.corners[rng.integers(len(self.corners))]
        x = x[0, :, cy:cy + self.size, cx:cx + self.size].clamp(-1, 1)
        if self.random_flip and rng.random() < 0.5:
            x = x.flip(-1)
        return x.repeat(3, 1, 1), {}
