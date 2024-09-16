import multiprocessing as mp
import os
import numpy as np

from contextlib import closing
from functools import partial
from itertools import product
from common import *


def detect_haze_v2(img0, img1, x, y):
    x_end, y_end = x + PatchWidth, y + PatchHeight
    patch0, patch1 = img0[y:y_end, x:x_end], img1[y:y_end, x:x_end]
    patch_haze = solve_haze_detection_v1(patch0, patch1, verbose=False)

    return patch_haze * PatchWeight2D[:, :, np.newaxis]


def detect_haze_parallel_v2(img0, img1):
    height, width = img0.shape[:2]

    with Stopwatch(f"detect_haze_parallel_v2 {width}x{height}, {width * height} pixels") as st:
        max_x_inclusive = width - PatchWidth + 1
        max_y_inclusive = height - PatchHeight + 1
        num_patches_x = max(1, int(np.ceil(max_x_inclusive / PatchStrideX)))
        num_patches_y = max(1, int(np.ceil(max_y_inclusive / PatchStrideY)))

        lsx = np.linspace(0, width - PatchWidth, num_patches_x, dtype=int)
        lsy = np.linspace(0, height - PatchHeight, num_patches_y, dtype=int)

        with closing(mp.Pool(os.cpu_count())) as pool:
            patches = [
                pool.apply_async(detect_haze_v2, args=(img0, img1, x, y)) 
                for (y, x) in product(lsy, lsx)
            ]
            
        pool.join()

        haze_img = np.zeros((height, width, 4))
        weight_sum = np.zeros((height, width, 1))

        for i, (y, x) in enumerate(product(lsy, lsx)):
            x_end, y_end = x + PatchWidth, y + PatchHeight
            haze_img[y:y_end, x:x_end] += patches[i].get()
            weight_sum[y:y_end, x:x_end] += PatchWeight2D[:, :, np.newaxis]

        return haze_img / weight_sum