# Capture arbitrarily long amounts of videos

import multiprocessing as mp
import numpy as np
import os

from itertools import product
from common import *
from thumbnail_api import Thumbnail


FRAME_OFFSETS = [0.02, 0.11, 0.19, 0.27, 0.36, 0.44, 0.52, 0.61, 0.69, 0.77, 0.86, 0.94]

PatchWidth = 8
PatchHeight = 8
PatchDepth = 8
PatchOverlapX = 4
PatchOverlapY = 4
PatchOverlapZ = 4
PatchStrideX = PatchWidth - PatchOverlapX
PatchStrideY = PatchHeight - PatchOverlapY
PatchStrideZ = PatchDepth - PatchOverlapZ


def get_frames(url: str, subsample: int, seconds: int) -> list[np.array]:
   return get_n_frames(url, subsample, seconds * 12)


def get_n_frames(url: str, subsample: int, count: int) -> list[np.array]:
   thumbnail = Thumbnail.from_url(url)
   time_start = thumbnail.t
   time_base = int(str(time_start)[:-3])
   time_offset_index = FRAME_OFFSETS.index(float(str(time_start)[-3:]))
   frames = [None] * count
   frames[0] = read_image_from_url(url, subsample=subsample) / 255.0

   for i in range(1, count):
      time_offset_index = time_offset_index + 1

      if time_offset_index == 12:
         time_offset_index = 0
         time_base += 1

      thumbnail.t = time_base + FRAME_OFFSETS[time_offset_index]

      frames[i] = read_image_from_url(thumbnail.to_url(), subsample=subsample) / 255.0

   return frames


def stack_frames(url: str, subsample: int, seconds: int) -> np.array:
   return np.array(get_n_frames(url, subsample=subsample, count=seconds * 12)).reshape((32, 50, 12, 4))


def stack_n_frames(url: str, subsample: int, count: int) -> np.array:
   return np.array(get_n_frames(url, subsample=subsample, count=count)).reshape((32, 50, 12, 4))


def solve_3d_haze_detection_v1(vid0, vid1, verbose=False):
    height, width, depth = vid0.shape[:3]
    pixels = height * width * depth

    with Stopwatch(f"solve_3d_haze_detection_v1 {width}x{height}x{depth}, {pixels} pixels", print_stats = verbose):
      # Number of variables: each pixel has 4 variables (o, r, g, b)
        num_vars = pixels * 4
        l1 = pixels * 3
        l2 = num_vars - 4 * height * width 
        l3 = num_vars - 4 * height * depth
        l4 = num_vars - 4 * width * depth

        
        equations = [0] * (l1 + l2 + l3 + l4)
        # Helper functions to get variable indices
        def v(i, j, k, o):
            return ((i * width + j) * height + k) * 4 + l

        idx = 0
        
        # Add the haze model equations

        for i, j, k, l in product(range(height), range(width), range(depth), range(3)):
            equations[idx] = [(1, v(i, j, k, l)), (vid0[i, j, k, l], v(i, j, k, 3)), vid1[i, j, k, l]]
            idx += 1

        # Add the smoothness constraints for opacity and haze color

        for i, j, k, l in product(range(height), range(width), range(depth - 1), [3, 0, 1, 2]): # depth smoothness
            equations[idx] = [(1, v(i, j, k, l)), (-1, v(i, j, k + 1, l)), 0]
            idx += 1

        for i, j, k, l in product(range(height), range(width - 1), range(depth), [3, 0, 1, 2]): # horizontal smoothness
            equations[idx] = [(1, v(i, j, k, l)), (-1, v(i, j + 1, k, l)), 0]
            idx += 1

        for i, j, k, l in product(range(height - 1), range(width), range(depth), [3, 0, 1, 2]): # vertical smoothness
            equations[idx] = [(1, v(i, j, k, l)), (-1, v(i + 1, j, k, l)), 0]
            idx += 1

    # Extract the optimized opacity and haze color
        x_opt = solve_sparse_equations(equations, (0,1))
        haze_image = np.zeros((height, width, depth, 4))

        for (i, j, k) in product(range(height), range(width), range(depth)):
            haze_image[i, j, k, 3] = 1 - x_opt[v(i, j, k, 3)]

            for l in range(3):
                haze_image[i, j, k, l] = x_opt[v(i, j, k, l)]

        return haze_image


def solve_3d_haze_detection_v2(vid0, vid1, verbose=False):
    height, width, depth = vid0.shape[:3]

    with Stopwatch(f"solve_haze_detection_v2 {width}x{height}, {width*height} pixels", print_stats=verbose) as st:
        haze_image = np.zeros((height, width, depth, 4))
        weight_sum = np.zeros((height, width, depth, 1))

        # Calculate minimum number of patches required to completely cover the image, with overlap greater than or equal to patch_overlap_x, patch_overlap_y
        max_x_inclusive = width - PatchWidth + 1
        max_y_inclusive = height - PatchHeight + 1
        max_z_inclusive = depth - PatchDepth + 1

        num_patches_x = max(1, int(np.ceil(max_x_inclusive / PatchStrideX)))
        num_patches_y = max(1, int(np.ceil(max_y_inclusive / PatchStrideY)))
        num_patches_z = max(1, int(np.ceil(max_z_inclusive / PatchStrideZ)))

        # Create a weight matrix for the patch (higher weight in the center, lower at the edges)

        for y in np.linspace(0, height - PatchHeight, num_patches_y, dtype=int):
            for x in np.linspace(0, width - PatchWidth, num_patches_x, dtype=int):
                for z in np.linspace(0, depth - PatchDepth, num_patches_z, dtype=int):
                # Define patch boundaries
                    y_end = y + PatchHeight
                    x_end = x + PatchWidth
                    z_end = z + PatchDepth
                    
                    # Extract patches
                    patch_vid0 = vid0[y:y_end, x:x_end, z:z_end]
                    patch_vid1 = vid1[y:y_end, x:x_end, z:z_end]
                    
                    # Solve haze removal for the patch
                    patch_haze = solve_3d_haze_detection_v1(patch_vid0, patch_vid1, verbose=False)
                    
                    # Add the weighted patch to the final image
                    haze_image[y:y_end, x:x_end, z:z_end] += patch_haze * PatchWeight3D[:, :, np.newaxis]
                    weight_sum[y:y_end, x:x_end, z:z_end] += PatchWeight3D[:, :, :, np.newaxis]

        # Normalize the final image by the total weights
        haze_image /= weight_sum

        return haze_image
