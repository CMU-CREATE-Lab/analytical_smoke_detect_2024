# Capture arbitrarily long amounts of videos

import multiprocessing as mp
import numpy as np
import os

from itertools import product, chain
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
    depth, height, width = vid0.shape[:3]
    pixels = height * width * 4

    with Stopwatch(f"solve_3d_haze_detection_v1 {width}x{height}x{depth}, {pixels} pixels", print_stats = verbose):
      # Number of variables: each pixel has 4 variables (o, r, g, b)
        num_vars = pixels * depth
        equations = []
        idx = 0

        # Helper functions to get variable indices
        def v(i, j, k, o):
            return (i * pixels) + (j * width + k) * 4 + o
        
        # Add the haze model equations

        for i, j, k, l in product(range(depth), range(height), range(width), range(3)):
            equations.append([
                (1, v(i, j, k, l)), (vid0[i, j, k, l], v(i, j, k, 3)), vid1[i, j, k, l]
            ])

        # Add the smoothness constraints for opacity and haze color

        for i, j, k, l in product(range(depth - 1), range(height), range(width), [3, 0, 1, 2]): # depth smoothness
            equations.append([
                (1, v(i, j, k, l)), (-1, v(i + 1, j, k, l)), 0
            ])

        for i, j, k, l in product(range(depth), range(height), range(width - 1), [3, 0, 1, 2]): # horizontal smoothness
            equations.append([
                (1, v(i, j, k, l)), (-1, v(i, j, k + 1, l)), 0
            ])


        for i, j, k, l in product(range(depth), range(height - 1), range(width), [3, 0, 1, 2]): # vertical smoothness
            equations.append([
                (1, v(i, j, k, l)), (-1, v(i, j + 1, k, l)), 0
            ])

        # Extract the optimized opacity and haze color

        x_opt = solve_sparse_equations(equations, (0,1))
        haze_image = np.zeros((depth, height, width, 4))

        for (i, j, k) in product(range(depth), range(height), range(width)):
            haze_image[i, j, k, 3] = 1 - x_opt[v(i, j, k, 3)]

            for l in range(3):
                haze_image[i, j, k, l] = x_opt[v(i, j, k, l)]

        return haze_image


def solve_3d_haze_detection_v2(vid0, vid1, verbose=False):
    depth, height, width = vid0.shape[:3]

    with Stopwatch(f"solve_haze_detection_v2 {width}x{height}, {width*height} pixels", print_stats=verbose) as st:
        haze_image = np.zeros((depth, height, width, 4))
        weight_sum = np.zeros((depth, height, width, 1))

        # Calculate minimum number of patches required to completely cover the image, with overlap greater than or equal to patch_overlap_x, patch_overlap_y
        max_x_inclusive = width - PatchWidth + 1
        max_y_inclusive = height - PatchHeight + 1
        max_z_inclusive = depth - PatchDepth + 1

        num_patches_x = max(1, int(np.ceil(max_x_inclusive / PatchStrideX)))
        num_patches_y = max(1, int(np.ceil(max_y_inclusive / PatchStrideY)))
        num_patches_z = max(1, int(np.ceil(max_z_inclusive / PatchStrideZ)))

        # Create a weight matrix for the patch (higher weight in the center, lower at the edges)

        for z in np.linspace(0, depth - PatchDepth, num_patches_z, dtype=int):
            for y in np.linspace(0, height - PatchHeight, num_patches_y, dtype=int):
                for x in np.linspace(0, width - PatchWidth, num_patches_x, dtype=int):
                    # Define patch boundaries
                    x_end = x + PatchWidth
                    y_end = y + PatchHeight
                    z_end = z + PatchDepth
                    
                    # Extract patches
                    patch_vid0 = vid0[z:z_end, y:y_end, x:x_end]
                    patch_vid1 = vid1[z:z_end, y:y_end, x:x_end]
                    
                    # Solve haze removal for the patch
                    patch_haze = solve_3d_haze_detection_v1(patch_vid0, patch_vid1, verbose=verbose)
                    
                    # Add the weighted patch to the final image
                    haze_image[z:z_end, y:y_end, x:x_end] += patch_haze * PatchWeight3D[:, :, :, np.newaxis]
                    weight_sum[z:z_end, y:y_end, x:x_end] += PatchWeight3D[:, :, :, np.newaxis]

        # Normalize the final image by the total weights
        haze_image /= weight_sum

        return haze_image


def solve_3d_haze_detection_v3(vid0, vid1, verbose=False):
    depth, height, width = vid0.shape[:3]
    pixels = height * width * 4

    with Stopwatch(f"solve_3d_haze_detection_v1 {width}x{height}x{depth}, {pixels} pixels", print_stats = verbose):
      # Number of variables: each pixel has 4 variables (o, r, g, b)
        num_vars = pixels * depth
        eqs = (15 * depth - 4) * height * width - 4 * depth * (height + width)
        b = [0] * eqs
        data = [1, -1] * eqs
        rowidx = list(chain(*[(i, i) for i in range(eqs)]))
        colidx = [0] * (2 * eqs)
        eq = 0
        

        # Helper functions to get variable indices
        def v(i, j, k, o):
            return (i * pixels) + (j * width + k) * 4 + o
        
        # Add the haze model equations

        for i, j, k, l in product(range(depth), range(height), range(width), range(3)):
            colidx[2 * eq] = v(i, j, k, l)
            colidx[2 * eq + 1] = v(i, j, k, 3)
            data[2 * eq + 1] = vid0[i, j, k, l]
            b[eq] = vid1[i, j, k, l]
            eq += 1

        # Add the smoothness constraints for opacity and haze color

        for i, j, k, l in product(range(depth - 1), range(height), range(width), [3, 0, 1, 2]): # depth smoothness
            colidx[2 * eq] = v(i, j, k, l)
            colidx[2 * eq + 1] = v(i + 1, j, k, l)
            eq += 1

        for i, j, k, l in product(range(depth), range(height), range(width - 1), [3, 0, 1, 2]): # horizontal smoothness
            colidx[2 * eq] = v(i, j, k, l)
            colidx[2 * eq + 1] = v(i, j, k + 1, l)
            eq += 1

        for i, j, k, l in product(range(depth), range(height - 1), range(width), [3, 0, 1, 2]): # vertical smoothness
            colidx[2 * eq] = v(i, j, k, l)
            colidx[2 * eq + 1] = v(i, j + 1, k, l)
            eq += 1

        # Extract the optimized opacity and haze color

        x_opt = solve_sparse_equations_from(data, rowidx, colidx, b, (0,1))
        haze_image = np.zeros((depth, height, width, 4))

        for (i, j, k) in product(range(depth), range(height), range(width)):
            haze_image[i, j, k, 3] = 1 - x_opt[v(i, j, k, 3)]

            for l in range(3):
                haze_image[i, j, k, l] = x_opt[v(i, j, k, l)]

        return haze_image
