# Capture arbitrarily long amounts of videos

import numpy as np
import os

from itertools import product, chain
from common import *

PatchWidth = 8
PatchHeight = 8
PatchDepth = 8
PatchOverlapX = 4
PatchOverlapY = 4
PatchOverlapZ = 4
PatchStrideX = PatchWidth - PatchOverlapX
PatchStrideY = PatchHeight - PatchOverlapY
PatchStrideZ = PatchDepth - PatchOverlapZ


def solve_3d_haze_detection_v1(vid0, vid1, verbose=True):
    depth, height, width = vid0.shape[:3]
    frame_pixels = height * width * 4
    pixels = height * width * depth

    with Stopwatch(f"solve_3d_haze_detection_v1 {width}x{height}x{depth}, {pixels} pixels", print_stats = verbose):
        def v(i, j, k, o):
            return (i * frame_pixels) + (j * width + k) * 4 + o

        def equations():
            for i, j, k, l in product(range(depth), range(height), range(width), range(3)):
                yield [(1, v(i, j, k, l)), (vid0[i, j, k, l], v(i, j, k, 3)), vid1[i, j, k, l]]

            for i, j, k, l in product(range(depth - 1), range(height), range(width), [3, 0, 1, 2]): # depth smoothness
                yield [(1, v(i, j, k, l)), (-1, v(i + 1, j, k, l)), 0]

            for i, j, k, l in product(range(depth), range(height), range(width - 1), [3, 0, 1, 2]): # horizontal smoothness
                yield [(1, v(i, j, k, l)), (-1, v(i, j, k + 1, l)), 0]

            for i, j, k, l in product(range(depth), range(height - 1), range(width), [3, 0, 1, 2]): # vertical smoothness
                yield [(1, v(i, j, k, l)), (-1, v(i, j + 1, k, l)), 0]


        haze_image = solve_sparse_equations(equations(), (0,1))
        
        for i in range(0, pixels * 4, 4):
            haze_image[i + 3] = 1 - haze_image[i]
            
        return haze_image.reshape(depth, height, width, 4)


def solve_3d_haze_detection_v2(vid0, vid1, verbose=False):
    depth, height, width = vid0.shape[:3]

    with Stopwatch(f"solve_haze_detection_v2 {depth}x{width}x{height}, {depth*width*height} pixels", print_stats=verbose) as st:
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