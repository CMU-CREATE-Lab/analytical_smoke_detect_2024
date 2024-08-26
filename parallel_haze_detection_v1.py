import multiprocessing as mp
import os

from itertools import product
import math
import numpy as np

from common import Stopwatch, gaussian_kernel_2d, solve_haze_detection_v1

PatchWidth = 20     #pixels
PatchHeight = 20    #pixels
PatchOverlapX = 10  #pixels
PatchOverlapY = 10  #pixels
PatchStrideX = PatchWidth - PatchOverlapX
PatchStrideY = PatchHeight - PatchOverlapY

def _init(haze_img, weight_sum):
    global Lock, HazeImg, WeightSum, PatchWeight

    HazeImg = haze_img
    WeightSum = weight_sum

    PatchWeight = gaussian_kernel_2d(PatchWidth, sigma=PatchWidth / 6.0)


def create_shared_array(dtype, shape):
    """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
    Note that the buffer values are not initialized.
    """
    dtype = np.dtype(dtype)
    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    # Create the RawArray instance.
    shared_arr = mp.RawArray(cdtype, math.prod(shape))
    # Get a NumPy array view.
    arr = shared_to_numpy(shared_arr, dtype, shape)

    return shared_arr, arr


def shared_to_numpy(shared_arr, dtype, shape):
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)


def solve_patch_haze_detection(idx):
    i0, i1, x, y = idx
    x_end, y_end = x + PatchWidth, y + PatchHeight
    haze_img, weight_sum = shared_to_numpy(*HazeImg), shared_to_numpy(*WeightSum)
    patch_i0, patch_i1 = i0[y:y_end, x:x_end], i1[y:y_end, x:x_end]
    patch_haze = solve_haze_detection_v1(patch_i0, patch_i1, verbose=False)

    haze_img[y:y_end, x:x_end] += patch_haze * PatchWeight[:, :, np.newaxis]
    weight_sum[y:y_end, x:x_end] += PatchWeight[:, :, np.newaxis]


def solve_parallel_haze_detection_v1(i0, i1):
    height, width = i0.shape[:2]

    with Stopwatch(f"solve_haze_detection_v3 {width}x{height}, {width * height} pixels") as st:
        max_x_inclusive = width - PatchWidth + 1
        max_y_inclusive = height - PatchHeight + 1
        num_patches_x = max(1, int(np.ceil(max_x_inclusive / PatchStrideX)))
        num_patches_y = max(1, int(np.ceil(max_y_inclusive / PatchStrideY)))

        lsx = np.linspace(0, width - PatchWidth, num_patches_x, dtype=int)
        lsy = np.linspace(0, height - PatchHeight, num_patches_y, dtype=int)

        (shared_haze_img, haze_img) = create_shared_array(None, (height, width, 4))
        (shared_weight_sum, weight_sum) = create_shared_array(None, (height, width, 1))

        haze_img.flat[:] = np.zeros((height * width) << 2)
        weight_sum.flat[:] = np.zeros(height * width)

        with mp.Pool(
                os.cpu_count(),
                initializer=_init,
                initargs=(
                        (shared_haze_img, None, (height, width, 4)),
                        (shared_weight_sum, None, (height, width, 1))
                )
        ) as pool:
            pool.map(solve_patch_haze_detection, product([i0], [i1], lsx, lsy))

        pool.close()
        pool.join()

        return haze_img / weight_sum
