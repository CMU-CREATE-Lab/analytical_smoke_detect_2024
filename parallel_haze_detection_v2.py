import multiprocessing as mp
import os
import numpy as np

from contextlib import closing
from itertools import chain, product
from scipy.sparse import csr_matrix
from scipy.optimize import lsq_linear
from common import Stopwatch, gaussian_kernel_2d

PatchWidth = 20  # pixels
PatchHeight = 20  # pixels
PatchOverlapX = 10  # pixels
PatchOverlapY = 10  # pixels
PatchStrideX = PatchWidth - PatchOverlapX
PatchStrideY = PatchHeight - PatchOverlapY
PatchWeight = gaussian_kernel_2d(PatchWidth, sigma=PatchWidth / 6.0)


def detect_haze_v2(i0, i1, verbose=True):
    height, width = i0.shape[:2]

    with Stopwatch(f"detect_haze_v2 {width}x{height}, {width * height} pixels", print_stats=verbose):
        x_opt = solve_sparse_equations_v3(i0, i1, (0, 1))

        haze_image = np.zeros((height, width, 4))

        for i in range(height):
            for j in range(width):
                haze_image[i, j, 3] = 1 - x_opt[(i * width + j) * 4 + 3]  # opacity
                for k in range(3):
                    haze_image[i, j, k] = x_opt[(i * width + j) * 4 + k]  # r, g, b

        return haze_image


def detect_patch_haze_v2(data):
    i0, i1, x, y = data
    x_end, y_end = x + PatchWidth, y + PatchHeight
    patch_haze = detect_haze_v2(i0[y:y_end, x:x_end], i1[y:y_end, x:x_end], verbose=False)

    return patch_haze * PatchWeight[:, :, np.newaxis]


def detect_haze_parallel_v2(i0, i1):
    height, width = i0.shape[:2]

    with Stopwatch(f"detect_haze_parallel_v2 {width}x{height}, {width * height} pixels") as st:
        max_x_inclusive = width - PatchWidth + 1
        max_y_inclusive = height - PatchHeight + 1
        num_patches_x = max(1, int(np.ceil(max_x_inclusive / PatchStrideX)))
        num_patches_y = max(1, int(np.ceil(max_y_inclusive / PatchStrideY)))

        lsx = np.linspace(0, width - PatchWidth, num_patches_x, dtype=int)
        lsy = np.linspace(0, height - PatchHeight, num_patches_y, dtype=int)

        with closing(mp.Pool(os.cpu_count())) as pool:

            processes = [
                pool.apply_async(
                    detect_patch_haze_v2, 
                    args=((i0, i1, x, y),)
                ) for y in lsy for x in lsx
            ]

            patches = [p.get() for p in processes]

        haze_img = np.zeros((height, width, 4))
        weight_sum = np.zeros((height, width, 1))

        for i, (y, x) in enumerate(product(lsy, lsx)):
            x_end, y_end = x + PatchWidth, y + PatchHeight
            haze_img[y:y_end, x:x_end] += patches[i]
            weight_sum[y:y_end, x:x_end] += PatchWeight[:, :, np.newaxis]

        return haze_img / weight_sum


def solve_sparse_equations_v2(i0, i1, bounds=(-np.inf, np.inf), verbose=False):
    height, width = i0.shape[:2]
    rowidx, colidx, data, b = [], [], [], []
    n = 0

    with Stopwatch("solve_sparse_equations_v2", print_stats=verbose) as st:
        for i, j, k in product(range(height), range(width), [0, 1, 2]):
            rowidx += [n, n]
            colidx += [(i * width + j) * 4 + k, (i * width + j) * 4 + 3]
            data += [1, i0[i, j, k]]
            b.append(i1[i, j, k])
            n += 1

        for i, j, k in product(range(height), range(width - 1), [3, 0, 1, 2]):
            rowidx += [n, n]
            colidx += [(i * width + j) * 4 + k, (i * width + j + 1) * 4 + k]
            data += [1, -1]
            b.append(0)
            n += 1

        for i, j, k in product(range(height - 1), range(width), [3, 0, 1, 2]):
            rowidx += [n, n]
            colidx += [(i * width + j) * 4 + k, ((i + 1) * width + j) * 4 + k]
            data += [1, -1]
            b.append(0)
            n += 1

        nvars = max(colidx) + 1
        A = csr_matrix((data, (rowidx, colidx)), shape=(n, nvars))
        b = np.array(b)

        res = lsq_linear(A, b, bounds, verbose=(1 if verbose else 0))
        st.set_stats_msg(f'nvars={nvars}, nequations={i + 1}')

        return res.x


def solve_sparse_equations_v3(i0, i1, bounds=(-np.inf, np.inf), verbose=False):
    height, width = i0.shape[:2]
    m = 11 * height * width - 4 * (height + width)
    colidx = []
    rowidx = list(chain(*[(i, i) for i in range(m)]))
    data = list(chain(*[(1, -1) for _ in range(m)]))
    b = [0] * m

    with Stopwatch("solve_sparse_equations_v2", print_stats=verbose) as st:
        for n, (i, j, k) in enumerate(product(range(height), range(width), [0, 1, 2])):
            colidx += [(i * width + j) * 4 + k, (i * width + j) * 4 + 3]
            data[2 * n + 1] = i0[i, j, k]
            b[n] = i1[i, j, k]

        for i, j, k in product(range(height), range(width - 1), [3, 0, 1, 2]):
            colidx += [(i * width + j) * 4 + k, (i * width + j + 1) * 4 + k]

        for i, j, k in product(range(height - 1), range(width), [3, 0, 1, 2]):
            colidx += [(i * width + j) * 4 + k, ((i + 1) * width + j) * 4 + k]

        nvars = max(colidx) + 1
        A = csr_matrix((data, (rowidx, colidx)), shape=(m, nvars))
        b = np.array(b)

        res = lsq_linear(A, b, bounds, verbose=(1 if verbose else 0))
        st.set_stats_msg(f'nvars={nvars}, nequations={i + 1}')

        return res.x