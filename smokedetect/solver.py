import multiprocessing as mp

import numpy as np
import os

from contextlib import closing
from itertools import product
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix
from typing import Protocol

from breathecam import BreatheCamFrame
from gaussian import Gaussian


class Solver(Protocol):
    def __call__(self, arr0: np.ndarray, arr1: np.ndarray, *args) -> np.ndarray: ...


def _solve_sparse_equations(equations, bounds=(-np.inf, np.inf), verbose=False):
    rowidx = []
    colidx = []
    data = []
    b = []

    for i, eq in enumerate(equations):
        for coef, var in eq[:-1]:
            rowidx.append(i)
            colidx.append(var)
            data.append(coef)
        b.append(eq[-1])

    nvars = max(colidx) + 1
    a = csr_matrix((data, (rowidx, colidx)), shape=(len(b), nvars))
    b = np.array(b)

    return lsq_linear(a, b, bounds, verbose=(1 if verbose else 0)).x


def detect_haze(url0: str, url1: str, *args, solver: Solver, frames: int, subsample: int, augment_frame_offsets: bool = False, dim: (int, int) = (300, 400)):
    if frames < 1:
        raise f"Invalid parameter `frames` :: {frames}"
    
    background = BreatheCamFrame.from_thumbnail(url0).remove_labels().set_size(*dim).image(subsample=subsample)
    hazy = BreatheCamFrame.from_thumbnail(url1).remove_labels().set_size(*dim)

    if frames == 1:
        hazy = hazy.image(subsample=subsample)

        return background, hazy, solver(background, hazy, *args)

    vid0 = np.array([background] * frames)
    vid1 = np.array(list(hazy.get_images(frames, subsample=subsample, augment_frame_offsets=augment_frame_offsets)))

    return vid0, vid1, solver(vid0, vid1, *args)


def solve2d(img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    height, width = img0.shape[:2]
    pixels = height * width * 4

    def v(r, c, p):
        return (r * width + c) * 4 + p

    def equations():
        for i, j, k in product(range(height), range(width), range(3)):
            yield [(1, v(i, j, k)), (img0[i, j, k], v(i, j, 3)), img1[i, j, k]]

        for i, j, k in product(range(height), range(width - 1), [3, 0, 1, 2]):
            yield [(1, v(i, j, k)), (-1, v(i, j, k + 1)), 0]

        for i, j, k, l in product(range(height - 1), range(width), [3, 0, 1, 2]):
            yield [(1, v(i, j, k)), (-1, v(i, j + 1, k)), 0]

    solution = _solve_sparse_equations(equations(), (0, 1))

    for pixel in range(0, pixels * 4, 4):
        solution[pixel + 3] = 1 - solution[pixel]

    return solution.reshape(height, width, 4)


def solve2d_parallel(img0: np.ndarray, img1: np.ndarray, dims: tuple[int, int], cpus: int = -1) -> np.ndarray:
    height, width, plen = img0.shape[0], img0.shape[1], dims[0]
    len_x, len_y = width - plen + 1, height - plen + 1
    patch_weight = Gaussian.kernel2d(plen, sigma=plen / 6.0)
    lx = np.linspace(0, len_x - 1, max(1, int(np.ceil(len_x / (plen - dims[1])))), dtype=int)
    ly = np.linspace(0, len_y - 1, max(1, int(np.ceil(len_y / (plen - dims[1])))), dtype=int)
    os_cpus = os.cpu_count()

    if cpus < 1 or cpus > (os_cpus := os.cpu_count()):
        cpus = os_cpus

    with closing(mp.Pool(cpus)) as pool:
        patches = [
            pool.apply_async(solve2d, args=(img0[y:(y + plen), x:(x + plen)], img1[y:(y + plen), x:(x + plen)]))
            for (y, x) in product(ly, lx)
        ]

    pool.join()

    haze_img, weight_sum = np.zeros((height, width, 4)), np.zeros((height, width, 1))

    for i, (y, x) in enumerate(product(ly, lx)):
        x_end, y_end = x + plen, y + plen
        haze_img[y:y_end, x:x_end] += patches[i].get() * patch_weight[:, :, np.newaxis]
        weight_sum[y:y_end, x:x_end] += patch_weight[:, :, np.newaxis]

    return haze_img / weight_sum


def solve2d_patches(img0: np.ndarray, img1: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    (height, width), patch_width, patch_height = img0.shape[:2], dims[0], dims[0]
    num_patches_x = max(1, int(np.ceil((width - patch_width + 1) / (patch_width - dims[1]))))
    num_patches_y = max(1, int(np.ceil((height - patch_height + 1) / (patch_height - dims[1]))))
    patch_weight = Gaussian.kernel2d(patch_width, sigma=patch_width / 6.0)
    haze_image, weight_sum = np.zeros((height, width, 4)), np.zeros((height, width, 1))

    for y in np.linspace(0, height - patch_height, num_patches_y, dtype=int):
        for x in np.linspace(0, width - patch_width, num_patches_x, dtype=int):
            x_end, y_end = x + patch_width, y + patch_height
            patch_img0, patch_img1 = img0[y:y_end, x:x_end], img1[y:y_end, x:x_end]
            patch_haze = solve2d(patch_img0, patch_img1)
            haze_image[y:y_end, x:x_end] += patch_haze * patch_weight[:, :, np.newaxis]
            weight_sum[y:y_end, x:x_end] += patch_weight[:, :, np.newaxis]

    return haze_image / weight_sum


def solve3d(vid0: np.ndarray, vid1: np.ndarray) -> np.ndarray:
    depth, height, width = vid0.shape[:3]
    pixels_per_frame = height * width * 4
    pixels = pixels_per_frame * depth

    def v(f, r, c, p):
        return (f * pixels_per_frame) + (r * width + c) * 4 + p

    def equations():
        for i, j, k, l in product(range(depth), range(height), range(width), range(3)):
            yield [(1, v(i, j, k, l)), (vid0[i, j, k, l], v(i, j, k, 3)), vid1[i, j, k, l]]

        for i, j, k, l in product(range(depth - 1), range(height), range(width), [3, 0, 1, 2]):
            yield [(1, v(i, j, k, l)), (-1, v(i + 1, j, k, l)), 0]

        for i, j, k, l in product(range(depth), range(height), range(width - 1), [3, 0, 1, 2]):
            yield [(1, v(i, j, k, l)), (-1, v(i, j, k + 1, l)), 0]

        for i, j, k, l in product(range(depth), range(height - 1), range(width), [3, 0, 1, 2]):
            yield [(1, v(i, j, k, l)), (-1, v(i, j + 1, k, l)), 0]

    solution = _solve_sparse_equations(equations(), (0, 1))

    for pixel in range(0, pixels, 4):
        solution[pixel + 3] = 1 - solution[pixel]

    return solution.reshape(depth, height, width, 4)


def solve3d_parallel(vid0: np.ndarray, vid1: np.ndarray, dims: tuple[int, int], cpus: int = -1) -> np.ndarray:
    (depth, height, width), plen = vid0.shape[:-1], dims[0]
    len_x, len_y, len_z = width - plen + 1, height - plen + 1, depth - plen + 1
    patch_weight = Gaussian.kernel3d(plen, sigma=plen / 6.0)
    lx = np.linspace(0, len_x - 1, max(1, int(np.ceil(len_x / (plen - dims[1])))), dtype=int)
    ly = np.linspace(0, len_y - 1, max(1, int(np.ceil(len_y / (plen - dims[1])))), dtype=int)
    lz = np.linspace(0, len_z - 1, max(1, int(np.ceil(len_z / (plen - dims[1])))), dtype=int)
    os_cpus = os.cpu_count()

    if cpus < 1 or cpus > os_cpus:
        cpus = os_cpus

    with closing(mp.Pool(cpus)) as pool:
        patches = pool.map(_3d_parallel_solver, product([[vid0, vid1, plen, patch_weight]], lz, ly, lx))

    pool.join()

    haze_img = np.zeros((depth, height, width, 4))
    weight_sum = np.zeros((depth, height, width, 1))

    for i, (z, y, x) in enumerate(product(lz, ly, lx)):
        x_end, y_end, z_end = x + plen, y + plen, z + plen
        haze_img[z:z_end, y:y_end, x:x_end] += patches[i]
        weight_sum[z:z_end, y:y_end, x:x_end] += patch_weight[:, :, :, np.newaxis]

    return haze_img / weight_sum


def solve3d_patches(vid0: np.ndarray, vid1: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    (depth, height, width), plen, pstride = vid0.shape[:3], dims[0], dims[0] - dims[1]
    num_patches_x = max(1, int(np.ceil((width - plen + 1) / pstride)))
    num_patches_y = max(1, int(np.ceil((height - plen + 1) / pstride)))
    num_patches_z = max(1, int(np.ceil((depth - plen + 1) / pstride)))
    patch_weight = Gaussian.kernel3d(plen, sigma=plen / 6.0)
    haze_image = np.zeros((depth, height, width, 4))
    weight_sum = np.zeros((depth, height, width, 1))

    for z in np.linspace(0, depth - plen, num_patches_z, dtype=int):
        for y in np.linspace(0, height - plen, num_patches_y, dtype=int):
            for x in np.linspace(0, width - plen, num_patches_x, dtype=int):
                x_end, y_end, z_end = x + plen, y + plen, z + plen
                patch_vid0, patch_vid1 = vid0[z:z_end, y:y_end, x:x_end], vid1[z:z_end, y:y_end, x:x_end]
                patch_haze = solve3d(patch_vid0, patch_vid1)
                haze_image[z:z_end, y:y_end, x:x_end] += patch_haze * patch_weight[:, :, :, np.newaxis]
                weight_sum[z:z_end, y:y_end, x:x_end] += patch_weight[:, :, :, np.newaxis]

    return haze_image / weight_sum

def solve3d_2_level_patches(vid0: np.ndarray, vid1: np.ndarray, dims: tuple[int, int, int, int]) -> np.ndarray:
    (depth, height, width), plen, pstride = vid0.shape[:3], dims[0], dims[0] - dims[1]
    plen2, poverlap2 = dims[2], dims[3]
    num_patches_x = max(1, int(np.ceil((width - plen + 1) / pstride)))
    num_patches_y = max(1, int(np.ceil((height - plen + 1) / pstride)))
    num_patches_z = max(1, int(np.ceil((depth - plen + 1) / pstride)))
    patch_weight = Gaussian.kernel3d(plen, sigma=plen / 6.0)
    haze_image = np.zeros((depth, height, width, 4))
    weight_sum = np.zeros((depth, height, width, 1))

    for z in np.linspace(0, depth - plen, num_patches_z, dtype=int):
        for y in np.linspace(0, height - plen, num_patches_y, dtype=int):
            for x in np.linspace(0, width - plen, num_patches_x, dtype=int):
                x_end, y_end, z_end = x + plen, y + plen, z + plen
                patch_vid0, patch_vid1 = vid0[z:z_end, y:y_end, x:x_end], vid1[z:z_end, y:y_end, x:x_end]
                patch_haze = solve3d_patches(patch_vid0, patch_vid1, (plen2, poverlap2))
                haze_image[z:z_end, y:y_end, x:x_end] += patch_haze * patch_weight[:, :, :, np.newaxis]
                weight_sum[z:z_end, y:y_end, x:x_end] += patch_weight[:, :, :, np.newaxis]

    return haze_image / weight_sum


def _3d_parallel_solver(data):
    (i0, i1, n, weight), x, y, z = data
    vid0 = i0[z:(z + n), y:(y + n), x:(x + n)]
    vid1 = i1[z:(z + n), y:(y + n), x:(x + n)]

    return solve3d(vid0, vid1) * weight[:, :, :, np.newaxis]