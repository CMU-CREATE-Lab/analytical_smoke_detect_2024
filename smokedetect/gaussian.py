import numpy as np

from scipy.signal.windows import gaussian

class Gaussian:

    @staticmethod
    def kernel2d(kernel_size: int, sigma: float) -> np.ndarray:
        window = gaussian(kernel_size, std=sigma).reshape(kernel_size, 1)

        return np.outer(window, window)

    @staticmethod
    def kernel3d(kernel_size: int, sigma: float) -> np.ndarray:
        window = gaussian(kernel_size, std=sigma).reshape(kernel_size, 1)

        return np.einsum("ai,aj,ak->ijk", window, window, window)
