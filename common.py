import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import lsq_linear
import requests
from PIL import Image
import io
from contextlib import contextmanager
import sys
import time
import psutil


# Stopwatch takes a name, or a label_fn and a boolean enable.  If enable is False, the context manager does nothing.  Otherwise, it measures the time taken to execute the code block and prints the time taken.
# label_fn is a function that takes a dataclass with fields wall_time, cpu_time, avg_cpu_used, and cpu_count, and returns a string to output

class Stopwatch:
    def __init__(self, name, print_stats=True):
        self.name = name
        self.stats_msg = None
        self.print_stats = print_stats

    def __enter__(self):
        self.start_wall_time = time.time()
        self.start_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
        self.start_cpu_count = psutil.cpu_count()
        return self

    def set_stats_msg(self, stats_msg):
        self.stats_msg = stats_msg
    
    def start(self):
        self.start_wall_time = time.time()
        self.start_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
        self.start_cpu_count = psutil.cpu_count()
        
    def wall_elapsed(self):
        return time.time() - self.start_wall_time

    def cpu_elapsed(self):
        end_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
        return end_cpu_time - self.start_cpu_time

    def __exit__(self, type, value, traceback):
        msg =  self.stats_msg = f'{self.name}: {self.wall_elapsed():.1f} seconds, {self.cpu_elapsed():.1f} seconds CPU' 
        if self.stats_msg is not None:
            msg += f', {self.stats_msg}'

        if self.print_stats:
            sys.stdout.write('%s: %s\n' % (self.name, self.stats_msg))
            sys.stdout.flush()

def solve_sparse_equations(equations, bounds=(-np.inf, np.inf), verbose=False):
    rowidx = []
    colidx = []
    data = []
    b = []

    with Stopwatch("Solving sparse equations", print_stats=verbose) as st:
        for i, eq in enumerate(equations):
            for coef, var in eq[:-1]:
                rowidx.append(i)
                colidx.append(var)
                data.append(coef)
            b.append(eq[-1])
        nvars = max(colidx) + 1
        A = csr_matrix((data, (rowidx, colidx)), shape=(len(equations), nvars))
        b = np.array(b)

        res = lsq_linear(A, b, bounds, verbose=(1 if verbose else 0))
        st.set_stats_msg(f'nvars={nvars}, nequations={len(equations)}')

        return res.x

def solve_haze_detection_v1(i0, i1, verbose=True):
    height, width = i0.shape[:2]
    with Stopwatch(f"solve_haze_detection {width}x{height}, {width*height} pixels", print_stats = verbose):
        # Number of variables: each pixel has 4 variables (o, r, g, b)
        num_vars = height * width * 4

        # Helper functions to get variable indices
        def h(i, j, k):
            return (i * width + j) * 4 + k
        
        def o(i, j):
            return h(i, j, 3)


        equations = []
        # Add the haze model equations
        for i in range(height):
            for j in range(width):
                for k in range(3):  # RGB
                    # h[i,j,[0:3]] + i0[i,j,[0:3]] * h[i,j,3]  = i1[i,j]
                    equations.append([(1, h(i, j, k)), (i0[i, j, k], o(i, j)), i1[i, j, k]])
        # Add the smoothness constraints for opacity and haze color
        for i in range(height):
            for j in range(width - 1): # horizontal smoothness
                equations.append([(1, o(i, j)), (-1, o(i, j + 1)), 0])
                for k in range(3):
                    equations.append([(1, h(i, j, k)), (-1, h(i, j + 1, k)), 0])
        for i in range(height - 1): # vertical smoothness
            for j in range(width):
                equations.append([(1, o(i, j)), (-1, o(i + 1, j)), 0])
                for k in range(3):
                    equations.append([(1, h(i, j, k)), (-1, h(i + 1, j, k)), 0])

        x_opt = solve_sparse_equations(equations, (0,1))

        # Extract the optimized opacity and haze color

        haze_image = np.zeros((height, width, 4))

        for i in range(height):
            for j in range(width):
                haze_image[i, j, 3] = 1 - x_opt[o(i, j)]  # opacity
                for k in range(3):
                    haze_image[i, j, k] = x_opt[h(i, j, k)]  # r, g, b

        return haze_image

# solve_haze_removal_v1 becomes much slower as i0 and i1 become larger.  Early testing suggests a 5000x slowdown once i0 and i1 become larger than around 600 pixels
# Let's try to optimize the function by breaking the image into overlapping patches, solving each patch separately, and then combining the patches into the final image
# Patches overlap so that we can interpolate the haze values across the overlap region

def solve_haze_detection_v2(i0, i1):
    height, width = i0.shape[:2]
    with Stopwatch(f"solve_haze_detection_v2 {width}x{height}, {width*height} pixels") as st:
        patch_width = 20 # pixels
        patch_height = 20 # pixels
        patch_overlap_x = 10 # pixels
        patch_overlap_y = 10 # pixels

        haze_image = np.zeros((height, width, 4))
        weight_sum = np.zeros((height, width, 1))

        # Calculate minimum number of patches required to completely cover the image, with overlap greater than or equal to patch_overlap_x, patch_overlap_y
        patch_stride_x = patch_width - patch_overlap_x
        patch_stride_y = patch_height - patch_overlap_y
        max_x_inclusive = width - patch_width + 1
        max_y_inclusive = height - patch_height + 1
        num_patches_x = max(1, int(np.ceil(max_x_inclusive / patch_stride_x)))
        num_patches_y = max(1, int(np.ceil(max_y_inclusive / patch_stride_y)))

        # Create a weight matrix for the patch (higher weight in the center, lower at the edges)
        assert patch_width == patch_height
        patch_weight = gaussian_kernel_2d(patch_width, sigma = patch_width / 6.0)

        for y in np.linspace(0, height - patch_height, num_patches_y, dtype=int):
            for x in np.linspace(0, width - patch_width, num_patches_x, dtype=int):
                # Define patch boundaries
                y_end = y + patch_height
                x_end = x + patch_width
                
                # Extract patches
                patch_i0 = i0[y:y_end, x:x_end]
                patch_i1 = i1[y:y_end, x:x_end]
                
                # Solve haze removal for the patch
                patch_haze = solve_haze_detection_v1(patch_i0, patch_i1, verbose=False)
                
                
                # Add the weighted patch to the final image
                haze_image[y:y_end, x:x_end] += patch_haze * patch_weight[:, :, np.newaxis]
                weight_sum[y:y_end, x:x_end] += patch_weight[:, :, np.newaxis]

        # Normalize the final image by the total weights
        haze_image /= weight_sum

        return haze_image

def gaussian_kernel_2d(kernel_size, sigma):
    from scipy.signal.windows import gaussian
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian(kernel_size, std=sigma).reshape(kernel_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

# def create_patch_weight(shape):
#     """Create a weight matrix for a patch, with higher weights in the center."""
#     y, x = np.ogrid[:shape[0], :shape[1]]
#     # np.ogrid creates a grid from 0 to shape[0]-1 and 0 to shape[1]-1 (inclusive)
#     # so we need to subtract 0.5 to get the true center
#     center_y, center_x = (shape[0] - 1) / 2, (shape[1] - 1) / 2
#     np.fabs(x - center_x) / center_x, np.fabs(y - center_y), center_y
#     normalized_distance = np.sqrt(((x - center_x) / center_x) ** 2 + 
#                                   ((y - center_y) / center_y) ** 2)
    
#     weight = 1 - np.minimum(1, )
#     # Usually there's enough overlap that 
#     return weight


def read_image_from_url(url, subsample=1):
    response = requests.get(url)
    im0 = Image.open(io.BytesIO(response.content))
    im0 = im0.resize((im0.width // subsample, im0.height // subsample))
    im0 = np.array(im0) / 255.0
    # Remove top 40 pixels (title and timestamp)
    im0 = im0[40//subsample:]
    return im0

# Create test function
def regression_test_1():
    xvar = 0
    yvar = 1
    equations = [
        [(1, xvar), (1, yvar), 5], # x+y=5
        [(1, xvar), (-1, yvar), 1] # x-y=1
    ]
    expected = [3, 2]
    x = solve_sparse_equations(equations)
    assert np.allclose(x, expected)

def regression_test_2():
    # Synthesize test images
    i0 = np.array([[[1, 0, 0], [0, 1, 0]]])  # Red, Green pixels
    i1 = np.array([[[1, 0.4, 0.4], [0.4, 1, 0.4]]])  # Red, Green pixels with haze

    # Solve for the haze image
    haze_image = solve_haze_detection_v1(i0, i1)

    # Expected haze RGBA values
    expected_haze = np.array([0.4, 0.4, 0.4, 0.4])

    np.testing.assert_array_almost_equal(haze_image[0, 0], expected_haze, decimal=2)
    np.testing.assert_array_almost_equal(haze_image[0, 1], expected_haze, decimal=2)

def regression_test():
    regression_test_1()
    regression_test_2()

    