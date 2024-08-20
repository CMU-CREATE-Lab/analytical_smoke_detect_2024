import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import lsq_linear
import requests
from PIL import Image
import io
from contextlib import contextmanager
import time

# Assuming Stopwatch is a custom context manager, let's define it
import time
import psutil

@contextmanager
def Stopwatch(name):
    start_time = time.time()
    start_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
    start_cpu_count = psutil.cpu_count()
    
    yield
    end_time = time.time()
    end_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
    
    wall_time = end_time - start_time
    cpu_time = end_cpu_time - start_cpu_time
    avg_cpu_used = cpu_time / wall_time if wall_time > 0 else 0
    
    print(f"{name}: Wall time: {wall_time:.2f}s, CPU time: {cpu_time:.2f}s, Avg CPUs: {avg_cpu_used:.2f}/{start_cpu_count}")

def solve_sparse_equations(equations, bounds=(-np.inf, np.inf)):
    rowidx = []
    colidx = []
    data = []
    b = []
    with Stopwatch("Building sparse matrix"):
        for i, eq in enumerate(equations):
            for coef, var in eq[:-1]:
                rowidx.append(i)
                colidx.append(var)
                data.append(coef)
            b.append(eq[-1])
        nvars = max(colidx) + 1
        A = csr_matrix((data, (rowidx, colidx)), shape=(len(equations), nvars))
        b = np.array(b)
        print(f"Built sparse matrix with {A.shape[0]} equations and {A.shape[1]} variables")
    with Stopwatch(f"Solving sparse matrix"):
        res = lsq_linear(A, b, bounds, verbose=1)
    return res.x

def solve_haze_removal_v1(i0, i1):
    height, width = i0.shape[:2]

    # Number of variables: each pixel has 4 variables (o, r, g, b)
    num_vars = height * width * 4

    # Helper functions to get variable indices
    def h(i, j, k):
        return (i * width + j) * 4 + k
    
    def o(i, j):
        return h(i, j, 3)


    equations = []
    with Stopwatch("Building equations"):
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
    # Count how many elements are less than 0 or greater than 1
    print("Number of elements less than 0 or greater than 1:", np.sum((x_opt < 0) | (x_opt > 1)))

    # Extract the optimized opacity and haze color

    haze_image = np.zeros((height, width, 4))

    for i in range(height):
        for j in range(width):
            haze_image[i, j, 3] = 1 - x_opt[o(i, j)]  # opacity
            for k in range(3):
                haze_image[i, j, k] = x_opt[h(i, j, k)]  # r, g, b

    return haze_image

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
    haze_image = solve_haze_removal_v1(i0, i1)

    # Expected haze RGBA values
    expected_haze = np.array([0.4, 0.4, 0.4, 0.4])

    np.testing.assert_array_almost_equal(haze_image[0, 0], expected_haze, decimal=2)
    np.testing.assert_array_almost_equal(haze_image[0, 1], expected_haze, decimal=2)

def regression_test():
    regression_test_1()
    regression_test_2()
