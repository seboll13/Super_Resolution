# ############################################################################
#
# File       : measurements.py
# Description: Better visualisation of samples generation for the main algorithm
# Author     : Sébastien Ollquist
# Date       : 2022 Spring Semester
# School     : EPFL - Lausanne
# Course     : (CS-498) Semester Project in Computer Science
# Topic      : Super resolution off the grid
# References : 
#  - https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
#
# ############################################################################

import numpy as np
import matplotlib.pyplot as plt
from time import time

m = 2000     # number of measurements
Delta = 0.05 # min distance between any two points on the plane
R = 1/Delta  # cutoff frequency


def timer(func):
    """Decorator function that calls any subroutine function"""
    def f(*args, **kwargs):
        """Measure execution time of any function f"""
        start = time()
        ret = func(*args, **kwargs)
        end = time()
        print('Execution time of %s(): %.5f seconds' % (func.__name__, end - start))
        return ret
    return f


@timer # Decorator
def generate_gaussian_samples(num_samples):
    """Generate samples from a Gaussian distribution from uniformly distributed samples."""
    samples = []
    for _ in range(num_samples):
        U1, U2 = np.random.uniform(0,1), np.random.uniform(0,1)
        
        # Apply the Box-Muller transformation (only one sample suffices)
        Z1 = np.sqrt(-2*np.log(U1)) * np.cos(2*np.pi*U2)
        #Z2 = np.sqrt(-2*np.log(U1)) * np.sin(2*np.pi*U2)

        # Scale the variances by the cutoff frequency
        sigma = R**2
        samples.append(sigma*Z1)
    return samples


def draw_gaussian_samples(samples):
    plt.hist(samples, bins=100)
    plt.title('Gaussian samples with variance R**2')
    plt.xlabel('Values')
    plt.ylabel('# of samples')


@timer # Decorator
def generate_sample_from_unit_sphere(x=1):
    """Generate a triplet that lies on the surface of the unit sphere."""
    Z1 = np.random.normal(0,1)
    Z2 = np.random.normal(0,1)
    Z3 = np.random.normal(0,1)

    norm = (Z1**2 + Z2**2 + Z3**2) ** 0.5
    X,Y,Z = Z1/norm, Z2/norm, Z3/norm
    assert(round(X**2 + Y**2 + Z**2, 3) == 1.0)
    
    return (X,Y,Z)


def draw_unit_sphere(point):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    # Use spherical coordinate system
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    ax.plot_wireframe(x, y, z)

    # Draw the given point
    ax.scatter([point[0]], [point[1]], [point[2]], color="g", s=100)


if __name__ == '__main__':
    # Gaussian samples
    samples = generate_gaussian_samples(m)
    (V1,V2,V3) = generate_sample_from_unit_sphere()

    # Plotting part
    draw_gaussian_samples(samples)
    draw_unit_sphere((V1,V2,V3))
    plt.show()