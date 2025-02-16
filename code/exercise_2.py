# MOD550 - Assignment 2
# Dea Lana Asri - 277575

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import timeit as it

# Assignment Point 1: Fix the code

from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse


observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]
karg = {'observed': observed,'predicted': predicted}
factory = {'mse_vanilla' : vanilla_mse,
            'mse_numpy' : numpy_mse,
            'mse_sk' : sk_mse
            }
for talker, worker in factory.items():
    # SK method expects y_true and y_pred as an input, not dictionary
    if talker == 'mse_sk':
        exec_time = it.timeit('{worker(observed, predicted)}', globals=globals(), number=100) / 100
        mse = worker(observed, predicted)
    else:
        exec_time = it.timeit('{worker(**karg)}', globals=globals(), number=100) / 100
        mse = worker(**karg)
    print(f"Mean Squared Error, {talker} :", mse, f"Average execution time: {exec_time} seconds")

print('Task 1: Test successful')


# Assignment Point 2: Function that generate 1D oscillatory data
def generate_oscillatory_data(n_points = 200, amplitude = 2, range = [0, 3], noise = 1, random_seed=1234):
    """ Generate 1D oscillatory data.
    Parameters:
    -----------
    n_points : int, The number of points to generate.
    frequency : float, The frequency of the oscillation.
    range : tuple, The range of the data.
    noise : float, The standard deviation of the noise.
    random_seed : int, The random seed for reproducibility.

    Output:
    -------
    x : numpy.ndarray, The x values.
    y : numpy.ndarray, The y values.
    y_noise : numpy.ndarray, The y values with noise.
    """
    # Generate the data
    x = np.linspace(range[0], range[1], n_points)
    y = amplitude * np.sin(np.pi * x)
    
    # Adding noise
    noise_distribution = np.random.normal(0, noise, n_points)
    y_noise = y + noise_distribution
    return x, y, y_noise

# Generate the data
x, y, y_noise = generate_oscillatory_data()

# combining the data
x_combined = np.concatenate((x, x))
y_combined = np.concatenate((y, y_noise))

print(f"Task 2: Data generated {len(y)+len(y_noise)} points, range from {x[0]} to {x[-1]}, truth function: y = amplitude * sin(pi * x), truth function min: {min(y):.2f}, truth function max: {max(y):.2f}, noisy data min: {min(y_noise):.2f}, noisy data max: {max(y_noise):.2f}")

# Assignment Point 3: Clustering the data

from sklearn.cluster import KMeans

# Number of clusters
n_clusters = range(1, 11)
variance = []

# Lopping to cluster the data and print the information
print(f"Task 3: Clustering methond: KMeans clustering")
for i in n_clusters:
    kmeans = KMeans(n_clusters=i, random_state=1234).fit(y_combined.reshape(-1, 1))
    variance.append(kmeans.inertia_)
    print(f"\nNumber of clusters: {i}, Variance: {kmeans.inertia_}")

# Plotting the data


