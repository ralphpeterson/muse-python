"""Plausible constants for an audio scenario, used for testing."""
import numpy as np

from muse.util import make_xy_grid, mixing_matrix_from_n_mics, velocity_sound

AIR_TEMP = 20 # in degrees C

# --- AUDIO SAMPLING PARAMS ---
N = 100 # number of samples
SAMPLE_RATE = 1250
dt = 1 / SAMPLE_RATE

N_MICS = 4

# --- SPATIAL PARAMS ---
ROOM_DIM = (3.3, 2.17, 0.53)
X_DIM, Y_DIM, Z_DIM = ROOM_DIM
offset = 0.1

# (3, N_MICS) array storing position of each mic
MIC_POSITIONS = np.array([
    [X_DIM-offset, Y_DIM-offset, Z_DIM-offset],
    [offset, offset, Z_DIM-offset],
    [offset, Y_DIM-offset, Z_DIM-offset],
    [X_DIM-offset, offset, Z_DIM-offset]
]).T

GRID_RESOLUTION = 0.25

# calculate a plausible tau value, copying MUSE code
x_grid, y_grid = make_xy_grid(X_DIM, Y_DIM, resolution=GRID_RESOLUTION)
n_gridpoints = x_grid.shape[0] * x_grid.shape[1]
r_scan = np.zeros((3, 1, n_gridpoints))
r_scan[0] = x_grid.reshape((1, -1))
r_scan[1] = y_grid.reshape((1, -1))

# get the individual coordinate differences between each mic and each grid position
coord_differences = r_scan - MIC_POSITIONS[..., np.newaxis]
# from these, calculate the Euclidean distance
d = np.sqrt(np.sum(coord_differences ** 2, axis=0))
# divide this distance by the velocity of sound to get
# the expected time delay between each gridpoint and each microphone
tau = d / velocity_sound(AIR_TEMP)

M = mixing_matrix_from_n_mics(N_MICS)
tau_diff = np.matmul(M, tau)