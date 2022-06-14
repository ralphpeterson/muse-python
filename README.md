# muse-python
Python implementation of [Mouse Ultrasonic Source Estimator](https://github.com/JaneliaSciComp/Muse),
a MATLAB tool developed to identify the location of a sound from microphone array data in the paper [(Neunebel et al., 2015)](https://doi.org/10.7554/eLife.06203).

The main functions of this library are `r_est_naive` and `r_est_jackknife`. The function `r_est_naive` calculates the Reduced Steered Response Power (RSRP) at a grid
of points in an arena, returning both the location of the maximum RSRP value as well as the grid itself.

`r_est_jackknife` builds on this by calculating *multiple* location estimates as in [(Warren, 2018)](https://pubmed.ncbi.nlm.nih.gov/29309793).
For each vocal signal, `r_est_jackknife` systematically excludes one microphone, calculating a point estimate using only the data from the other microphones.
This process is repeated for each mic, leading to a group of location estimates that are then averaged.

For a detailed example, see `examples/pyroomacoustics_muse_usage.ipynb`,
where we use the library `pyroomacoustics` to simulate the microphone data that would be
recieved from a given audio source. 

For a quickstart, see the following simple example with randomly generated 'microphone input'.
```python
import numpy as np
from muse import r_est_jackknife

# generate some fake audio data
n_mics = 4
N = 10000  # number of samples
fs = 100000  # sample rate, in Hz
v = np.random.random(n_mics * N).reshape((n_mics, N))

temp = 20  # temperature of the room in Celsius
f_lo = 0  # minimum frequency considered, in Hz
f_hi = 62500  # max frequency considered, in Hz

# spatial parameters
x_len, y_len = (1, 2) # arbitrary dimensions of the room
res = 0.025  # desired resolution of grid at which RSRP values are calculated
# array representing locations of each mic in Cartesian coords
mic_pos = np.array([
    [0.2, 0.2, 1],  # units in meters
    [0.2, 0.8, 1],
    [1.8, 0.2, 1],
    [1.8, 0.8, 1]
])

avg_est, r_ests, rsrp_grids = r_est_jackknife(v, fs, f_lo, f_hi, temp, x_len, y_len, res, mic_pos)

print(f'--- Averaged Estimate --- \n{avg_est}')
print('--- Point Estimates --- ')
for i, est in enumerate(r_ests):
    print(f'Mic {i} removed:\n {est}')
```
Note that `avg_est` is the averaged result, `r_ests` is an array containing each point estimate, and `rsrp_grids` stores the full grid of RSRP values associated with each point estimate.

Out:
```
--- Averaged Estimate --- 
[[0.87431943]
 [1.57473846]]
--- Point Estimates --- 
Mic 0 removed:
 [[0.64102564]
 [0.2278481 ]]
Mic 1 removed:
 [[0.12820513]
 [0.4556962 ]]
Mic 2 removed:
 [[0.35897436]
 [0.83544304]]
Mic 3 removed:
 [[0.69230769]
 [0.58227848]]
```
