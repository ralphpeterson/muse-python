"""Utility functions to help set up pyroomacoustics with MUSE."""
from collections.abc import Sequence

import librosa
import numpy as np
import pyroomacoustics as pra

from pyroomacoustics.directivities import CardioidFamily, DirectionVector, DirectivityPattern

def generate_room(
    x_dim: float,
    y_dim: float,
    z_dim: float,
    sampling_rate=125000
) -> pra.Room:
    """
    Return a shoebox room of the provided dimensions using pyroomacoustics.

    This Shoebox is the basis for the example. After generating it, we'll place mics in it
    and play a sound stimulus. Pyroomacoustics will them simulate the signals that the
    mic would recieve based on their locations and the location of the sound stimulus.
    """
    # Materials of the room's walls
    # See different materials at https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.materials.database.html
    materials = pra.make_materials(
        ceiling = "fibre_absorber_2",
        floor   = "fibre_absorber_2",
        east    = "hard_surface",
        west    = "hard_surface",
        north   = "hard_surface",
        south   = "hard_surface"
    )
    
    shoebox_object = pra.ShoeBox(
        [x_dim, y_dim, z_dim],
        fs=sampling_rate,
        materials=materials,
        max_order=9  # Maximum number of times a sound wave can reflect
    ) 
    return shoebox_object

def place_mics(shoebox: pra.Room, mic_pos: np.ndarray):
    """
    Place mics at the specified positions in the given Shoebox, oriented towards
    the center of the room's floor.
    
    Args:
        shoebox: The room into which the mics will be placed.
        mic_pos: array of shape (n_mics, 3) storing the Cartesian coordinates
            at which to place the mics.
    """
    shoebox_dims = shoebox.shoebox_dim
    if any([coord >= box_size for coord, box_size in zip(mic_pos.max(axis=0), shoebox_dims)]) \
        or any([coord <= 0 for coord in mic_pos.min(axis=0)]):
        raise ValueError("The microphones must be within the box. They cannot be located along an egde or face.")
    
    # Compute directivity of microphones
    # Assumes the microphones are all pointed toward the center of the room's floor
    # Can be simplified by just pointing them downward
    room_center = np.array(shoebox_dims) / 2
    room_center[2] = 0
    
    mic_directions = room_center[np.newaxis, ...] - mic_pos  # List of vectors pointing from each mic to room_center
    mic_directions /= np.sqrt((mic_directions**2).sum(axis=1, keepdims=True))  # Normalize the vectors
    
    # Convert the direction vectors into polar coordinates for the directivities used by PRA
    directivities = list()
    for direction in mic_directions:
        # Angle within the x-y plane. 0 radians is toward the positive x direction
        # In spherical coordinates, this is \theta
        azimuth = np.arctan2(direction[1], direction[0])
        # Same thing here, but between the z axis and the x-y plane
        xy_vec_size = np.sqrt((direction[:2] ** 2).sum())
        # Using arctan instead of arctan2 because I need the result to be within Q1 and Q4
        # This would be called \phi in spherical coordinates
        colatitude = np.pi/2 - np.arctan(direction[2] / xy_vec_size)
        pra_direction = DirectionVector(
            azimuth=azimuth,
            colatitude=colatitude,
            degrees=False
        )
        # For visualizations, see https://en.wikipedia.org/wiki/Microphone#Polar_patterns
        # For supported options, see https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.directivities.html?highlight=DirectivityPattern#pyroomacoustics.directivities.DirectivityPattern
        directivities.append(CardioidFamily(
            orientation=pra_direction,
            pattern_enum=DirectivityPattern.HYPERCARDIOID
        ))
    
    # transpose because shoebox expects positions in shape (3, n_mics)
    shoebox.add_microphone_array(mic_array=mic_pos.T, directivity=directivities)
    
def add_source(
    speaker_pos: Sequence,
    stimulus: np.ndarray,
    sr_stimulus: int,
    room: pra.Room
    ):
    """
    Add an audio source playing the given stimulus at location speaker_pos.
    
    If the sample rate of the stimulus doesn't match the sampling rate used in the
    room for simulation, resample audio to match.
    """
    if not isinstance(speaker_pos, list):
        speaker_pos = list(speaker_pos)
    
    if room.fs != sr_stimulus:
        print('resampling audio')
        stimulus_resampled = librosa.resample(stimulus.astype('float'), orig_sr=sr_stimulus, 
                                                                      target_sr=room.fs)
        room.add_source(speaker_pos, signal=stimulus_resampled, delay=0)
        return room.fs, stimulus_resampled
        
    else:
        room.add_source(speaker_pos, signal=stimulus, delay=0)
        return sr_stimulus, stimulus