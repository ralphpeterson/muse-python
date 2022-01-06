import numpy as np

def velocity_sound(T):
    """
    From https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/velocity_sound.m
    
    This function calculates the velocity of sound based on the temp during recording.
    
    Formula for speed of sound in air taken from signals, sound and sensation (hartmann).

    T is the temp in Celsius, Vsound is the speed of sound in m/s.
    """
    Vsound = 331.3*np.sqrt(1+(T/273.16))
    return Vsound