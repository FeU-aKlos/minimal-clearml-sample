import codecs
import numpy as np

def normalize(x, axis):
    eps = np.finfo(float).eps
    mean = np.mean(x, axis=axis, keepdims=True)
    # avoid division by zero
    std = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mean) / std

def get_int(b):  # CONVERTS 4 BYTES TO A INT
    return int(codecs.encode(b, "hex"), 16)

def some_function(text:str)->str:
    return text