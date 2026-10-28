"""ctypes wrapper of the C eyes (libeyes.so)"""
import ctypes, os
import numpy as np
LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libeyes.so"))
P = ctypes.c_void_p


def gray(rgb):
    rgb = np.ascontiguousarray(rgb, np.uint8); out = np.zeros(rgb.shape[0] * rgb.shape[1])
    LIB.eyes_gray(rgb.ctypes.data_as(P), ctypes.c_int(rgb.shape[0] * rgb.shape[1]), out.ctypes.data_as(P))
    return out.reshape(rgb.shape[:2])


def level(g, s, GE, DE, QSIG):
    g = np.ascontiguousarray(g, np.float64); H, W = g.shape
    cap = int((H / s + 2 * 8 + 4) * (W / s + 2 * 8 + 4))
    cd = np.zeros(cap, np.int8); cg = np.zeros(cap, np.int8); Hp, Wp = ctypes.c_int(), ctypes.c_int(); noise = np.zeros(2)
    LIB.eyes_level(g.ctypes.data_as(P), ctypes.c_int(H), ctypes.c_int(W), ctypes.c_double(s), ctypes.c_double(GE), ctypes.c_double(DE),
                   ctypes.c_double(QSIG), cd.ctypes.data_as(P), cg.ctypes.data_as(P), ctypes.byref(Hp), ctypes.byref(Wp), noise.ctypes.data_as(P))
    n = Hp.value * Wp.value
    return cd[:n].reshape(Hp.value, Wp.value), cg[:n].reshape(Hp.value, Wp.value), noise
