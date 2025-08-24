import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

try:
    from py_vollib.black_scholes_merton import black_scholes_merton as bsm_price
    from py_vollib.black_scholes_merton.greeks.analytical import delta as bsm_delta
except Exception:
    bsm_price = None
    bsm_delta = None

Number = Union[float, int]
ArrayLike = Union[Number, np.ndarray, list, tuple]

def compute_K1K2(K: Number, P: Number):
    """USDA rules: K1 = K - 1.5P, K2 = K - 5P."""
    return K - 1.5*P, K - 5.*P

def _ensure_array(S: ArrayLike): #S needs to be a numpy array 
    return np.asarray(S, dtype=float)

def _check_vollib():
    if bsm_price is None or bsm_delta is None:
        raise RuntimeError("py_vollib not found. Install via pip install py_vollib")

def put_price_vollib(S: ArrayLike, K: Number, T: Number, r: Number, sigma: Number, q: Number=0.):
    """
    Vectorized long put price (BSM with vollib). Uses intrinsic value if T<=0.
    """
    S = _ensure_array(S) #S needs to be a numpy array!
    if T <= 0 or sigma == 0.: #Return inrinsic value 
        return np.maximum(K - S, 0.)
    _check_vollib() # Check if vollib has been installed
    f = np.vectorize(lambda s: bsm_price('p', float(max(s, 1e-12)), float(K), float(T),
                                         float(r), float(sigma), float(q)))
    return f(S)

def put_delta_vollib(S: ArrayLike, K: Number, T: Number, r: Number, sigma: Number, q: Number=0.):
    """
    Long put delta (BSM with vollib). At expiry, returns left-limit cadlag slope (-1 for S<K, else 0).
    """
    S = _ensure_array(S) #S needs to be a numpy array!
    if T <= 0. or sigma == 0.: #Return inrinsic derivative
        return np.where(S < K, -1., 0.)
    _check_vollib() # Check if vollib has been installed
    f = np.vectorize(lambda s: bsm_delta('p', float(max(s, 1e-12)), float(K), float(T),
                                         float(r), float(sigma), float(q)))
    return f(S)

def lrp_value(S: ArrayLike, K: Number, P: Number, T: Number, r: Number, sigma: Number, q: Number=0.,
              backstop1: bool=True, backstop2: bool=True):
    """
    Value as the underwriter= +P - Put(K) + 0.9*Put(K1) [if backstop1 on] + 0.1*Put(K2) [if backstop2 on].
    """
    S = _ensure_array(S) #S needs to be a numpy array!
    K1, K2 = compute_K1K2(K, P)
    val  = P - put_price_vollib(S, K,  T, r, sigma, q) #intrinstic solution 
    if backstop1:
        val += 0.9 * put_price_vollib(S, K1, T, r, sigma, q) # (K2, K1)
    if backstop2:
        val += 0.1 * put_price_vollib(S, K2, T, r, sigma, q)
    return val

def lrp_delta(S: ArrayLike, K: Number, P: Number, T: Number, r: Number, sigma: Number, q: Number=0.,
              backstop1: bool=True, backstop2: bool=True):
    """Underwriter delta: -\Delta(K) + 0.9\Delta(K1) [if backstop1 on] + 0.1\Delta(K2) [if backstop2 on]."""
    S = _ensure_array(S) #S needs to be a numpy array!
    K1, K2 = compute_K1K2(K, P)
    d  = - put_delta_vollib(S, K,  T, r, sigma, q) #derivative of intrinsic solution
    if backstop1:
        d += 0.9 * put_delta_vollib(S, K1, T, r, sigma, q)
    if backstop2:
        d += 0.1 * put_delta_vollib(S, K2, T, r, sigma, q)
    return d

def lrp_value_expiry(S: ArrayLike, K: Number, P: Number, backstop1: bool=True, backstop2: bool=True) -> np.ndarray:
    """Exact piecewise-linear value at T=0."""
    S = _ensure_array(S)
    K1, K2 = compute_K1K2(K, P)
    val = P - np.maximum(K - S, 0.)          # region (K1,\infty)
    if backstop1: val += 0.9 * np.maximum(K1 - S, 0.)  # region (K2, K1]
    if backstop2: val += 0.1 * np.maximum(K2 - S, 0.)  # region (0, K2]
    return val

def main():
    args = parse_args()

    K1, K2 = compute_K1K2(args.K, args.P)
    print(f"K1={K1:.4f}, K2={K2:.4f}")

    if args.S_check is not None:
        v = lrp_value_expiry(args.S_check, args.K, args.P, args.backstop1, args.backstop2)
        # when S_check is scalar, v is scalar-like; cast to float for a neat print
        print(f"Expiry value at S={args.S_check}: {float(v):.4f}")
        return

if __name__ == "__main__":
    main()

exit()
