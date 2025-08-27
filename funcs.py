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
    val  = P - put_price_vollib(S, K,  T, r, sigma, q) #intrinstic solution for region (K1,\infty)
    if backstop1:
        val += 0.9 * put_price_vollib(S, K1, T, r, sigma, q) # (K2, K1]
    if backstop2:
        val += 0.1 * put_price_vollib(S, K2, T, r, sigma, q) # (0,K2]
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
    val = P - np.maximum(K - S, 0.)          # (K1,\infty)
    if backstop1: val += 0.9 * np.maximum(K1 - S, 0.)  # (K2, K1]
    if backstop2: val += 0.1 * np.maximum(K2 - S, 0.)  # (0, K2]
    return val

def plot_value_and_delta(
    K: Number, P: Number, T: Number, r: Number, sigma: Number, q: Number = 0.0,
    backstop1: bool = True, backstop2: bool = True,
    save_prefix: str = None, show: bool = True
):
    """Draw Value and Delta vs S with colored curves and colored K/K1/K2 lines."""
    K1, K2 = compute_K1K2(K, P)
    S_expiry = np.linspace(0.0, 2.0*K, 500)
    S_bsm    = np.linspace(1e-8, 2.0*K, 500)  # avoid S=0 for BSM logs/derivatives 

    # Choose colours
    color_value = "tab:blue"
    color_delta = "tab:orange"
    color_K  = "tab:red"
    color_K1 = "tab:green"
    color_K2 = "tab:purple"

    if T <= 0.0: #intinstric solution 
        x_val, y_val = S_expiry, lrp_value_expiry(S_expiry, K, P, backstop1, backstop2)
    else: # use BSM
        x_val, y_val = S_bsm, lrp_value(S_bsm, K, P, T, r, sigma, q, backstop1, backstop2)

    # Plot underwriter val fig
    plt.figure()
    value_line, = plt.plot(x_val, y_val, color=color_value, linewidth=2, label="Underwriter Value")

    # Vertical lines
    line_K  = plt.axvline(K,  color=color_K,  linestyle="--", linewidth=1.5, label="K")
    line_K1 = plt.axvline(K1, color=color_K1, linestyle="--", linewidth=1.5, label="K1")
    line_K2 = plt.axvline(K2, color=color_K2, linestyle="--", linewidth=1.5, label="K2")

    plt.title(f"LRP Underwriter Value (T={T:.4f}, sigma={sigma}, r={r}, q={q})")
    plt.xlabel("Underlying price S")
    plt.ylabel("Underwriter value")

    # Legend 
    handles = [value_line, line_K, line_K1, line_K2]
    labels  = [h.get_label() for h in handles]
    plt.legend(handles, labels, loc="best")
    if save_prefix:
        plt.savefig(f"{save_prefix}_value.png", dpi=180, bbox_inches="tight")

    # Figure for delta
    T_for_delta = 1e-8 if T <= 0.0 else T #cap to avoid log(0)
    d = lrp_delta(S_bsm, K, P, T_for_delta, r, sigma, q, backstop1, backstop2)

    plt.figure()
    delta_line, = plt.plot(S_bsm, d, color=color_delta, linewidth=2, label="Underwriter Delta")
    line_K  = plt.axvline(K,  color=color_K,  linestyle="--", linewidth=1.5, label="K")
    line_K1 = plt.axvline(K1, color=color_K1, linestyle="--", linewidth=1.5, label="K1")
    line_K2 = plt.axvline(K2, color=color_K2, linestyle="--", linewidth=1.5, label="K2")

    plt.title(f"LRP Underwriter Delta (T={T:.4f}, sigma={sigma}, r={r}, q={q})")
    plt.xlabel("Underlying price S")
    plt.ylabel("Delta (dValue/dS)")
    handles = [delta_line, line_K, line_K1, line_K2]
    labels  = [h.get_label() for h in handles]
    plt.legend(handles, labels, loc="best")
    if save_prefix:
        plt.savefig(f"{save_prefix}_delta.png", dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close("all")


# interface with command line 
def parse_args():
    p = argparse.ArgumentParser(description="LRP reinsured value & delta (vollib).")
    p.add_argument("--K", type=float, default=100., help="Coverage price (strike)")
    p.add_argument("--P", type=float, default=10., help="Premium")
    p.add_argument("--T", type=float, default=0.25, help="Fraction of year to expiry (e.g 0 (exp) or 1 (year))")
    p.add_argument("--sigma", type=float, default=1., help="Annualized volatility")
    p.add_argument("--r", type=float, default=0., help="Risk-free")
    p.add_argument("--q", type=float, default=0., help="Continuous yield (set to zero here)")

    # backstops with on/off flags
    g1 = p.add_mutually_exclusive_group()
    g1.add_argument("--backstop1", dest="backstop1", action="store_true", help="Enable Backstop 1 (default)")
    g1.add_argument("--no-backstop1", dest="backstop1", action="store_false", help="Disable Backstop 1")
    p.set_defaults(backstop1=True)

    g2 = p.add_mutually_exclusive_group()
    g2.add_argument("--backstop2", dest="backstop2", action="store_true", help="Enable Backstop 2 (default)")
    g2.add_argument("--no-backstop2", dest="backstop2", action="store_false", help="Disable Backstop 2")
    p.set_defaults(backstop2=True)

    p.add_argument("--save-prefix", type=str, default=None, help="If set, save PNGs with this prefix")
    p.add_argument("--no-show", action="store_true", help="Do not open figure windows")

    # SANITY CHECK print exact expiry value at a specific S 
    p.add_argument("--S-check", type=float, default=None, help="Print expiry value at this S and exit")

    return p.parse_args()
