import argparse
from funcs import * 

def main():
    args = parse_args()

    K1, K2 = compute_K1K2(args.K, args.P)
    print(f"K1={K1:.4f}, K2={K2:.4f}")

    if args.S_check is not None: #Sanity check only 
        v = lrp_value_expiry(args.S_check, args.K, args.P, args.backstop1, args.backstop2)
        print(f"Expiry value at S={args.S_check}: {float(v):.3f}")
        return

    plot_value_and_delta( #Plot graphs of value and deltas
        K=args.K, P=args.P, T=args.T, r=args.r, sigma=args.sigma, q=args.q,
        backstop1=args.backstop1, backstop2=args.backstop2,
        save_prefix=args.save_prefix, show=not args.no_show
    )

if __name__ == "__main__":
    main()

exit()
