This tool models the underwriter’s PnL for a Livestock Risk Protection (LRP) policy with USDA reinsurance backstops and produces Value and Delta charts vs. the underlying livestock price S. Pricing and greeks use the Black-Scholes-Merton (BSM) framework via py_vollib.

For strike $K$, and premium $P$, we define the backstrops: 1) $K_1 = K-1.5P$ and $K_2 = K - 5P$. The underwriter's reinsured payoff at expiration is 
   Value as the underwriter= +P - Put(K) + 0.9*Put(K1) [if backstop1 on] + 0.1*Put(K2) [if backstop2 on]. 
This code provide charts 1) S vs value V (underwriter's value) and 2) S vs \delta V/\delta S (delta) which provides the change in value per dollar change in stock price. 

# TO Install
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

# Run (this is an example)
python main.py \
--K 100 \
--P 10 \
--T 0 \
--sigma 1. \
--r 0. \
--q 0.0  \
--save-prefix charts \
--backstop1 \
--backstop2 \
--no-show \
--S-check 10

or simply 
python main.py \
--K 100 \
--P 10 \
--T 0 \
--sigma 1. \
--r 0. \
--q 0.0  \
--save-prefix charts \
--no-show \
--S-check 10

Here in this example: 
--K 100 \ #Coverage/strike price K - float at \$100
--P 10 \ # Premium collected by underwriter - float at \$10
--T 0 \ #Fraction of year to expiry - float (example is at expiry)
--sigma 1. \ # Annualized volatility - float. 
--r 0. \ # Risk-free rate - float (example at 0)
--q 0.0  # Continuous yield/convenience yield - float (example at 0)
--save-prefix mycharts # If set, save PNGs with this prefix - string
--backstop1 \ # --backstop1/--no-backstop1 to toggle USDA backstop 1 on/off (default on)
--backstop2 \ # --backstop2/--no-backstop2 to toggle USDA backstop 2 on/off (default on)
--no-show \ #Don’t open figure windows - flag
--S-check 10 \ # Sanity mode: print exact expiration value at livestock price and exit (example at \$10)

 