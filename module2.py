import pandas as pd
import numpy as np

loss = pd.Series(observations)
VaR_95 = loss.quantile(0.95)
print("VaR_95 =", VaR_95)


loss = pd.Series(scipy.stats.norm.rvs(size = 1000))
VaR_95 = scipy.stats.norm.ppf(0.95)
CVaR_95 =  (1/(1-0.95))*scipy.stats.norm.expect(lambda x: x, lb = VaR_95)
print("CVaR_95 = ", CVaR_95)

# Create the VaR measure at the 95% confidence level using norm.ppf()
VaR_95 = norm.ppf(0.95)

# Create the VaR measure at the 99% confidence level using numpy.quantile()
draws = norm.rvs(size = 100000)
VaR_99 = np.quantile(draws, 0.99)

# Compare the 95% and 99% VaR
print("95% VaR: ",VaR_95, "; 99% VaR: ", VaR_99)

# Plot the normal distribution histogram and 95% VaR measure
plt.hist(draws, bins = 100)
plt.axvline(x = VaR_95, c='r', label = "VaR at 95% Confidence Level")
plt.legend(); plt.show()

from scipy.stats import stats

params = t.fit(portfolio_losses)
VaR_95 = t.ppf(0.95, *params )
# Import the Student's t-distribution
from scipy.stats import t


# Create rolling window parameter list
mu = losses.rolling(30).mean()
sigma = losses.rolling(30).std()
rolling_parameters = [(29, mu[i], s) for i,s in enumerate(sigma)]

# Compute the 99% VaR array using the rolling window parameters
VaR_99 = np.array( [ t.ppf(losses, *params) 
                    for params in rolling_parameters ] )

# Plot the minimum risk exposure over the 2005-2010 time period
plt.plot(losses.index, 0.01 * VaR_99 * 100000)
plt.show()


ec = pyfopt.efficient_frontier.EfficientCVaR(None, returns)
optimal_weights = ec.min_cvar()

ef = efficientFrontier(None, e_cov)
min_vol_weights = ef.min_volatility()
print(min_vol_weights)

ec = pypfopt.efficient_frontier.EfficienctCVaR(None, returns)
min_cvar_weights = ec.min_cvar()
print(min_cvar_weights)

S = 70; X = 80; T = 0.5; r = 0.2; sigma = 0.2
option_value = black_scholes(S,X,T,r,sigma, option_type = "put")
print(option_value)