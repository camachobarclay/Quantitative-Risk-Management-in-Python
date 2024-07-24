import pandas as pd
import numpy as np


prices =  pd.read_csv("portfolio.csv")
returns = prices.pct_change()
weights = (weight_1, weight_2)
portfolio_returns = returns.dot(weights)

covariance = returns.cov()*252
print(covariance)

weights = 0.25*np.ones(4)
portfolio_variance = np.transpose(weights)@covariance@weights
portfolio_volatility = np.sqrt(portfolio_variance)
windowed = portfolio_returns.rolling(30)
volatility = windowed.std()*np.sqrt(252)
volatility.plot().set_ylabel("standard Deviation...")

#-------------

# Select portfolio asset prices for the middle of the crisis, 2008-2009
asset_prices = portfolio.loc['2008-01-01':'2009-12-31']

# Plot portfolio's asset prices during this time
asset_prices.plot.set_ylabel("Closing Prices, USD")
plt.show()

# Compute the portfolio's daily returns
asset_returns = asset_prices.pct_change()
portfolio_returns = asset_returns.dot(weights)

# Plot portfolio returns
portfolio_returns.plot().set_ylabel("Daily Return, %")
plt.show()

# Calculate the 30-day rolling window of portfolio returns
returns_windowed = portfolio_returns.rolling(30)

# Compute the annualized volatility series
volatility_series = returns_windowed.std()*np.sqrt(252)

# Plot the portfolio volatility
volatility_series.plot().set_ylabel("Annualized Volatility, 30-day Window")
plt.show()

########################

import statsmodel.api as sm

regression = sm.OLS(returns, delinquencies).fit()
print(regression.summary())

#--------------

# Convert daily returns to quarterly average returns
returns_q = returns.resample('Q').mean()

# Examine the beginning of the quarterly series
print(returns_q.head())

# Now convert daily returns to weekly minimum returns
returns_w = returns.resample('W').min()

# Examine the beginning of the weekly series
print(returns_w.head())

# Transform the daily portfolio_returns into quarterly average returns
portfolio_q_average = portfolio_returns.resample('Q').mean().dropna()

# Create a scatterplot between delinquency and quarterly average returns
plot_average.scatter(mort_del, portfolio_q_average)

# Transform daily portfolio_returns returns into quarterly minimum returns
portfolio_q_min = portfolio_returns.resample('Q').min().dropna()

# Create a scatterplot between delinquency and quarterly minimum returns
plot_min.scatter(mort_del, portfolio_q_min)
plt.show()

# Add a constant to the regression
mort_del = sm.add_constant(mort_del)

# Create the regression factor model and fit it to the data
results = sm.OLS(vol_q_mean,mort_del).fit()

# Print a summary of the results
print(results.summary())


###################

import PyPortfolioOpt as pypfopt
#EfficientFrontier
#CLA

expected_returns = mean_historical_return(prices)
efficient_cov = CovarianceShrinkage(prices).ledoit_wolf()
cla = CLA(expected_returns, efficient_cov)
minimum_variance = cla.min_volatility()
(ret, vol, weights) = cla.efficient_frontier()

#----------------

# Load the investment portfolio price data into the price variable.
prices = pd.read_csv("portfolio.csv")

# Convert the 'Date' column to a datetime index
prices['Date'] = pd.to_datetime(prices['Date'], format='%d/%m/%Y')
prices.set_index(['Date'], inplace = True)

# Import the mean_historical_return method
from pypfopt.expected_returns import mean_historical_return

# Compute the annualized average historical return
mean_returns = mean_historical_return(prices, frequency = 252)

# Plot the annualized average historical return
plt.plot(mean_returns, linestyle = 'None', marker = 'o')
plt.show()