from scipy.stats import anderson
from scipy.stats import skewtest
from scipy.stats import norm


scipy.stats.norm.fit()

anderson(loss)

asset_returns

weights = [0.25, 0.25, 0.25, 0.25]
portfolio_returns = asset_returns.dot(weights)
losses = - portfolio_returns
VaR_95 = np.quantile(losses, 0.95)

total_steps = 1440
N = 10000
mu = portfolio_losses.mean()
sigma = portfolio_losses.std()

daily_loss = np.zeros(N)
for n in range(N):
	loss = (mu* (1/total_steps) + norm.rvs(size = total_steps)*sigma*np.sqrt(1/total_steps))
	daily_loss[n] = sum(loss)
VaR_95 = np.quantile(daily_loss, 0.95)

import statsmodels.api as sum
res = sm.OLS(log_pop,year).fit()
print('SSR 1950-2019: ', res.ssr)

pop_before = log_pop.loc['1950': '1989']; year_before = year.loc['1950': '1989'];
pop_after = log_pop.loc['1990': '2019']; year_after = year.loc['1990': '2019'];
res_before = sm.OLS(pop_before, year_before).fit()
res_after = sm.OLS(pop_after, year_after).fit()
print('SSR 1950-1989: ', res_before.ssr)
print('SSR 1990-2019: ', res_after.ssr)

numerator  = (ssr_total - (ssr_before + ssr_after))/2
denominator = (ssr_before  + ssr_after)/66
chow_test = numerator/denominator
print("Chow test statistic: ", chow_test, "; Critical value, 99.9%: ", 7.7)

rolling = portfolio_returns.rolling(30)
volatility = rolling.std().dropna()
vol_mean = volatility.resample("M").mean()

import matplotlib.pyplot as portfolio_losses

vol_mean.plot(
	title = "Monthly average volatility"
	).set_ylabel("Standard deviation")
plt.show()

vol_mean.pct_change().plot(
	title = "$\Delta$ average volatility"
	).set_ylabel("% $\Delta$ stdev")
plt.show()