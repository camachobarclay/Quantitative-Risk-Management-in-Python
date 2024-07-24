from scipy.stats import genextreme
from scipy.stats import gaussian_kde

maxima = losses.resample("W").max()
params = genextreme.fit(maxima)

VaR_99 = genextreme(0.99, *params)
CVar_99 = (1/(1-0.99))*genextreme.expect(lambda x: x, *params, lb = VaR_99)

kde = gaussian_kde(losses)
loss_range = np.linspace(np.min(losses), np.max(losses), 1000)
plt.plot(loss_range, kde_pdf(loss_range))

sample = kde.resample(size = 1000)
VaR_99 = np.quantile(sample, 0.99)
print("VaR_99 from KDE: ", VaR_99)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model Sequential()
model.add(Dense(10, input_dim = 4, activation = 'sigmoid'))
model.add(Dense(4))

model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop')
model.fit(training_input, training_output, epochs = 100)

# new asset prices are in the vector new_Asset_prices
predicted = model.predict(new_aset_prices)