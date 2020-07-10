import scipy.signal as sig
import scipy.stats as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg #ARX
from statsmodels.tsa.arima_model import ARMA #ARMAX
from statsmodels.tsa.arima_model import ARIMA #drugi naziv za box-jenkins

data = pd.read_csv("preparedData.csv", sep = ";", header = None, names = ["t", "x", "y"])
t = data["t"].astype(float)
x = data["x"].astype(float)
y = data["y"].astype(float)

# # contrived dataset
# # fit model
# model = AutoReg(data, lags=1)
# model_fit = model.fit()
# # make prediction
# yhat = model_fit.predict(len(data), len(data))

# # fit model
# model = ARMA(data, order=(2, 1))
# model_fit = model.fit(disp=False)
# # make prediction
# yhat = model_fit.predict(len(data), len(data))
# print(yhat)

# model = ARIMA(data, order=(1, 1, 1))
# model_fit = model.fit(disp=False)
# # make prediction
# yhat = model_fit.predict(len(data), len(data), typ='levels')

