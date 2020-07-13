import scipy.signal as sig
import scipy.stats as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg #ARX
from statsmodels.tsa.arima_model import ARMA #ARMAX
from statsmodels.tsa.arima_model import ARIMA #drugi naziv za box-jenkins
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("preparedData.csv", sep = ";", header = None, names = ["t", "x", "y"])
t = data["t"].astype(float)
x = data["x"].astype(float)
y = data["y"].astype(float)
x = x.tolist()
y = y.tolist()
t = t.tolist()

scaler = StandardScaler()
scaler.fit([x,y,t])
[x,y,t] = scaler.transform([x,y,t])

yTrain = y[0:round(0.8*len(y))]
xTrain = x[0:round(0.8*len(x))]
tTrain = t[0:round(0.8*len(t))]
yTest = y[round(0.8*len(y)):len(y)]
xTest = x[round(0.8*len(x)):len(x)]
tTest = t[round(0.8*len(t)):len(t)]

trainData = np.column_stack([np.array(xTrain), np.array(yTrain)])
bestARX = 1000
bestARMAX = 1000
bestARIMAX = 1000

for na in range(1, 6):
    modelARX = AutoReg(yTrain, exog=xTrain, lags=na)
    modelFitARX = modelARX.fit()
    yPredictedARX = modelFitARX.predict(len(yTrain), len(yTrain) + len(xTest) - 1, exog=xTrain, exog_oos=xTest)
    yError = np.sum(np.abs(yPredictedARX - yTest))/len(yPredictedARX)
    if yError < bestARX:
        bestARX = yError
        bestOrderARX = [na]
        bestyPredictedARX = yPredictedARX
    for nb in range(1, 6):
        try:
            modelARMAX = ARMA(yTrain, exog=xTrain, order=(na, nb))
            modelFitARMAX = modelARMAX.fit(disp=False)
            yPredictedARMAX = modelFitARX.predict(len(yTrain), len(yTrain) + len(xTest) - 1, exog=xTrain, exog_oos=xTest)
            yError = np.sum(np.abs(yPredictedARMAX - yTest))/len(yPredictedARMAX)
            if yError < bestARMAX:
                bestARMAX = yError
                bestOrderARMAX = [na ,nb]
                bestyPredictedARMAX = yPredictedARMAX
        except:
            print("Error ARMAX at:")
            print([na, nb ,nc])
                                      
        for nc in range(1, 6):
            try:
                modelARIMAX = ARIMA(yTrain, exog=xTrain, order=(na, nb, nc))
                modelFitARIMAX = modelARIMAX.fit(disp=True)
                yPredictedARIMAX = modelFitARIMAX.predict(len(yTrain), len(yTrain) + len(xTest) - 1, exog=xTest, typ='levels')
                yError = np.sum(np.abs(yPredictedARIMAX - yTest))/len(yPredictedARIMAX)
                if yError < bestARIMAX:
                    bestARIMAX = yError
                    bestOrderARIMAX = [na ,nb, nc]
                    bestyPredictedARIMAX = yPredictedARIMAX
            except:
                print("Error ARIMAX at:")
                print([na,nb,nc])

f = open("LV2_output.txt", "w")
f.write("Best ARX order is:")
f.write("Best ARX order is:")
f.write(str(bestOrderARX))
f.write("with mean absolute error of: " + str(bestARX))
f.write("\n")
f.write("Best ARMAX order is:")
f.write(str(bestOrderARMAX))
f.write("with mean absolute error of: " + str(bestARMAX))
f.write("\n")
f.write("Best ARIMAX order is:")
f.write(str(bestOrderARIMAX))
f.write("with mean absolute error of: " + str(bestARIMAX))
f.write("\n")

print("Best ARX order is:")
print(bestOrderARX)
print("\n")
print("Best ARMAX order is:")
print(bestOrderARMAX)
print("\n")
print("Best ARIMAX order is:")
print(bestOrderARIMAX)
print("\n")

plt.figure(num = 0, figsize = (6.4 * 15, 4.8 * 5))
plt.title(label = "Best ARX model", fontsize = 60)
plt.plot(tTest, bestyPredictedARX)
plt.savefig("LV2_BestPredictedARX.png")

plt.figure(num = 1, figsize = (6.4 * 15, 4.8 * 5))
plt.title(label = "Best ARMAX model", fontsize = 60)
plt.plot(tTest, bestyPredictedARMAX)
plt.savefig("LV2_BestPredictedARMAX.png")

plt.figure(num = 2, figsize = (6.4 * 15, 4.8 * 5))
plt.title(label = "Best ARIMAX model", fontsize = 60)
plt.plot(tTest, bestyPredictedARIMAX)
plt.savefig("LV2_BestPredictedARIMAX.png")

bestString = ["ARX", "ARMAX", "ARIMAX"]
bestIndex = [bestARX, bestARMAX, bestARIMAX].index(min([bestARX, bestARMAX, bestARIMAX]))

print("Best model is " + bestString[bestIndex])
f.write("Best model is " + str(bestString[bestIndex]))

f.close()
