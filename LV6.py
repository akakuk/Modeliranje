from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import TweedieRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn import svm
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


data = sio.loadmat("bodydata.mat")
bodyData = data["bodydata"]
attributes = ["razmak izmedu ramena", "opseg ramena", "opseg prsa", "opseg struka", "opseg struka oko pupka", "opseg bokova", "opseg bedra", "opseg bicepsa", "opseg podlaktice", "opseg koljena ispod casice", "max. opseg lista", "min. opseg gleznja", "opseg zapesca", "age", "weight", "height", "sex"]

np.random.shuffle(bodyData)
X = bodyData[:,0:13]
Y = bodyData[:,16]
X_train = X[0:round(len(X)*0.8)]
X_test = X[round(len(X)*0.8):len(X)]
Y_train = Y[0:round(len(X)*0.8)]
Y_test = Y[round(len(X)*0.8):len(X)]
bestKNN = 0
bestSVM = 0
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    cvScores = cross_val_score(knn, X, Y, cv=5)
    cvScores = np.mean(cvScores)
    if cvScores > bestKNN:
        bestKNN = cvScores
        bestNNeigh = i
        
for i in range(5, 1000):
    sv = svm.SVC(gamma=i/10000)
    cvScores = cross_val_score(sv, X, Y, cv=5)
    cvScores = np.mean(cvScores)
    if cvScores > bestSVM:
        bestSVM = cvScores
        bestGamma = i/10000


print(bestNNeigh)
print(bestKNN)
print(bestGamma)
print(bestSVM)
f = open("LV6_output.txt", "w")
f.write("Best n Neighbours parametar is : " + str(bestNNeigh) + "\n")
f.write("With cross validation mean : " + str(bestKNN) + "\n")
f.write("Best SVM gamma parametar is : " + str(bestGamma) + "\n")
f.write("With cross validation mean : : " + str(bestSVM) + "\n")




Y = bodyData[:,14]
X_train = X[0:round(len(X)*0.8)]
X_test = X[round(len(X)*0.8):len(X)]
Y_train = Y[0:round(len(X)*0.8)]
Y_test = Y[round(len(X)*0.8):len(X)]




reg = make_pipeline(StandardScaler(), TweedieRegressor(power=1, alpha=0.5, link='log'))
reg.fit(X_train, Y_train)
Y_predicted = reg.predict(X_test)
regMSE = mean_squared_error(Y_test, Y_predicted)

svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2, gamma=0.008))
svr.fit(X_train, Y_train)
Y_predicted = svr.predict(X_test)
svrMSE = mean_squared_error(Y_test, Y_predicted)

print(regMSE)
print(svrMSE)
f.write("General regresion MSE is : " + str(bestGamma) + "\n")
f.write("SVR MSE is : " + str(svrMSE) + "\n")


f.close()


