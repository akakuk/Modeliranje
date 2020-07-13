import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score


data = sio.loadmat("bodydata.mat")
bodyData = data["bodydata"]
attributes = ["razmak izmedu ramena", "opseg ramena", "opseg prsa", "opseg struka", "opseg struka oko pupka", "opseg bokova", "opseg bedra", "opseg bicepsa", "opseg podlaktice", "opseg koljena ispod casice", "max. opseg lista", "min. opseg gleznja", "opseg zapesca", "age", "weight", "height"]

scaler = StandardScaler()
PCA1Data = np.column_stack([bodyData[:,0], bodyData[:,7]])
PCA1Data = np.column_stack([PCA1Data, bodyData[:,15]])

scaler.fit(PCA1Data)
PCA1 = scaler.transform(PCA1Data)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(PCA1)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)
print(pca.explained_variance_)

fig = plt.figure(num=0)
ax = fig.gca(projection='3d')
fig.suptitle("PCA1 original: razmak izmedu ramena, opseg bicepsa, visina")
vec1 = pca.explained_variance_[0] * pca.components_[0,:] * 3
vec2 = pca.explained_variance_[1] * pca.components_[1,:] * 3
ax.plot([0,vec1[0]], [0,vec1[1]], zs=[0,vec1[2]], color="black" ,)
ax.plot([0,vec2[0]], [0,vec2[1]], zs=[0,vec2[2]], color="black" ,)
fig.savefig("LV5_PCA1AxesOrg.png")
ax.scatter(PCA1[0:247,0], PCA1[0:247,1], PCA1[0:247,2], color="blue")
ax.scatter(PCA1[247:508,0], PCA1[247:508,1], PCA1[247:508,2], color="red")
#ax.view_init(10, 45) #provjerava da pod kut sve štima


fig.savefig("LV5_PCA1AxesWithDataOrg.png")

plt.figure(num=1)
plt.title("PCA1 projicirani: razmak izmedu ramena, opseg bicepsa, visina")
plt.scatter(principalComponents[0:247,0], principalComponents[0:247,1], color="blue", label="M")
plt.scatter(principalComponents[247:508,0], principalComponents[247:508,1], color="red", label="F")
plt.plot([0,0], [0,1], color="black")
plt.plot([0,1], [0,0], color="black")
plt.savefig("LV5_PCA1Projected.png")

pca = PCA()
principalComponents = pca.fit_transform(PCA1)
n = []
for x in range(0,pca.explained_variance_ratio_.size):
    n.append(x)
weights = pca.explained_variance_ratio_ / pca.explained_variance_ratio_.sum()
cumsum = weights.cumsum()
fig, ax1 = plt.subplots()
ax1.bar(n ,pca.explained_variance_ratio_)
ax2 = ax1.twinx()
ax2.plot(n, cumsum, '-ro', alpha=0.5)
ax2.set_ylabel('', color='r')
ax2.tick_params('y', colors='r')
vals = ax2.get_yticks()
ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
fig.savefig("LV5_PCA1Parets")



PCA2Data = np.column_stack([bodyData[:,0], bodyData[:,1]])
for i in range(2,13):
    PCA2Data = np.column_stack([PCA2Data, bodyData[:,i]])
    
scaler = StandardScaler()
scaler.fit(PCA2Data)
PCA2 = scaler.transform(PCA2Data)
pca = PCA()
principalComponents = pca.fit_transform(PCA2)
print("PCA2")
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)
print(pca.explained_variance_)
plt.figure(num=2)
n = []
for x in range(0,pca.explained_variance_ratio_.size):
    n.append(x)
weights = pca.explained_variance_ratio_ / pca.explained_variance_ratio_.sum()
cumsum = weights.cumsum()
fig, ax1 = plt.subplots()
ax1.bar(n ,pca.explained_variance_ratio_)
ax2 = ax1.twinx()
ax2.plot(n, cumsum, '-ro', alpha=0.5)
ax2.set_ylabel('', color='r')
ax2.tick_params('y', colors='r')
vals = ax2.get_yticks()
ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
fig.savefig("LV5_PCA2Parets")

Kmeans1Data = np.column_stack([bodyData[:,5], bodyData[:,14]])

silhouette_avg = []
for x in [2, 3 , 4 , 5 , 6]:
    kmeans = KMeans(n_clusters=x, random_state=0).fit(Kmeans1Data)
    labels = kmeans.predict(Kmeans1Data)
    silhouette_avg.append(silhouette_score(Kmeans1Data, labels))
    
nClusters = silhouette_avg.index(max(silhouette_avg)) + 2

kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(Kmeans1Data)
labels = kmeans.predict(Kmeans1Data)
plt.figure(num=5)
plt.title("Kmeans: opseg bokova i težina")
plt.scatter(Kmeans1Data[labels[:] == 1,0], Kmeans1Data[labels[:] == 1,1], color="blue", label="M", alpha=0.5)
plt.scatter(Kmeans1Data[labels[:] == 0,0], Kmeans1Data[labels[:] == 0,1], color="red", label="F", alpha=0.5)
# plt.scatter(Kmeans1Data[labels[:] == 2,0], Kmeans1Data[labels[:] == 2,1], color="yellow", label="F", alpha=0.5)
plt.savefig("LV5_KMeans1.png")
# kmeans.labels_
# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_

gmm = GaussianMixture(n_components=nClusters).fit(Kmeans1Data)
labels = gmm.predict(Kmeans1Data)

plt.figure(num=6)
plt.title("Gauss: opseg bokova i težina")
plt.scatter(Kmeans1Data[labels[:] == 1,0], Kmeans1Data[labels[:] == 1,1], color="blue", label="M", alpha=0.5)
plt.scatter(Kmeans1Data[labels[:] == 0,0], Kmeans1Data[labels[:] == 0,1], color="red", label="F", alpha=0.5)
# plt.scatter(Kmeans1Data[labels[:] == 2,0], Kmeans1Data[labels[:] == 2,1], color="yellow", label="F", alpha=0.5)
plt.savefig("LV5_Gauss1.png")

Kmeans2Data = np.column_stack([bodyData[:,2], bodyData[:,3]])
Kmeans2Data = np.column_stack([Kmeans2Data, bodyData[:,14]])

silhouette_avg = []

for x in [2, 3 , 4 , 5 , 6]:
    kmeans = KMeans(n_clusters=x, random_state=0).fit(Kmeans2Data)
    labels = kmeans.predict(Kmeans2Data)
    silhouette_avg.append(silhouette_score(Kmeans2Data, labels))
    
nClusters = silhouette_avg.index(max(silhouette_avg)) + 2

kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(Kmeans2Data)
labels = kmeans.predict(Kmeans2Data)

plt.figure(num=7)
Axes3D(plt.figure(num=7))
plt.axis(projection="3d")
plt.title("Kmeans: opseg prsa,opseg struka i težina")
plt.scatter(Kmeans2Data[labels[:] == 1,0], Kmeans2Data[labels[:] == 1,1], Kmeans2Data[labels[:] == 1,2], color="blue", label="M", alpha=0.5)
plt.scatter(Kmeans2Data[labels[:] == 0,0], Kmeans2Data[labels[:] == 0,1], Kmeans2Data[labels[:] == 0,2], color="red", label="F", alpha=0.5)
# plt.scatter(Kmeans1Data[labels[:] == 2,0], Kmeans1Data[labels[:] == 2,1], color="yellow", label="F", alpha=0.5)
# plt.view_init(10, 45)
plt.savefig("LV5_KMeans2.png")
# kmeans.labels_
# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_

gmm = GaussianMixture(n_components=nClusters).fit(Kmeans2Data)
labels = gmm.predict(Kmeans2Data)

plt.figure(num=8)
Axes3D(plt.figure(num=8))
plt.axis(projection="3d")
plt.title("Gauss: opseg prsa,opseg struka i težina")
plt.scatter(Kmeans2Data[labels[:] == 1,0], Kmeans2Data[labels[:] == 1,1], Kmeans2Data[labels[:] == 1,2], color="blue", label="M", alpha=0.5)
plt.scatter(Kmeans2Data[labels[:] == 0,0], Kmeans2Data[labels[:] == 0,1], Kmeans2Data[labels[:] == 0,2], color="red", label="F", alpha=0.5)
# plt.scatter(Kmeans1Data[labels[:] == 2,0], Kmeans1Data[labels[:] == 2,1], color="yellow", label="F", alpha=0.5)
# plt.view_init(10, 45)
plt.savefig("LV5_Gauss2.png")   
