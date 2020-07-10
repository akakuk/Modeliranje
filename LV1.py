import scipy.signal as sig
import scipy.stats as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalizeData(inVector, lBound = 0, uBound = 1):
    xMax = max(inVector)
    xMin = min(inVector)
    outVector = [None] * len(inVector)
    for i in range(0, len(inVector)):
        outVector[i] = (((inVector[i] - xMin) / (xMax - xMin)) * uBound) + lBound
    return outVector

def standardizeData(inVector):
    xMean = np.mean(inVector)
    xDeviation = np.std(inVector)
    outVector = [None] * len(inVector)
    for i in range(0, len(inVector)):
        outVector[i] = (inVector[i] - xMean) / xDeviation
    return outVector

#Prvo sam učitao Data.txt file u excel i spremio ga kao Data.csv
#Učitavanje Data.csv pomoću pandasa i kastanje u format pogodan za korištenje (float)
data = pd.read_csv("Data.csv", sep = ";", header = None, names = ["t", "x", "y"])
t = data["t"].astype(float)
x = data["x"].astype(float)
y = data["y"].astype(float)

NaNs = []
#Uklanjanje i zamijena NaN vrijednosti iz podataka
for i in range(0, len(y)):
    if(np.isnan(y[i])):
        print(str(y[i]) + " at " + str(i) + ". place in y.")
        #y[i]=(0.25 * y[i-2] + 0.75 * y[i-1] + 0.75 * y[i+1] + 0.25 * y[i+2] ) / 2
        y[i] = np.interp(i, t[i-10:i+10], y[i-10:i+10])
        NaNs.append(i)
    if(np.isnan(x[i])):
        print(str(x[i]) + " at " + str(i) + ". place in x.")
        #x[i]=(x[i-1] + x[i+1]) / 2
        x[i] = np.interp(i, t[i-10:i+10], x[i-10:i+10])
        NaNs.append(i)
    if(np.isnan(t[i])):
        print(str(y[i]) + " at " + str(i) + ". place in t.")
        #t[i]=(t[i-1] + t[i+1]) / 2
        t[i] = np.interp(i, t[i-10:i+10], t[i-10:i+10])
        NaNs.append(i)
plt.figure(num = 0, figsize = (6.4 * 15, 4.8 * 5))
plt.title(label = "Detected NaNs", fontsize = 60)
plt.plot(t, y)
for NaNIndex in NaNs:
    plt.plot(t[NaNIndex], y[NaNIndex], color="r", marker="o")
plt.savefig("NaNs.png")
     
#Kako bi uklonili outliere potrebno je "zagladiti" izlaznu veličinu median filterom
#Ovdje se uspoređuju različite veličine pomičnog prozora median filteri
#Bitno je zagladiti izlazne vrijednosti ali ne izgubiti korisne informacije
#Pomični prozor veličine 5 ne izgladi podatke o izlazu dovoljno, dok onaj veličine 51 izrezuje vrhove i s njima korisne informacije
plt.figure(num = 1, figsize = (6.4 * 15, 4.8 * 5))
plt.plot(t, y)
y_median5 = sig.medfilt(y, 5)
plt.plot(t, y_median5, color = "y")
y_median17 = sig.medfilt(y, 17)
plt.plot(t, y_median17, color = "r")
y_median51 = sig.medfilt(y, 51)
plt.plot(t, y_median51, color = "g")
plt.title(label = "Influence of sliding window size on median filter", fontsize = 60)
plt.legend(["Original output", "Median filter 5", "Median filter 17", "Median filter 51"], fontsize = 40)
plt.savefig("y_median.png")

#U ovom koraku računamo razliku između odabrane filtrirane izlazne funkcije i stvarne izlazne funkcije i spremamo u vektor razlika
y_deviation = []
for i in range(0, len(y)):
    y_deviation.append(y[i]-y_median17[i])
#Koristeći vektor razlika računamo medijansku apsolutnu devijaciju
deviation = stat.median_absolute_deviation(y_deviation)
#Sada koristeći devijaciju i vektor razlika možemo detektirati i ukloniti outliere
#Detektiramo svaku razliku između filtrirane i stvarne izlazne veličine koja je veća od 3 srednje devijacije
outliers = []
for i in range(0, len(y)):
    if(abs(y_deviation[i]) >= 3 * deviation):
        outliers.append(i)  
    
plt.figure(num=2, figsize = (6.4 * 15, 4.8 * 5))
plt.title(label = "Detected outliers", fontsize = 60)
plt.plot(t, y)
for outlierIndex in outliers:
    plt.plot(t[outlierIndex], y[outlierIndex], color="r", marker="o")
plt.savefig("outliers.png")

#Izbacivanje outliera
y_new = []
x_new = []
for i in range(0, len(y)):
    if(abs(y_deviation[i]) < 3 * deviation):
        y_new.append(y[i])
        x_new.append(x[i])
y = np.array(y_new).flatten()
x = np.array(x_new).flatten()
t = (np.arange(0, (5001 - len(outliers)) * 0.2, 0.2)).flatten()

#Izračun srednje vrijednosti 
y_mean = np.mean(y)
#Izračun najbolje linearne aproksimacije
coeff = np.polyfit(t, y, 1)
y_linear = []
#Budući da se dobiju samo koeficijenti pravca potrebno je uvrstiti sve t vrijednosti u jednadžbu pravca kako bi dobili pravac
for i in range(0, len(y)):
    y_linear.append(coeff[0] * t[i] + coeff[1])
#Oduzimanje srednje vrijednosti i vrijednosti najbolje linearne aproksimacije od original vrijednosti
y_mean_removed = y - y_mean
y_linear_removed = y - y_linear
plt.figure(num = 3, figsize = (6.4 * 15, 4.8 * 5))
plt.title(label = "Removing average and best linear approximation", fontsize = 60)
plt.plot(t, y)
plt.plot(t, y_mean_removed, color = "r")
plt.plot(t, y_linear_removed, color = "y")
plt.legend(["Original output", "Original output - average", "Original output - best linear approximation"], fontsize = 40)
plt.savefig("y_linear_and_mean_remove.png")


#Djeljenje podataka na trening i test skupove (80% train i 20% test)
y_train = y[0:round(0.8*len(y))]
x_train = x[0:round(0.8*len(x))]
t_train = t[0:round(0.8*len(t))]
y_test = y[round(0.8*len(y)):len(y)]
x_test = x[round(0.8*len(x)):len(x)]
t_test = t[round(0.8*len(t)):len(t)]

df = pd.DataFrame([t, x, y])
df.to_csv("preparedData.csv", index=False, sep=";")

