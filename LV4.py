import scipy.io as sio
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

data = sio.loadmat("bodydata.mat")

bodyData = data["bodydata"]
maleBodyData =  bodyData[bodyData[:,16] == 1]
femaleBodyData = bodyData[bodyData[:,16] == 0]
table = np.zeros([6, 16]);
f = open("LV4_output.txt", "w")
workbook = xlsxwriter.Workbook('LV4.xlsx')
worksheet = workbook.add_worksheet()
attributes = ["razmak izmedu ramena", "opseg ramena", "opseg prsa", "opseg struka", "opseg struka oko pupka", "opseg bokova", "opseg bedra", "opseg bicepsa", "opseg podlaktice", "opseg koljena ispod casice", "max. opseg lista", "min. opseg gleznja", "opseg zapesca", "age", "weight", "height"]
worksheet.write(2, 1, "Mean male")
worksheet.write(3, 1, "Median male")
worksheet.write(4, 1, "STDev male")
worksheet.write(5, 1, "Mean female")
worksheet.write(6, 1, "Median female")
worksheet.write(7, 1, "STDev female")



for i in range(0,16):
    table[0,i] = np.mean(maleBodyData[:,i])
    table[1,i] = np.median(maleBodyData[:,i])
    table[2,i] = np.std(maleBodyData[:,i])
    table[3,i] = np.mean(femaleBodyData[:,i])
    table[4,i] = np.median(femaleBodyData[:,i])
    table[5,i] = np.std(femaleBodyData[:,i])
    
    f.write("Mean male " + str(attributes[i]) + " : " + str(table[0,i]) + "\n")
    f.write("Median male " + str(attributes[i]) + " : " + str(table[1,i]) + "\n")
    f.write("Standard deviation male " + str(attributes[i]) + " : " + str(table[2,i]) + "\n")
    f.write("Mean female " + str(attributes[i]) + " : " + str(table[3,i]) + "\n")
    f.write("Median female " + str(attributes[i]) + " : " + str(table[4,i]) + "\n")
    f.write("Standard deviation female " + str(attributes[i]) + " : " + str(table[5,i]) + "\n")
    f.write("\n")
    
    worksheet.write(1, i + 2, attributes[i])
    worksheet.write(2, i + 2, table[0,i])
    worksheet.write(3, i + 2, table[1,i])
    worksheet.write(4, i + 2, table[2,i])
    worksheet.write(5, i + 2, table[3,i])
    worksheet.write(6, i + 2, table[4,i])
    worksheet.write(7, i + 2, table[5,i])


    plt.figure(num = 0 * len(attributes) + i)
    plt.hist(maleBodyData[:,i], color="blue", label="M", fc=(0,0,1,0.5))
    plt.hist(femaleBodyData[:,i], color="red", label="F", fc=(1,0,0,0.5))
    plt.title(attributes[i])
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("LV4_Hist_" + str(attributes[i]) + ".png")
    
    ecdfMale = ECDF(maleBodyData[:,i])
    ecdfFemale = ECDF(femaleBodyData[:,i])
    plt.figure(num = 1 * len(attributes) + i)
    plt.title(attributes[i])
    plt.plot(ecdfFemale.x, ecdfFemale.y, color="red")
    plt.plot(ecdfMale.x, ecdfMale.y, color="blue")
    male25 = np.percentile(maleBodyData[:,i], 25)
    male75 = np.percentile(maleBodyData[:,i], 75)
    female25 = np.percentile(femaleBodyData[:,i], 25)
    female75 = np.percentile(femaleBodyData[:,i], 75)
    plt.plot(male25, 0.25, "bo")
    plt.plot(male75, 0.75, "bo")
    plt.plot(female25, 0.25, "ro")
    plt.plot(female75, 0.75, "ro")
    plt.savefig("LV4_ECDF_" + str(attributes[i]) + ".png")

    plt.figure(num = 2 * len(attributes) + i)
    plt.title(attributes[i])
    plt.boxplot([maleBodyData[:,i], femaleBodyData[:,i]] , labels=["M", "F"])
    plt.savefig("LV4_Boxplot_" + str(attributes[i]) + ".png")
c = 0    
for i in [0, 2, 3, 4]:
    for j in [14, 15]:
        # plt.figure(num = 3 * len(attributes) + c)
        # plt.scatter(maleBodyData[:,i], maleBodyData[:,j])
        # plt.xlabel(attributes[i])
        # plt.ylabel(attributes[j])
        # plt.savefig("LV4_Boxplot_" + str(attributes[i]) + "-" + str(attributes[j]) + ".png")
        # c += 1

        fig = sns.jointplot(maleBodyData[:,j], maleBodyData[:,i])
        fig.set_axis_labels(attributes[j], attributes[i])
        fig.savefig("LV4_scatterhist_" + str(attributes[i]) + "-" + str(attributes[j]) + ".png")
    

    
workbook.close()
f.close()