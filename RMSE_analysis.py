from __future__ import print_function
import numpy as np
from tkinter import *
from tkinter import messagebox
import pandas as pd
from get_inputs import *
import spectral.io.envi as envi
from spectral import *
import matplotlib as mp
mp.rcParams['figure.figsize'] = [6.4, 5.8]
mp.rcParams.update({'axes.titlesize': 14,'xtick.labelsize': 12,'ytick.labelsize' : 12, 'legend.fontsize' : 10, 'figure.titlesize' : 16, 'axes.labelsize':14})
import matplotlib.pyplot as plt
import scipy.signal as ss
import glob
import os,shutil
import statsmodels.formula.api as smf

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

################### generalized version #################
path='D:/Polarization_Results/RMSE_Nepheline'
if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)

messagebox.showinfo('choose csv data directory to process')
folder=get_dir_path()
file_paths = sorted(os.listdir(folder), key=numericalSort)

names=path.split('_')
sample='AGSCO Nepheline'

#define lists
P_mean=[]
Pmin=[]
Pmax=[]
alpha=[]
P_ms=[]
al=[]
gmin=[]
gmax=[]

#create label lists
label_Pmean=[]
label_Pmin=[]
label_Pmax=[]
label_ms=[]

########## process data and labels for plotting
for sub_dir in file_paths:
    if 'Pmean.csv' in sub_dir:
        label_Pmean.append(str(sub_dir[8:-17])+str(u"\u03bcm"))  # first number 6 for lab olivine, 8 for AGSCO samples
        mean_path = os.path.join(folder, sub_dir)
        meanData = np.genfromtxt(mean_path, delimiter=',', skip_header=1)
        P_mean.append(meanData[:,2:])
        alpha.append(meanData[:,1])


############# Polyfits- whole region with RMSE ####################
for n in range(1,6):
    p = plt.figure()
    for i in range(len(P_mean)):
        f_avg=np.poly1d(np.polyfit(alpha[i],P_mean[i][:,144],n))
        _, res, _, _, _ =np.polyfit(alpha[i],P_mean[i][:,144],n, full=True)
        rmse=np.sqrt(res)
        x_new=np.linspace(0,110,5000)
        y_avg=f_avg(x_new)
        plt.plot(x_new,y_avg,label='x='+str(label_Pmean[i])+', RMSE='+str(round(rmse[0],4)))
        plt.title('Polyfit Polarization, 633nm, n=%i '%n +sample)
        #plt.ylim(0.1,0.25)
        plt.ylabel('Polarization')
        plt.xlabel('Phase Angle (deg)')
    plt.legend(loc="upper left")
    plt.show()
    p.savefig(os.path.join(path,'polyfit'+str(n)+'_633nm_'+sample+'.png'))

    p = plt.figure()
    for i in range(len(P_mean)):
        f_avg=np.poly1d(np.polyfit(alpha[i],P_mean[i][:,16],n))
        _, res, _, _, _ =np.polyfit(alpha[i],P_mean[i][:,16],n, full=True)
        rmse=np.sqrt(res)
        f_avg3=np.poly1d(np.polyfit(alpha[i],P_mean[i][:,278],n))
        _, res3, _, _, _ =np.polyfit(alpha[i],P_mean[i][:,278],n, full=True)
        rmse3=np.sqrt(res3)
        x_new=np.linspace(0,110,5000)
        y_avg=f_avg(x_new)
        y_avg3=f_avg3(x_new)
        plt.plot(x_new,y_avg,label='λ=425nm, x='+str(label_Pmean[i])+', RMSE='+str(round(rmse[0],4)))
        plt.plot(x_new,y_avg3,label='λ=850nm, x='+str(label_Pmean[i])+', RMSE='+str(round(rmse3[0],4)))
        plt.title('Polyfit Polarization, n=%i '%n +sample)
        plt.ylabel('Polarization')
        plt.xlabel('Phase Angle (deg)')

    plt.legend(loc='best', fontsize=10)
    plt.show()
    p.savefig(os.path.join(path,'polyfit'+str(n)+'425and850_'+sample+'.png'))


# Define Linear Region
P_m=[]
g=[]
for i in range(len(P_mean)):
    idxs=np.array(np.where(np.logical_and(alpha[i]>=20, alpha[i]<=80))).flatten()
    P_m.append(P_mean[i][idxs,:])
    g.append(alpha[i][idxs])

# plot polyfits for linear region...
for n in range(1,6):
    p = plt.figure()
    for i in range(len(P_mean)):
        f_avg=np.poly1d(np.polyfit(g[i],P_m[i][:,144],n))
        _, res, _, _, _ =np.polyfit(g[i],P_m[i][:,144],n, full=True)
        rmse=np.sqrt(res)
        x_new=np.linspace(20,80,int(5000*(80-20)/110))
        y_avg=f_avg(x_new)
        plt.plot(x_new,y_avg,label='x='+str(label_Pmean[i])+', RMSE='+str(round(rmse[0],4)))
        plt.title('Polyfit Linear Region, 633nm, n=%i '%n +sample)
        #plt.ylim(0.12,0.22)
        plt.ylabel('Polarization')
        plt.xlabel('Phase Angle (deg)')
    plt.legend(loc="upper left")
    plt.show()
    p.savefig(os.path.join(path,'polyfit'+str(n)+'_633nm_linearRegion_'+sample+'.png'))

    p = plt.figure()
    for i in range(len(P_mean)):
        f_avg=np.poly1d(np.polyfit(g[i],P_m[i][:,16],n))
        _, res, _, _, _ =np.polyfit(g[i],P_m[i][:,16],n, full=True)
        rmse=np.sqrt(res)
        x_new = np.linspace(20, 80, int(5000 * (80 - 20) / 110))
        y_avg=f_avg(x_new)
        plt.plot(x_new,y_avg,label='λ=425nm, x='+str(label_Pmean[i])+', \n RMSE='+str(round(rmse[0],4)))

    for i in range(len(P_mean)):
        f_avg3 = np.poly1d(np.polyfit(g[i], P_m[i][:, 278], n))
        _, res3, _, _, _ = np.polyfit(g[i], P_m[i][:, 278], n, full=True)
        rmse3 = np.sqrt(res3)
        x_new = np.linspace(20, 80, int(5000 * (80 - 20) / 110))
        y_avg3=f_avg3(x_new)
        plt.plot(x_new, y_avg3, label='λ=850nm, x=' + str(label_Pmean[i]) + ', \n RMSE=' + str(round(rmse3[0], 4)))
        plt.title('Polyfit Linear Region, n=%i \n' % n + sample)
        plt.ylabel('Polarization')
        plt.xlabel('Phase Angle (deg)')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.ylim(0.1,0.3)
    plt.show()
    p.savefig(os.path.join(path,'polyfit'+str(n)+'425and850_linearRegion_'+sample+'.png'))