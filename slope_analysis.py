from __future__ import print_function
import numpy as np
from tkinter import *
from tkinter import messagebox
import pandas as pd
from get_inputs import *
import spectral.io.envi as envi
from spectral import *
import matplotlib as mp
mp.rcParams.update({'axes.titlesize': 14,'xtick.labelsize': 12,'ytick.labelsize' : 12, 'legend.fontsize' : 10, 'figure.titlesize' : 16, 'axes.labelsize':14})
import matplotlib.pyplot as plt
import scipy.signal as ss
import glob
import os,shutil
import statsmodels.formula.api as smf

path='D:/Polarization_Results/Slope_lam_Lab Olivine'
if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)

names=path.split('_')
sample='Washington Mills Olivine'


################### generalized version #################
messagebox.showinfo('choose csv data directory to process')
folder=get_dir_path()
file_paths = os.listdir(folder)

#define lists
P_mean=[]
alpha=[]

#create label lists
label_Pmean=[]

########## process data and labels for plotting
for sub_dir in file_paths:

    if 'Pmean.csv' in sub_dir:
        label_Pmean.append(str(sub_dir[6:-17])+str(u"\u03bcm")) # first number 6 for lab olivine, 8 for AGSCO samples
        mean_path = os.path.join(folder, sub_dir)
        meanData = np.genfromtxt(mean_path, delimiter=',', skip_header=1)
        P_mean.append(meanData[:,2:])
        alpha.append(meanData[:,1])

    elif 'Pmin' in sub_dir:
        min_path = os.path.join(folder, sub_dir)
        minData= np.genfromtxt(min_path, delimiter=',', skip_header=1)
        lam=minData[:,1]

# Define Linear Region
P_m=[]
g=[]
for i in range(len(P_mean)):
    idxs=np.array(np.where(np.logical_and(alpha[i]>=20, alpha[i]<=80))).flatten()
    P_m.append(P_mean[i][idxs,:])
    g.append(alpha[i][idxs])

# Plot stuff
lam3=np.array([16,144,278])
lam100s=np.array([63,124,185,247,308])
lam50s=np.array([32,93,155,216,278,339])
x_new = np.linspace(20, 80, 5000)

# plot polyfits for linear region...
for i in range(len(P_m)):
    p = plt.figure()
    for x,j in enumerate(lam3):
        f_avg=np.poly1d(np.polyfit(g[i],P_m[i][:,j],1))
        x_new=np.linspace(20,80,int(5000*(80-20)/110))
        y_avg=f_avg(x_new)
        plt.plot(x_new,y_avg,label='λ='+str(lam[j]) + 'nm: ' + str(f_avg))
        plt.title('Polyfit Linear Region, slopes for \n x=' +str(label_Pmean[i])+' '+sample)
        plt.ylabel('Polarization')
        plt.xlabel('Phase Angle (deg)')
        plt.legend(loc="upper left")
    plt.show()
    p.savefig(os.path.join(path,'slope'+str(label_Pmean[i])+'3lam_linearRegion_'+sample+'.png'))

for i in range(len(P_m)):
    p = plt.figure()
    for x,j in enumerate(lam50s):
        f_avg=np.poly1d(np.polyfit(g[i],P_m[i][:,j],1))
        x_new=np.linspace(20,80,int(5000*(80-20)/110))
        y_avg=f_avg(x_new)
        plt.plot(x_new,y_avg,label='λ='+str(lam[j]) + 'nm: ' + str(f_avg))
        plt.title('Polyfit Linear Region, slopes for \n x='+str(label_Pmean[i])+' '+sample)
        plt.ylabel('Polarization')
        plt.xlabel('Phase Angle (deg)')
        plt.legend(loc="upper left")
    plt.show()
    p.savefig(os.path.join(path,'slope'+str(label_Pmean[i])+'50slam_linearRegion_'+sample+'.png'))

for i in range(len(P_m)):
    p = plt.figure()
    for x, j in enumerate(lam100s):
        f_avg = np.poly1d(np.polyfit(g[i], P_m[i][:, j], 1))
        x_new = np.linspace(20, 80, int(5000 * (80 - 20) / 110))
        y_avg = f_avg(x_new)
        plt.plot(x_new, y_avg, label='λ=' + str(lam[j]) + 'nm: ' + str(f_avg))
        plt.title('Polyfit Linear Region, slopes for \n x=' + str(label_Pmean[i])+' '+sample)
        plt.ylabel('Polarization')
        plt.xlabel('Phase Angle (deg)')
        plt.legend(loc="upper left")
    plt.show()
    p.savefig(os.path.join(path, 'slope' + str(label_Pmean[i]) + '100slam_linearRegion_'+sample+'.png'))