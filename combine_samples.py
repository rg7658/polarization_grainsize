from __future__ import print_function
import numpy as np
from tkinter import *
from tkinter import messagebox
import pandas as pd
from get_inputs import *
import spectral.io.envi as envi
from spectral import *
import matplotlib as mp
mp.rcParams.update({'axes.titlesize': 14,'xtick.labelsize': 12,'ytick.labelsize' : 12, 'legend.fontsize' : 12, 'figure.titlesize' : 16, 'axes.labelsize':14})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

############### generalized version ################

# choose folder of csv file data to work with
path='D:/Polarization_Results/combined_results_Nepheline'
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
    if 'Pmax' in sub_dir:
        label_Pmax.append(str(sub_dir[8:-16])+str(u"\u03bcm")) # first number 6 for lab olivine, 8 for AGSCO samples
        max_path = os.path.join(folder, sub_dir)
        maxData= np.genfromtxt(max_path, delimiter=',', skip_header=1)
        lam=maxData[:,1]
        gmax.append(maxData[:,2])
        Pmax.append(maxData[:,3])

    elif 'Pmin' in sub_dir:
        label_Pmin.append(str(sub_dir[8:-16])+str(u"\u03bcm"))
        min_path = os.path.join(folder, sub_dir)
        minData= np.genfromtxt(min_path, delimiter=',', skip_header=1)
        lam=minData[:,1]
        gmin.append(minData[:,2])
        Pmin.append(minData[:,3])

    elif 'Pmean.csv' in sub_dir:
        label_Pmean.append(str(sub_dir[8:-17])+str(u"\u03bcm"))
        mean_path = os.path.join(folder, sub_dir)
        meanData = np.genfromtxt(mean_path, delimiter=',', skip_header=1)
        P_mean.append(meanData[:,2:])
        alpha.append(meanData[:,1])

    elif 'PmeanSmooth' in sub_dir:
        label_ms.append(str(sub_dir[8:-23])+str(u"\u03bcm"))
        ms_path = os.path.join(folder, sub_dir)
        msData = np.genfromtxt(ms_path, delimiter=',', skip_header=1)
        P_ms.append(meanData[:,2:])
        al.append(meanData[:,1])


# plot data
p=plt.figure()
for i in range(len(P_mean)):
    plt.plot(alpha[i],P_mean[i][:,144],'.')
    plt.title('Mean Polarization 633nm '+sample)
    plt.ylim(-0.01,.14)
    plt.ylabel('Polarization')
    plt.xlabel('Phase Angle (deg)')
plt.gca().legend((label_Pmean))
plt.show()
p.savefig(os.path.join(path,'averaged_data_633nm_'+sample+'.png'))

p=plt.figure()
for i in range(len(P_mean)):
    plt.plot(alpha[i],P_mean[i][:,16],'.')
    plt.title('Mean Polarization 425nm '+sample)
    plt.ylabel('Polarization')
    plt.xlabel('Phase Angle (deg)')
plt.gca().legend((label_Pmean))
plt.show()
p.savefig(os.path.join(path,'averaged_data_425nm_'+sample+'.png'))

p=plt.figure()
for i in range(len(P_mean)):
    plt.plot(alpha[i],P_mean[i][:,278],'.')
    plt.title('Mean Polarization 850nm '+sample)
    plt.ylabel('Polarization')
    plt.xlabel('Phase Angle (deg)')
plt.gca().legend((label_Pmean))
plt.show()
p.savefig(os.path.join(path,'averaged_data_850nm_'+sample+'.png'))

#plot mean smoothened data
p=plt.figure()
for i in range(len(P_ms)):
    plt.plot(alpha[i],P_ms[i][:,144],'.')
    plt.title('Smoothened Mean Polarization 633nm '+sample)
    plt.ylabel('Polarization')
    plt.xlabel('Phase Angle (deg)')
plt.gca().legend((label_ms))
plt.show()
p.savefig(os.path.join(path,'avg_sm_data_633nm_'+sample+'.png'))


#plot min and max data
fig4,ax4=plt.subplots(1,2)
for i in range(len(Pmin)):
    ax4[0].plot(lam,Pmin[i])
    ax4[0].set_ylabel('Polarization Minimum  ')
    ax4[0].set_xlabel('wavelength(nm)')
    ax4[1].scatter(lam,gmin[i],s=3)
    ax4[1].set_ylabel('Phase angle of Polarization Minimum')
    ax4[1].set_xlabel('wavelength(nm)')
    ax4[0].set_ylim([-0.03,0.25])
    ax4[1].set_ylim([0,50])
plt.suptitle('P_min and g_min vs band '+sample, y=1.0, ha='center')
plt.tight_layout(pad=4)
plt.gca().legend((label_Pmin))
plt.show()
fig4.savefig(os.path.join(path,'wavelength_min_effects_'+sample+'.png'))

fig4,ax4=plt.subplots(1,2)
for i in range(len(Pmin)):
    ax4[0].plot(lam,Pmax[i])
    ax4[0].set_ylabel('Polarization Maximum ')
    ax4[0].set_xlabel('wavelength(nm)')
    ax4[1].scatter(lam,gmax[i],s=3)
    ax4[1].set_ylabel('Phase angle of Polarization Maximum')
    ax4[1].set_xlabel('wavelength(nm)')
    ax4[0].set_ylim([0,0.4])
    ax4[1].set_ylim([75, 120])
plt.suptitle('P_max and g_max vs band '+sample, y=1.0, ha='center')
plt.tight_layout(pad=4)
plt.gca().legend((label_Pmax))
plt.show()
fig4.savefig(os.path.join(path,'wavelength_max_effects_'+sample+'.png'))

############### polyfits: whole range of data ####################

# for n in range(1,6):
#     p = plt.figure()
#     for i in range(len(P_mean)):
#         f_avg=np.poly1d(np.polyfit(alpha[i],P_mean[i][:,144],n))
#         x_new=np.linspace(0,110,5000)
#         y_avg=f_avg(x_new)
#         plt.plot(x_new,y_avg)
#         plt.title('Polyfit Polarization, 633nm, n=%i '%n +sample)
#         #plt.ylim(0.1, .25)
#         plt.ylabel('P, ratio out of 1')
#         plt.xlabel('Phase Angle (deg)')
#
#     plt.gca().legend((label_Pmean))
#     plt.show()
#     p.savefig(os.path.join(path,'polyfit'+str(n)+'_633nm_'+sample+'.png'))
#
#     p = plt.figure()
#     for i in range(len(P_mean)):
#         f_avg=np.poly1d(np.polyfit(alpha[i],P_mean[i][:,16],n))
#         x_new=np.linspace(0,110,5000)
#         y_avg=f_avg(x_new)
#         plt.plot(x_new,y_avg)
#         plt.title('Polyfit Polarization, 425nm, n=%i '%n +sample)
#         plt.ylabel('P, ratio out of 1')
#         plt.xlabel('Phase Angle (deg)')
#
#     plt.gca().legend((label_Pmean))
#     plt.show()
#     p.savefig(os.path.join(path,'polyfit'+str(n)+'_425nm_'+sample+'.png'))
#
#     p = plt.figure()
#     for i in range(len(P_mean)):
#         f_avg=np.poly1d(np.polyfit(alpha[i],P_mean[i][:,278],n))
#         x_new=np.linspace(0,110,5000)
#         y_avg=f_avg(x_new)
#         plt.plot(x_new,y_avg)
#         plt.title('Polyfit Polarization, 850nm, n=%i '%n +sample)
#         plt.ylabel('P, ratio out of 1')
#         plt.xlabel('Phase Angle (deg)')
#
#     plt.gca().legend((label_Pmean))
#     plt.show()
#     p.savefig(os.path.join(path,'polyfit'+str(n)+'_850nm_'+sample+'.png'))

