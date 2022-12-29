from __future__ import print_function
import numpy as np
from tkinter import *
from tkinter import messagebox
import pandas as pd
from get_inputs import *
import spectral.io.envi as envi
from spectral import *
import matplotlib as mp
mp.rcParams['figure.figsize'] = [8,6]
mp.rcParams.update({'axes.titlesize': 14,'xtick.labelsize': 12,'ytick.labelsize' : 12, 'legend.fontsize' : 12, 'figure.titlesize' : 16, 'axes.labelsize':14})
import matplotlib.pyplot as plt
import scipy.signal as ss
import glob
import os,shutil
import statsmodels.formula.api as smf

################### generalized version #################
path='D:/Polarization_Results/RMSE_NephSmall'
if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)

names=path.split('_')
sample='AGSCO Nepheline 1-5 Microns'

########## process data and labels for plotting
meanData=np.genfromtxt('D:\Polarization_Results\csv_files_nephelineSmall_AGSCO/aprAGSCO1to5microns_Pmean.csv',delimiter=',',skip_header=1)
#
msData=np.genfromtxt('D:\Polarization_Results\csv_files_nephelineSmall_AGSCO/aprAGSCO1to5microns_PmeanSmooth.csv',delimiter=',',skip_header=1)

minData=np.genfromtxt('D:\Polarization_Results\csv_files_nephelineSmall_AGSCO/aprAGSCO1to5microns_Pmin.csv',delimiter=',',skip_header=1)

maxData=np.genfromtxt('D:\Polarization_Results\csv_files_nephelineSmall_AGSCO/aprAGSCO1to5microns_Pmax.csv',delimiter=',',skip_header=1)

P_mean=meanData[:,2:]
alpha=meanData[:,1]

P_ms=msData[:,2:]
al=msData[:,1]

lam=minData[:,1]
gmin=minData[:,2]
Pmin=minData[:,3]

gmax=maxData[:,2]
Pmax=maxData[:,3]

######### Polyfits ############

for n in range(1,8):
    x=alpha
    f_avg=np.poly1d(np.polyfit(x,P_mean[:,144],n))
    _,res,_,_,_=np.polyfit(x,P_mean[:,144],n,full=True)
    x_new=np.linspace(0,120,5000)
    y_avg=f_avg(x_new)

    rmse=np.sqrt(res[0])


    p=plt.figure(figsize=(8,6))
    plt.plot(x_new,y_avg,label='RMSE='+str(round(rmse,4)))
    plt.title('Polyfit Polarization and RMSE at 633nm, n=%i'%n)
    plt.ylabel('Polarization')
    plt.xlabel('Phase Angle (deg)')
    plt.legend(loc="upper left")
    plt.show()
    p.savefig(os.path.join(path,'polyfit'+str(n)+'_rmse_analysis.png'))


########## Spline ###############
from scipy import interpolate
idx=np.argsort(alpha)
x=alpha[idx]
f=P_mean[idx,144]
tck=interpolate.splrep(x,f,k=3)
x_new=np.linspace(5,110,3000)
y=interpolate.splev(x_new,tck,der=0)
_,res,_,_=interpolate.splrep(x,f,k=3, full_output=True)
rmse=np.sqrt(res)

# p = plt.figure(figsize=(8,6))
# plt.plot(x_new, y, label='RMSE=' + str(rmse))
# plt.title('Spline Fit Polarization and RMSE at 633nm')
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.legend(loc="upper left")
# plt.show()
# p.savefig(os.path.join(path, 'Spline_fit_rmse_analysis.png'))

p = plt.figure(figsize=(8,6))
plt.plot(alpha, P_mean[:,144],'.')
plt.title('Mean Polarization 633nm '+sample)
plt.ylabel('Polarization')
plt.xlabel('Phase Angle (deg)')
plt.legend(loc="upper left")
plt.show()
p.savefig(os.path.join(path, 'averaged_data_633nm_ZOOMEDin.png'))

p = plt.figure(figsize=(8,6))
plt.plot(alpha, P_mean[:,144],'.')
plt.title('Mean Polarization 633nm '+sample)
plt.ylabel('Polarization')
plt.xlabel('Phase Angle (deg)')
plt.ylim(-0.01,.14)
plt.legend(loc="upper left")
plt.show()
p.savefig(os.path.join(path, 'averaged_data_633nm.png'))

fig4,ax4=plt.subplots(1,2)
ax4[0].plot(lam,Pmin,'b')
ax4[0].set_ylabel('Polarization Minimum ')
ax4[0].set_xlabel('wavelength(nm)')
ax4[1].scatter(lam,gmin,c='r',s=3)
ax4[1].set_ylabel('Phase angle of Polarization Minimum')
ax4[1].set_xlabel('wavelength(nm)')
ax4[0].set_ylim([-0.03, 0.25])
ax4[1].set_ylim([0, 50])
plt.suptitle('P_min and g_min vs band '+' '+sample, y=1.0, ha='center')
plt.tight_layout(pad=4)
plt.show()
fig4.savefig(os.path.join(path,'wavelength_min_effects.png'))

fig4,ax4=plt.subplots(1,2)
ax4[0].plot(lam,Pmax,'b')
ax4[0].set_ylabel('Polarization Maximum ')
ax4[0].set_xlabel('wavelength(nm)')
ax4[1].scatter(lam,gmax,c='r',s=3)
ax4[1].set_ylabel('Phase angle of Polarization Maximum')
ax4[1].set_xlabel('wavelength(nm)')
ax4[0].set_ylim([0, 0.4])
ax4[1].set_ylim([75, 120])
plt.suptitle('P_max and g_max vs band '+' '+sample, y=1.0, ha='center')
plt.tight_layout(pad=4)
plt.show()
fig4.savefig(os.path.join(path,'wavelength_max_effects.png'))