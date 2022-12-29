from __future__ import print_function
import numpy as np
from tkinter import *
from tkinter import messagebox
import pandas as pd
from get_inputs import *
import spectral.io.envi as envi
from spectral import *
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy.signal as ss
import glob
import os,shutil
import statsmodels.formula.api as smf
mp.rcParams['figure.figsize'] = [12,9]
mp.rcParams.update({'axes.titlesize': 16,'xtick.labelsize': 14,'ytick.labelsize' : 14, 'legend.fontsize' : 12, 'figure.titlesize' : 18, 'axes.labelsize':16})

path='D:/Polarization_Results/Slope_grainSize_2022'


names=path.split('_')
sample='AGSCO Silica'


################### generalized version #################
messagebox.showinfo('choose csv data directory to process')
folder=get_dir_path()
file_paths = os.listdir(folder)

#define lists
P_mean=[]
alpha=[]

#create label lists
label_Pmean=[]

####### Define Grain Size Arrays ############
x_sil=np.array([308,452,550,725])
x_neph=np.array([446,550,704])
x_labOlivine=np.array([331, 568, 178])
x_AGSCOolivine=np.array([379,463,550,735,247])

if sample=='AGSCO Nepheline':
    x=x_neph
    x_new=np.linspace(400,700,5000)

elif sample=='AGSCO Silica':
    x=x_sil
    x_new=np.linspace(300,750,5000)

elif sample == 'AGSCO Olivine':
    x=x_AGSCOolivine
    x_new=np.linspace(200,750,5000)

elif sample=='Washington Mills Olivine':
    x=x_labOlivine
    x_new = np.linspace(160, 600, 5000)

########## process data and labels for plotting
for sub_dir in file_paths:

    if 'Pmean.csv' in sub_dir:
        label_Pmean.append(str(sub_dir[8:-17])+str(u"\u03bcm")) # first number 6 for lab olivine, 8 for AGSCO samples
        print(label_Pmean)
        mean_path = os.path.join(folder, sub_dir)
        meanData = np.genfromtxt(mean_path, delimiter=',', skip_header=1)
        P_mean.append(meanData[:,2:])
        alpha.append(meanData[:,1])

    if 'Pmin' in sub_dir:
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

# plot 3d slope/grain size for linear region...
slope=np.zeros((len(P_m),len(lam)))

p = plt.figure()

for i in range(len(P_m)):
    for j, val in enumerate(lam):
        m,b=np.polyfit(g[i],P_m[i][:,j],1)
        slope[i,j]=m

#print(lam[93],lam[124],lam[155],lam[185])
m_550=slope[:,93]
m_600=slope[:,124]
m_650=slope[:,155]
m_700=slope[:,185]

m=np.vstack((m_550, m_600,m_650,m_700))

lams=['550nm','600nm','650nm','700nm']
colors=['b','r','g','c']

#rsquared function (https://www.statology.org/quadratic-regression-python/)
def rsquared(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results= ssreg / sstot

    return results

p=plt.figure()
for i in range(4):
    plt.plot(x, m[i], 'o', c=colors[i])
leg1=plt.legend(lams, loc='upper left')
for i in range(4):
    f = np.poly1d(np.polyfit(x, m[i], 1))
    r_sq=rsquared(x, m[i], 1)
    y = f(x_new)
    _, res, _, _, _ = np.polyfit(x, m[i], 1, full=True)
    rmse = np.sqrt(res)
    plt.plot(x_new,y,'-',c=colors[i],label=str(f)+', $r^2$='+str(round(r_sq,3))+', RMSE='+str(round(rmse[0],5)))
    plt.title('Grain Size dependence of Polarization Slope in Linear \n Region (g=20-80 deg) for '+str(sample)+' Dataset')
    plt.ylabel('Slope of linear fit')
    plt.xlabel('grain size ('+str(u"\u03bcm")+')')
    plt.xlim([150,850])
    plt.ylim([.0003, .0014])
leg2=plt.legend(loc='upper right')
p.add_artist(leg1)
plt.show()
p.savefig(os.path.join(path,'slope_grainSize_'+str(sample)+'1.png'))

p=plt.figure()
for i in range(4):
    plt.plot(x, m[i], 'o', c=colors[i])
leg1=plt.legend(lams, loc='upper left')
for i in range(4):
    f = np.poly1d(np.polyfit(x, m[i], 2))
    fmatr=np.polyfit(x, m[i], 2)
    f_mat=[]
    for n in range(len(fmatr)):
        f_mat.append('{:0.3e}'.format(fmatr[n]))
    f_eq=str(f_mat[0])+'$x^2$+'+str(f_mat[1])+'x+'+str(f_mat[2])
    r_sq=rsquared(x, m[i], 2)
    y = f(x_new)
    _, res, _, _, _ = np.polyfit(x, m[i], 2, full=True)
    rmse = np.sqrt(res)
    if len(res) == 0:
        rmse=np.array([0])
    plt.plot(x_new,y,'-',c=colors[i],label=str(f_eq)+', $r^2$='+str(round(r_sq,3))+', RMSE='+str(round(rmse[0],5)))
    plt.title('Grain Size dependence of Polarization Slope in Linear \n Region (g=20-80 deg) for '+str(sample)+' Dataset')
    plt.ylabel('Slope of linear fit')
    plt.xlabel('grain size ('+str(u"\u03bcm")+')')
    plt.xlim([150,850])
    plt.ylim([.0003, .0014])
leg2=plt.legend(loc='upper right')
p.add_artist(leg1)
plt.show()
p.savefig(os.path.join(path,'slope_grainSize_'+str(sample)+'2.png'))

p=plt.figure()
for i in range(4):
    plt.plot(x, m[i], 'o', c=colors[i])
leg1=plt.legend(lams, loc='upper left')
for i in range(4):
    f = np.poly1d(np.polyfit(x, m[i], 3))
    fmatr=np.polyfit(x, m[i], 3)
    f_mat=[]
    for n in range(len(fmatr)):
        f_mat.append('{:0.3e}'.format(fmatr[n]))
    f_eq=str(f_mat[0])+'$x^3$+'+str(f_mat[1])+'$x^2$+'+str(f_mat[2])+'x+'+str(f_mat[3])
    r_sq=rsquared(x, m[i], 3)
    y = f(x_new)
    _, res, _, _, _ = np.polyfit(x, m[i], 3, full=True)
    rmse = np.sqrt(res)
    if len(res) == 0:
        rmse=np.array([0])
    plt.plot(x_new,y,'-',c=colors[i],label=str(f_eq)+', $r^2$='+str(round(r_sq,3))+', RMSE='+str(round(rmse[0],5)))
    plt.title('Grain Size dependence of Polarization Slope in Linear \n Region (g=20-80 deg) for '+str(sample)+' Dataset')
    plt.ylabel('Slope of linear fit')
    plt.xlabel('grain size ('+str(u"\u03bcm")+')')
    plt.xlim([150,850])
    plt.ylim([.0003, .0014])
leg2=plt.legend(loc='upper right')
p.add_artist(leg1)
plt.show()
p.savefig(os.path.join(path,'slope_grainSize_'+str(sample)+'3.png'))

p=plt.figure()
for i in range(4):
    plt.plot(x, m[i], 'o', c=colors[i])
leg1=plt.legend(lams, loc='upper left')
for i in range(4):
    f = np.poly1d(np.polyfit(x, m[i], 4))
    fmatr = np.polyfit(x, m[i], 4)
    f_mat = []
    for n in range(len(fmatr)):
        f_mat.append('{:0.3e}'.format(fmatr[n]))
    f_eq = str(f_mat[0]) + '$x^4$+' +str(f_mat[1]) + '$x^3$+' + str(f_mat[2]) + '$x^2$+' + str(f_mat[3]) + 'x+' + str(f_mat[4])
    r_sq=rsquared(x, m[i], 4)
    y = f(x_new)
    _, res, _, _, _ = np.polyfit(x, m[i], 4, full=True)
    rmse = np.sqrt(res)
    if len(res) == 0:
        rmse=np.array([0])
    plt.plot(x_new,y,'-',c=colors[i],label=str(f_eq)+', $r^2$='+str(round(r_sq,3))+', RMSE='+str(round(rmse[0],5)))
    plt.title('Grain Size dependence of Polarization Slope in Linear \n Region (g=20-80 deg) for '+str(sample)+' Dataset')
    plt.ylabel('Slope of linear fit')
    plt.xlabel('grain size ('+str(u"\u03bcm")+')')
    plt.xlim([150,850])
    plt.ylim([.0003, .0014])
leg2=plt.legend(loc='upper right')
p.add_artist(leg1)
plt.show()
p.savefig(os.path.join(path,'slope_grainSize_'+str(sample)+'4.png'))
