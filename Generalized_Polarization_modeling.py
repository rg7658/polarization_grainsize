# Polarization modeling for hyperspectral image data
#
# Author: Rachel Golding
#
# Script to process polarized data from the GRIT lab's camera and calculate polarization from perpendicular
# and parallel images. Requires one image with polarizer at 0 deg and 1 at 90 for each orientation of the
#light source used. Assumes lab setup remains constant other than light angle, camera angle, and polarization
# angle.No white Reference required. Images must be .img files with matching envi header file.
#NOTE: light and polarizer angles are parsed from subdirectory name, so name files accordingly
#
# References: A significant portion of code in the main for loop was found in Chris Lapynski's "Subset Polar
# Example" program, which has been altered for the desired output of this program.

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


#read in calibration files
slope_par=np.genfromtxt('D:\Feb-8-2022_sphere_recal\Feb-8-2022_sphere_recal-0deg-pol-slopes.csv', delimiter=',') #0deg
int_par=np.genfromtxt('D:\Feb-8-2022_sphere_recal\Feb-8-2022_sphere_recal-0deg-pol-intercepts.csv', delimiter=',')
slope_per=np.genfromtxt('D:\Feb-8-2022_sphere_recal\Feb-8-2022_sphere_recal-90deg-pol-slopes.csv', delimiter=',') #90deg
int_per=np.genfromtxt('D:\Feb-8-2022_sphere_recal\Feb-8-2022_sphere_recal-90deg-pol-intercepts.csv', delimiter=',')

# Have user select directory containg polarization data for each incident light, and polarizer position
messagebox.showinfo('Polarization Subset', 'Select Root Directory Containing Polarized Imagery Sub-Directories')
polar_directory_path = get_dir_path()

# Create results folder in data directory
path=polar_directory_path+'/results'
if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)

# List all sub directories in selected directory
sub_directory_paths = os.listdir(polar_directory_path)

# Have user select mask image containing pixels desired for analysis
messagebox.showinfo('Polarization Subset', 'Select Large Mask Image')
mask_path = get_envi_hdr_file()

# Have user select smaller mask image, since there's a shadow
messagebox.showinfo('Polarization Subset', 'Select Small Mask Image')
mask_path25 = get_envi_hdr_file()

#set empty lists
P=[]
alpha=[]
theta_i=[]

# For each sub directory loop through, find the directory containing polarization data oriented at 0deg first
# Then find matching 90deg file, compute ratio and crop out masked parts. Append to list of polarization and
# phase angle datasets from above
for sub_dir in sub_directory_paths:
    if 'polAng90' in sub_dir:
        for sample_file in os.listdir(os.path.join(polar_directory_path, sub_dir)):
            if sample_file.endswith('.hdr'):

                #for normal images
                polar_mask = envi.open(mask_path)
                polar_mask_data = polar_mask.load()
                polar_mask_data[polar_mask_data == 0] = np.nan

                #for shadowed ones
                polar_mask25 = envi.open(mask_path25)
                polar_mask_data25 = polar_mask25.load()
                polar_mask_data25[polar_mask_data25 == 0] = np.nan

                # Polarizer transmission in axis labeled on polarizer.
                # Ie: 90 degrees perpendicular 0 parallel
                per_path = os.path.join(polar_directory_path, sub_dir, sample_file)

                per_image = envi.open(per_path)
                per_image_data = per_image.load()

                wavelength = per_image.bands.centers

                par_path = per_path.replace('polAng90', 'polAng0')
                par_image = envi.open(par_path)
                par_image_data = par_image.load()


                if par_image_data.size != per_image_data.size:
                    if per_image_data.size > par_image_data.size:
                        per_image_data = per_image_data[0:par_image_data.shape[0],
                                            0:par_image_data.shape[1],
                                            0:par_image_data.shape[2]]
                    else:
                        par_image_data = par_image_data[0:per_image_data.shape[0],
                                               0:per_image_data.shape[1],
                                               0:per_image_data.shape[2]]

                # calculate phase angle for each pair of files
                tmp_parts = par_path.split('_')

                light_angle = float(tmp_parts[6][5:])  # note: this depends on location of light source angle in filename

                frame_time, frame_value = get_frame_data(per_path)
                motion_time, motion_position = get_motion_data(per_path, frame_time)
                camera_angle = calc_motion_position(frame_time, frame_value, motion_time, motion_position)

                phase_angle = calc_phase_angle(0, 90 - light_angle, 0, 90 - abs(camera_angle))

                scan_info=np.genfromtxt(polar_directory_path+'/'+sub_dir+'/'+'motion.txt',usecols=2,skip_header=2)

                scan_start=np.min(abs(scan_info))
                scan_end=np.max(abs(scan_info))
                diff=int(scan_end-scan_start)

                #convert dat from DN's to radiance (calibrate)
                for i in range(len(per_image_data)):
                    per_image_data[i]=np.multiply(per_image_data[i],slope_per.T).reshape((1600,371))+int_per.T
                    par_image_data[i] = np.multiply(par_image_data[i], slope_par.T).reshape((1600,371)) + int_par.T


                #calculate polarization for each pair of files
                polar = (per_image_data - par_image_data) / (per_image_data + par_image_data)

                if light_angle>=120:
                    polar = polar[:polar_mask_data25.shape[0], :polar_mask_data25.shape[1], :] * polar_mask_data25[:polar.shape[0]]
                else:
                    polar = polar[:polar_mask_data.shape[0], :polar_mask_data.shape[1], :] * polar_mask_data[:polar.shape[0]]

                #crop out masked polarization values and append list
                pol = polar[:, ~np.isnan(polar).all((0, 2)), :]
                pol = pol[~np.isnan(pol).all((1, 2)), :, :]
                P.append(pol[:-1, :, ~np.isnan(pol).all((0, 1))])

                #crop out masked phase angle values and append list
                x=[]
                for i in range(polar.shape[0]):
                    for j in range(polar.shape[1]):
                        if ~np.isnan(polar[i,j,0]):
                            x.append(i) #indices of non-masked polarization values in dimension corresponding to phase angle

                x=np.array(x)
                alpha.append(phase_angle[int(x.min()):int(x.max())])
                theta_i.append(light_angle)


######## Post Processing and modeling ################

lam=np.array(wavelength)
th_i=np.array(theta_i)

#calculate mean, standard deviation, smooth raw and averaged data
P_mean=[]
P_std=[]
P_smooth=[]
P_center=[]
P_center1=[]
P_avg=[]
alpha1=[]
P_center3=[]
alpha3=[]
for i in range(len(P)):
    P_mean.append(np.mean(P[i],axis=1))
    P_center.append(P[i][:,int(P[i].shape[1]/2)])
    P_std.append(np.std(P[i], axis=1))
    P_smooth.append(ss.savgol_filter(np.float64(P[i]),11,5,axis=0))
    P_center1.append(P[i][int(P[i].shape[0] / 2), int(P[i].shape[1] / 2),:])
    alpha1.append(alpha[i][int(P[i].shape[0] / 2)])
    P_center3.append(P[i][int(P[i].shape[0] / 2-1):int(P[i].shape[0] / 2+2), int(P[i].shape[1] / 2-1):int(P[i].shape[1] / 2+2),:])
    alpha3.append(alpha[i][int(P[i].shape[0] / 2-1):int(P[i].shape[0] / 2+2)])

P_c3=[]
P_std3=[]
for i in range(len(P_mean)):
    P_avg.append(np.mean(P_mean[i],axis=0))
    P_c3.append(np.mean(P_center3[i],axis=(0,1)))
    P_std3.append(np.std(P_center3[i], axis=(0, 1)))

#convert lists to arrays and stack list elements
P_mean=np.concatenate(P_mean,axis=0)
P_center=np.concatenate(P_center,axis=0)
P_std=np.concatenate(P_std,axis=0)
P_ms=ss.savgol_filter(np.float64(P_mean),11,5,axis=0)
P_cs=ss.savgol_filter(np.float64(P_center),11,5,axis=0)
P_center1=np.array(P_center1)
alpha1=np.array(alpha1)
P_center3=np.array(P_center3)
alpha3=np.array(alpha3)
P_avg=np.array(P_avg)
P_c3=np.array(P_c3)
P_std3=np.array(P_std3)

# g1=np.concatenate(alpha[4:8],axis=0)
# g2=np.concatenate((alpha[:4],alpha[8:]),axis=0)
# g2=g2.flatten()
# p1=np.concatenate(P[4:8],axis=0)
# p2=np.concatenate((P[:4],P[8:]),axis=0)
# p2=p2.reshape((p2.shape[0]*p2.shape[1],p2.shape[2],p2.shape[3]))
# ps1=np.concatenate(P_smooth[4:8],axis=0)
# ps2=np.concatenate((P_smooth[:4],P_smooth[8:]),axis=0)
# ps2=ps2.reshape((ps2.shape[0]*ps2.shape[1],ps2.shape[2],ps2.shape[3]))
alpha=np.concatenate(alpha,axis=0)

Pmin=np.zeros(lam.shape)
Pmax=np.zeros(lam.shape)
gmax=np.zeros(lam.shape)
gmin=np.zeros(lam.shape)
for j in range(len(lam)):
    for i in range(len(alpha)):
        if P_mean[i, j] == np.max(P_mean[:, j]):
            Pmax[j] = P_mean[i, j]
            gmax[j] = alpha[i]

        elif P_mean[i, j] == np.min(P_mean[:, j]):
            Pmin[j] = P_mean[i, j]
            gmin[j] = alpha[i]

# for j in range(len(lam)):
#     for i in range(len(alpha)):
#         if P_mean[i, j] == np.max(P_center[:, j]):
#             Pmax[j] = P_center[i, j]
#             gmax[j] = alpha[i]
#
#         elif P_mean[i, j] == np.min(P_center[:, j]):
#             Pmin[j] = P_center[i, j]
#             gmin[j] = alpha[i]

#polyfits - averaged
x=alpha
f_avg=np.poly1d(np.polyfit(x,P_mean[:,144],5))
f_savg=np.poly1d(np.polyfit(x,P_ms[:,144],5))
x_new=np.linspace(0,110,5000)
y_avg=f_avg(x_new)
y_savg=f_savg(x_new)
_, res, _, _, _ = np.polyfit(x,P_mean[:,144],5, full=True)
rmse=np.sqrt(res)[0]

#polyfits - averaged
x2=alpha
f_avg2=np.poly1d(np.polyfit(x,P_center[:,144],5))
f_savg2=np.poly1d(np.polyfit(x,P_cs[:,144],5))
x_new2=np.linspace(0,110,5000)
y_avg2=f_avg2(x_new2)
y_savg2=f_savg2(x_new2)

#save to csv files
folder='D:/Polarization_Results/csv_files_Whiteref'
if not os.path.exists(folder):
    os.mkdir(folder)
polar_mean_out = pd.DataFrame(data=P_mean, columns=lam)
phase_out = pd.DataFrame(data=alpha, columns=['phase_angle'])
output_csv_file = pd.concat([phase_out, polar_mean_out], axis=1)
output_csv_file.to_csv(folder +'/'+ str(tmp_parts[4][:-3])+'_Pmean.csv')

polar_ms_out = pd.DataFrame(data=P_ms, columns=lam)
output_csv_file = pd.concat([phase_out, polar_ms_out], axis=1)
output_csv_file.to_csv(folder +'/'+ str(tmp_parts[4][:-3])+ '_PmeanSmooth.csv')

polar_min_out = pd.DataFrame(data=Pmin, columns=['Pmin'])
phase_min_out = pd.DataFrame(data=gmin, columns=['gmin'])
polar_max_out = pd.DataFrame(data=Pmax, columns=['Pmax'])
phase_max_out = pd.DataFrame(data=gmax, columns=['gmax'])
lam_out = pd.DataFrame(data=lam, columns=['wavelength'])
output_csv_file = pd.concat([lam_out,phase_min_out, polar_min_out], axis=1)
output_csv_file.to_csv(folder +'/'+ str(tmp_parts[4][:-3])+ '_Pmin.csv')
output_csv_file = pd.concat([lam_out,phase_max_out, polar_max_out], axis=1)
output_csv_file.to_csv(folder +'/'+ str(tmp_parts[4][:-3])+ '_Pmax.csv')

#processing for possible shift to correct shading in light angle 25 data
# t1=P[0]
# t2=P[1]
# t3=par_image_data[:,:,144]
# shift1=(t2[:,:,144].shape[0]-t1[:,:,144].shape[0])/t3.shape[0]
# shift2=(t2[:,:,144].size-t1[:,:,144].size)/t3.size
# P_mean25_shift1=np.mean(P[0],axis=1)*(1-shift1)
# P_mean25_shift2=np.mean(P[0],axis=1)*(1-shift2)

########### Plot results and save in a results folder to data directory ###################
# p=plt.figure()
# plt.plot(g1,p1[:,:,144],'b.',g2,p2[:,:,144],'b.')
# plt.title('Raw Polarization '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.show()
# p.savefig(os.path.join(path,'raw_data_633nm.png'))
#
# p=plt.figure()
# plt.plot(g1,ps1[:,:,144],'r.',g2,ps2[:,:,144],'r.')
# plt.title('Smoothened Polarization '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.show()
# p.savefig(os.path.join(path,'smooth_data_633nm.png'))

p=plt.figure()
# plt.plot(alpha,P_mean[:,144],'b.',g1,P_mean25_shift1[:,144],'r.',g1,P_mean25_shift2[:,144],'g.')
# note: above line is just for sept collect
plt.plot(alpha,P_mean[:,144],'b.')
plt.title('Mean Polarization 633nm '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
plt.ylim(0.1, .25)
plt.ylabel('Polarization')
plt.xlabel('Phase Angle (deg)')
# plt.gca().legend(('raw averaged data','scaled by % obscured lines','scaled by % obscured pixels'))
# note: above line is just for sept collect
plt.show()
p.savefig(os.path.join(path,'averaged_data_633nm.png'))

# p=plt.figure()
# plt.plot(alpha,P_center[:,144],'b.')
# plt.title('Line Center Polarization 633nm '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.show()
# p.savefig(os.path.join(path,'lineCenter_data_633nm.png'))
# #
# p=plt.figure()
# plt.plot(alpha,P_std[:,144],'.')
# plt.title('Stdev Polarization 633nm '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.show()
# p.savefig(os.path.join(path,'stdev_633nm.png'))
#
# p=plt.figure()
# plt.plot(alpha,P_ms[:,144],'.')
# plt.title('Smoothened Mean Polarization 633nm '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.show()
# p.savefig(os.path.join(path,'avg_plus_smooth_data_633nm.png'))
# #
# p=plt.figure()
# plt.plot(alpha,P_cs[:,144],'.')
# plt.title('Smoothened Line Center Polarization 633nm '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.show()
# p.savefig(os.path.join(path,'lineCenter_smooth_data_633nm.png'))

p=plt.figure()
plt.plot(x_new,y_avg, label='RMSE='+str(round(rmse,4)))
plt.title('5th Order Polyfit Averaged Polarization 633nm \n'+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
plt.ylabel('Polarization')
plt.xlabel('Phase Angle (deg)')
plt.ylim(0.1, .25)
plt.legend()
plt.show()
p.savefig(os.path.join(path,'polyfit_avg_633nm.png'))

# p=plt.figure()
# plt.plot(x_new,y_avg)
# plt.title('5th Order Polyfit Averaged Polarization 633nm \n'+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.show()
# p.savefig(os.path.join(path,'polyfit_avg_633nm2.png'))
#
# p=plt.figure()
# plt.plot(x_new2,y_avg2,'r',x_new2,y_savg2,'b')
# plt.title('5th Order Polyfit Center Line Polarization 633nm '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.gca().legend(('Line Center fit','smoothened fit'))
# plt.show()
# p.savefig(os.path.join(path,'polyfits_center_633nm.png'))
#
fig4,ax4=plt.subplots(1,2)
ax4[0].plot(lam,Pmin,'b')
ax4[0].set_ylabel('Polarization Minimum  ')
ax4[0].set_xlabel('wavelength(nm)')
ax4[1].scatter(lam,gmin,c='r',s=3)
ax4[1].set_ylabel('Phase angle of Polarization Minimum')
ax4[1].set_xlabel('wavelength(nm)')
plt.suptitle('P_min and g_min vs band '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"), y=1.0, ha='center')
plt.tight_layout(pad=4)
plt.show()
fig4.savefig(os.path.join(path,'wavelength_min_effects.png'))

fig4,ax4=plt.subplots(1,2)
ax4[0].plot(lam,Pmax,'b')
ax4[0].set_ylabel('Polarization Maximum  ')
ax4[0].set_xlabel('wavelength(nm)')
ax4[1].scatter(lam,gmax,c='r',s=3)
ax4[1].set_ylabel('Phase angle of Polarization Maximum')
ax4[1].set_xlabel('wavelength(nm)')
plt.suptitle('P_max and g_max vs band '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"), y=1.0, ha='center')
plt.tight_layout(pad=4)
plt.show()
fig4.savefig(os.path.join(path,'wavelength_max_effects.png'))

# p=plt.figure()
# plt.plot(alpha1,P_center1[:,144],'b.')
# plt.title('Center pixel Polarization 633nm '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.show()
# p.savefig(os.path.join(path,'CenterPixel_data_633nm.png'))
#
# p=plt.figure()
# plt.plot(alpha3,P_center3[:,:,0,144],'b.',alpha3,P_center3[:,:,1,144],'b.',alpha3,P_center3[:,:,2,144],'b.')
# plt.title('Center window Polarization 633nm '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.show()
# p.savefig(os.path.join(path,'CenterWindow_data_633nm.png'))
#
# p=plt.figure()
# plt.plot(alpha1,P_c3[:,16],'b.',alpha1,P_c3[:,144],'g.',alpha1,P_c3[:,278],'r.')
# plt.title('Center window Mean Polarization '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.gca().legend(('425nm','633nm','850nm'))
# plt.show()
# p.savefig(os.path.join(path,'CenterWindowAvg.png'))
#
# p=plt.figure()
# plt.plot(alpha1,P_std3[:,16],'b',alpha1,P_std3[:,144],'g',alpha1,P_std3[:,278],'r')
# plt.title('Stdev of Center window Mean Polarization '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.gca().legend(('425nm','633nm','850nm'))
# plt.show()
# p.savefig(os.path.join(path,'CenterWindowStdev.png'))
#
# p=plt.figure()
# plt.plot(th_i,P_center1[:,16],'b.',th_i,P_center1[:,144],'g.',th_i,P_center1[:,278],'r.')
# plt.title('Center pixel Polarization '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Light Angle (deg)')
# plt.gca().legend(('425nm','633nm','850nm'))
# plt.show()
# p.savefig(os.path.join(path,'incidence_angle_center.png'))
#
# p=plt.figure()
# plt.plot(th_i,P_avg[:,16],'b.',th_i,P_avg[:,144],'g.',th_i,P_avg[:,278],'r.')
# plt.title('Average pixel Polarization '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Light Angle (deg)')
# plt.gca().legend(('425nm','633nm','850nm'))
# plt.show()
# p.savefig(os.path.join(path,'incidence_angle_avg.png'))
#
# p=plt.figure()
# plt.plot(th_i,P_c3[:,16],'b.',th_i,P_c3[:,144],'g.',th_i,P_c3[:,278],'r.')
# plt.title('Center window Mean Polarization '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Light Angle (deg)')
# plt.gca().legend(('425nm','633nm','850nm'))
# plt.show()
# p.savefig(os.path.join(path,'CenterWindowAvg_incidence.png'))
#
# p=plt.figure()
# plt.plot(alpha,P_mean[:,16],'r.',alpha,P_mean[:,144],'g.',alpha,P_mean[:,278],'b.')
# plt.title('Mean Polarization '+' '+str(tmp_parts[2])+' '+str(tmp_parts[4][8:-10])+str(u"\u03bcm"))
# plt.ylabel('P, ratio out of 1')
# plt.xlabel('Phase Angle (deg)')
# plt.gca().legend(('425nm','633nm','850nm'))
# plt.show()
# p.savefig(os.path.join(path,'averaged_Pol_3lam.png'))