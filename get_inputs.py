import tkinter as tk
from tkinter import filedialog

import math

from sklearn import linear_model

import os

from datetime import datetime
import numpy as np

root = tk.Tk()
root.withdraw()


def get_dir_path():

    root.update()
    directory_path = filedialog.askdirectory(title='Select HSI Directory')
    root.withdraw()

    return directory_path


def get_envi_hdr_file():

    root.update()
    hdr_file_path = filedialog.askopenfilename(filetypes=(("hdr files", "*.hdr"), ("all files", "*.*")),
                                               title='Select "*.hdr" file')
    root.withdraw()

    return hdr_file_path


def get_frame_data(hdr_file_path):

    dir_path = os.path.split(hdr_file_path)
    file_number = dir_path[1].split('_')[1]
    with open(dir_path[0] + "/" + 'frameIndex_' + file_number[:-4] + '.txt', 'r') as f:
        data_frame = f.readlines()

    data_frame.remove(data_frame[0])

    frame_value = []
    frame_time = []
    frame_timestamp = []
    for i in range(len(data_frame)):
        tmp_data = data_frame[i].split()
        frame_value.append(tmp_data[0])
        frame_time.append(datetime.strptime(' '.join([tmp_data[1], tmp_data[2]]), '%H:%M:%S %f'))
        #frame_timestamp.append(frame_time[i].timestamp())
    frame_value = np.asarray(frame_value).astype(float)
    frame_time = np.asarray(frame_time)

    return frame_time, frame_value


def get_motion_data(hdr_file_path, frame_time):

    dir_path = os.path.split(hdr_file_path)

    with open(dir_path[0] + "/" + 'motion.txt', 'r') as f:
        data_motion = f.readlines()

    data_motion.remove(data_motion[0])

    motion_time = []
    position = []
    counts = []
    micro1 = []
    micro2 = []
    target = []
    for i in range(len(data_motion)):
        tmp_data = data_motion[i].split()
        motion_time.append(datetime.strptime(' '.join([tmp_data[0], tmp_data[1]]), '%H:%M:%S %f'))
        position.append(tmp_data[2])
        counts.append(tmp_data[3])
        micro1.append(tmp_data[4])
        micro2.append(tmp_data[5])
        target.append(tmp_data[6])
    motion_time = np.asarray(motion_time)
    motion_position = np.asarray(position).astype(float)

    motion_position_out = []
    motion_time_out = []
    for i in range(len(frame_time)):
        nearest = min(motion_time, key=lambda x: abs(x - frame_time[i]))
        index_value = (np.where(motion_time == nearest))
        motion_position_out.append(motion_position[index_value])
        motion_time_out.append(motion_time[index_value])

    motion_position_out = np.asarray(motion_position_out).astype(float)
    motion_time_out = np.asarray(motion_time_out)

    return motion_time_out, motion_position_out


def calc_motion_position(frame_time, frame_value, motion_time, motion_position):

    motion_angle = []
    for i in range(len(frame_time)):
        nearest = min(motion_time, key=lambda x: abs(x - frame_time[i]))
        index_value = (np.where(motion_time == nearest))
        motion_angle.append(motion_position[index_value[0]])

    motion_angle = np.asarray(motion_angle)
    motion_angle2 = [float(numeric_string[0]) for numeric_string in motion_angle]
    motion_angle2 = np.asarray(motion_angle2)

    model = linear_model.LinearRegression()
    model.fit(frame_value.reshape((-1, 1)), motion_angle2)

    position_pred = model.predict(frame_value.reshape((-1, 1)))

    return position_pred


def calc_phase_angle(solar_azimuth, solar_zenith, sensor_azimuth, sensor_zenith):

    phi = (solar_azimuth - sensor_azimuth) * (math.pi/180)

    phase_angle = np.arccos((np.cos(solar_zenith*(math.pi/180))*np.cos(sensor_zenith*(math.pi/180))) +
                            np.sin(solar_zenith*(math.pi/180)) *
                            np.sin(sensor_zenith*(math.pi/180)) *
                            np.cos(phi))*(180/math.pi)

    return phase_angle
