# Imports
import numpy as np
from numpy.linalg import inv
import scipy as sp
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET
import pandas as pd

# Importing data into code
# DV - dataset with Different variance
# SV - dataset with Constatn variance
# OUT - dataset containing outliers
treeDV = ET.parse('measurements_nooutlier_differentvar.xml')
DataDV = treeDV.getroot()
treeSV = ET.parse('measurements_nooutlier_samevar.xml')
DataSV = treeSV.getroot()
treeOUT = ET.parse('measurements_outlier_samevar.xml')
DataOUT = treeOUT.getroot()

#Empty array to be used for saving and graphing extimated data

# Constants
# Gravitational constant
g = 9.81
# Spectral density
q = 0.1
# time step (constant in this case)
ts = 0.05


# Definition of initial state and initial covariance
# Measurement dataSV and dataDV sugests that the pendulum is at maximal displacement so arcsin(1) was chosen as the starting point
x_init = np.array([np.arcsin(1), 0])
P_init = np.eye(2)

#selecting dataset
Data = DataSV


# state space model (x_k = f(x_(k-1) + process noise)
# x = (a,a')
# a = angle, a' = angular velocity
# a_k  = a_(k-1) + a'_(k-1) * ts
# a'_k = a'_(k-1) - g sin (a_(k-1)) *ts
# Defining evolution of x
def fx(x):
    return np.array([x[0] + x[1]*ts, x[1] - g*ts*np.sin(x[0])])

# for EKF I ll use standart KF with linearized state space model and update it at every step
# F = df/dx|_(x_k-1) Jakobian derived from f at predicted x_k-1
# F = (       1           , ts)
#     (-g*ts*cos(a_(k-1)) , 1 )
# s is cos of predited value of a_(k-1) from previous step
def Fx(x):
    s = np.cos(float(x[0]))
    return np.array([[1, ts], [-g*ts*s, 1]])

#Process noise
Q = np.array([[q*pow(ts,3)/3, q*pow(ts,2)/2], [q*pow(ts, 2)/2, q*ts]])

#measurement
# m = sin (a) + r
# r = observation noise with variance from dataset at given time
# defining merurement model
def hx(x):
    return np.sin(x[0])

# Here we also linearize the mesurement model
# H will be the Jakobian for measurment transformation
# H = (cos(a) , 0)^T
def Hx (x):
    s = np.cos(float(x[0]))
    return np.array([[s, 0]])


# Kalman filter
# Working variables in filter
# x_apri and x_post, P_apri and P_post (a priory and posteriori)
# x apri (P_apri) = value of x (P) from previous step
# x post, predicted value of x (P) from system mechanics
# x_next, P_next, estimated value of x (P) given by kalman filter
# filling them with empty sets to set correct datatype
x_apri = x_init
x_post = np.array([0, 0])
x_next = np.array([0, 0])
P_apri = P_init
P_post = np.eye(2)
P_next = np.eye(2)

for i in range(101):
    # preparing linearized system for evolution of covariance
    F = Fx(x_init)
    # measurment noise prom data
    r = float(Data[i][2].text)
    R = np.array([[r]])
    

    # prediction step
    # based on evolution of x using system state equations
    # linearized evolution of covariance
    x_post = fx(x_apri)
    P_post = F.dot(P_apri).dot(F.T) + Q

    # update step
    # updating Jakobian of mesurement predistion

    H = Hx(x_post)
    # Kalman gain computation
    K = (P_post.dot(H.T)) @ np.linalg.inv(H.dot(P_post).dot(H.T) + R)
    # Measurements m and predicted measurement h
    m = float(Data[i][1].text)
    h = hx(x_post)

    # Estimation step
    # Calculating estimated vaues of x and P with Kalman gain and measurements
    x_next = x_post + K.dot(m-h)[0]
    P_next = (np.eye(2) - K.dot(H)).dot(P_post)

    # Print, save and assign results
    print('time = ', round(i*ts,3), '    estimated angle = ', round(x_next[0],3), '    estimated angular velocity =', round(x_next[1],3),'    estimated displacement = ', round(np.sin(x_post[0]),3), '    measured displacement = ',round(m,3))
    x_apri = x_next
    P_apri = P_next

