# Imports
import numpy as np
#from numpy.linalg import inv
import scipy as sp
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET

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
P_init = np.array([[1, 0],[0, 1]])



# state space model (x_k = f(x_(k-1) + process noise)
# x = (a,a')
# a = angle, a' = angular velocity
# a_k  = a_(k-1) + a'_(k-1) * ts
# a'_k = a'_(k-1) - g sin (a_(k-1)) *ts


# for EKF I ll use standart KF with linearized state space model and update it at every step
# F = df/dx|_(x_k-1) Jakobian derived from f at predicted x_k-1
# F = (       1           , ts)
#     (-g*ts*cos(a_(k-1)) , 1 )
# s is cos of predited value of a_(k-1) from previous step
s = np.cos(x_init[0])
F = np.array([[1, ts], [-g*ts*s, 1]])

#Process noise
Q = np.array([[q*pow(ts,3)/3, q*pow(ts,2)/2], [q*pow(ts, 2)/2, q*ts]])

#measurement
# m = sin (a) + r
# r = observation noise with variance from dataset at given time
# Here we also linearize the mesurement model
# H will be the Jakobian for measurment transformation
# H = (cos(a) , 0)^T
H = np.array([s, 0])


# Kalman filter
# Working variables in filter
# x_apri and x_post, P_apri and P_post (a priory and posteriori)
# x apri (P_apri) = value of x (P) from previous step
# x post, predicted value of x (P) from system mechanics
# x_next, P_next, estimated value of x (P) given by kalman filter
# filling them if empti sets to set correct datatype
x_apri = x_init
x_post = np.array([0, 0])
x_next = np.array([0, 0])
P_apri = P_init
P_post = np.array([[1, 0],[0, 1]])
P_next = np.array([[1, 0],[0, 1]])

# K_gain - Kalman gain

for i in range(101):
    # preparing linearized system for evolution of covariance
    s = np.cos(x_apri[0])
    F = np.array([[1, ts],[-g*ts*s, 1]])
    # measurment noise prom data
    r = float(DataSV[i][2].text)
    R = r
    

    # prediction step
    # based on evolution of x using system state equations
    # linearized evolution of covariance
    x_post = np.array([x_apri[0] + x_apri[1]*ts, x_apri[1] - g*ts*np.sin(x_apri[0])])
    P_post = F.dot(P_apri).dot(F.T) + Q

    # update step
    # updating Jakobian of mesurement predistion
    s = np.cos(x_post[0])
    H = np.array([s, 0])
    # Kalman gain computation
    K = (P_post.dot(H.T)) /(H.dot(P_post).dot(H.T) + R)
    # Measurements m and predicted measurement h
    m = float(DataSV[i][1].text)
    h = np.sin(x_post[0])

    # Estimation step
    # Calculating estimated vaues of x and P with Kalman gain and measurements
    x_next = x_post + K.dot(m-h)
    P_next = (np.array([[1, 0], [0, 1]]) - K.dot(H)).dot(P_post)

    # Print, save and assign results
    print('time = ', round(i*ts,3), '    estimated angle = ', round(x_next[0],3), '    estimated angular velocity =', round(x_next[1],3),'    estimated displacement = ', round(np.sin(x_post[0]),3), '    measured displacement = ',round(m,3))
    x_apri = x_next
    P_apri = P_next




