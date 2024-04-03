# Imports
import numpy as np
import scipy as sp
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET

# Importing data into code
# DV - dataset with Different variance
# SV - dataset with Constatn variance
# OUT - dataset containing outliers
treeDV = ET.parse('measurements_nooutlier_differentvar.xml')
DataDV = treeDV.getroot()
treeSV = ET.parse('measurements_nooutlier_differentvar.xml')
DataSV = treeSV.getroot()
treeOUT = ET.parse('measurements_nooutlier_differentvar.xml')
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
x_init = np.array([0,0])
P_init = np.array ([1,0],[0,1])



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
F = np.array([1,ts],[-g*ts*s,1])

#Process noise
Q = np.array([q*pow(ts,3)/3,q*pow(ts,2)/2],[q*pow(ts,2)/2,q*ts])

#measurement
# m = sin (a) + r
# r = observation noise with variance from dataset at given time
# Here we also linearize the mesurement model
# H will be the Jakobian for measurment transformation
# H = (cos(a) , 0)^T
H = np.array([s,0])


# Kalman filter
# Working variables in filter
# x_apri and x_post, P_apri and P_post (a priory and posteriori)
# x apri (P_apri) = value of x (P) from previous step
# x post, predicted value of x (P) from system mechanics
# x_next, P_next, estimated value of x (P) given by kalman filter
# filling them if empti sets to set correct datatype
x_apri = x_init
x_post = np.array([0,0])
x_next = np.array([0,0])
P_apri = P_init
P_post = np.array([1,0],[0,1])
P_next = np.array([1,0],[0,1])

# K_gain - Kalman gain

for i in range(100):
    # preparing linearised system for current step
    s = np.cos(x_apri[0])
    F = np.array([1,ts],[-g*ts*s,1])
    H = np.array([s,0])
    # measurment noise prom data
    R = float(DataSV[i][2].text)
    

    # prediction
    # based on evolution of x using system state equations
    # linearized evolution of covariance
    x_post = np.array([x_apri[0] + x_apri[1]*ts, x_apri[1] -g*ts*np.sin(x_apri[0])])
    P_post = F.dot(P_apri).dot(F.T) + Q

    # update
    # Kalman gain computation
    K = P_post.dot(H.T)*pow(H.dot(P_post).dot(H.T) + R,-1)
    # Measurements m and predicted measurement h
    m = float(DataSV[i][1].text)
    h = H.dot(x_post)
    # Calculating estimated vaues of x and P with Kalman gain and measurements
    x_next = x_post + K*(m-h)
    P_next = (np.array([1,0],[0,1]) - K.dot(H)).dot(P_post)

    # Print, save and assign results
    print('time = ', i*ts, ',estimated angle = ', x_next[0], 'estimated angular velocity =', x_next[1], 'estimated displacement = ', H.dot(x_next))




