#Imports
import numpy as np
import scipy as sp
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET

#Importing data into code
#DV - dataset with Different variance
#SV - dataset with Constatn variance
#OUT - dataset containing outliers
treeDV = ET.parse('measurements_nooutlier_differentvar.xml')
DataDV = treeDV.getroot()
treeSV = ET.parse('measurements_nooutlier_differentvar.xml')
DataSV = treeSV.getroot()
treeOUT = ET.parse('measurements_nooutlier_differentvar.xml')
DataOUT = treeOUT.getroot()

a = float(DataSV[1][0].text)
print(a)

# Constants
# Gravitational constant
g = 9.81
# Spectral density
q = 0.1
# time step (constant in this case)
ts = 0.05


# Definition of initial state and initial covariance
x_init = np.array([0,0])
P = np.array ([1,0],[0,1])



#state space model
# a = angle, a' = angular velocity, x_ = at previous step
# a  = a_  + a'_ * ts
# a' = a'_ - g sin (a_) *ts
# for EKF I ll use standart KF with linearized state space model and update it at every step
# a' = a' -g k a_ *ts, where k = cos (a_)
# A = linearised transition matrix
k = np.cos(x_init[0])
A = np.array([1,ts],[-g*ts*k,1])

#Process noise
Q = np.array([q*pow(ts,3)/3,q*pow(ts,2)/2],[q*pow(ts,2)/2,q*ts])

#measurement
# m = sin (a) + r
# r = variance from dataset at given time
# Here we also linearise the mesurement model
# M = linearised observation model (function from a to m)

