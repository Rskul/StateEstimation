import numpy as np

# Constatns of symple pendulum system
g = 9.81  # Gravitational constant
q = 0.1  # Spectral density
ts = 0.05  # time step (constant in this case)

# Functons for calculation of KF of simple pendulum system
# x = State of system
# x is vector with (a, a') where a is angle and a' is angular velocity
# P = System covariances


def KF_xEvol(x):
    """
    Calculate Time evolution of simple pendulum system.

    Inputs system state x.
    Outputs state x after time ts has passed.

    Time evolution according following mathematical model:

    state space model (x_k = f(x_(k-1) + process noise)
    x = (a,a')
    a = angle, a' = angular velocity
    a_k  = a_(k-1) + a'_(k-1) * ts
    a'_k = a'_(k-1) - g sin (a_(k-1)) *ts.

    both input and output x are vectors (a, a') where a is angle and a' is angular velocity.
    """
    return np.array([x[0] + x[1]*ts, x[1] - g*ts*np.sin(x[0])])


def EKF_PTransitionMatrix(x):
    """
    Calculate linearized transition matrix.

    Inputs is state of system x.
    Outputs linearized transition matrix time evlution of the system by time ts.

    Linearisation obtained from jacobian of time evolution of system.

    F = df/dx|_(x_k-1) Jakobian derived from f at predicted x_k-1
    F = (       1           , ts)
        (-g*ts*cos(a_(k-1)) , 1 )
    s is cos of predited value of a_(k-1) from previous step

    Input is vector x.
    Outputs 2x2 matrix.
    """
    s = np.cos(float(x[0]))
    return np.array([[1, ts], [-g*ts*s, 1]])


def KF_MeasurementTransformation(x):
    """
    Calculate displacement from the angle of pendulum.

    Inputs state of system x.
    Outputs value of displacement.

    measurement:
    m = sin(a) + r
    r = observation noise with variance from dataset at given time.

    Input is vector x.
    Output is scalar value.
    """
    return np.sin(x[0])


def EKF_LinearizedMeasurementTransformation(x):
    """
    Calculate Linearized calculation of displacement from system state x.

    Inputs state of system x.
    Outputs transtition matrix that transforms system state values to displacement.

    Linearization is obtained by calculation of Jakobian. 

    H = (cos(a)
        (  0  )

    Input is vector x.
    Output is 2x1 matrix.
    """

    s = np.cos(float(x[0]))
    return np.array([[s, 0]])


def EKF_KalmanGain(x, P, R):
    """
    Calculate Kalman gain

    Inputs system state x, state covariance matrix P and measurement covariance matrix R
    Outputs Kallman gain

    K = PH^T / (H^TPH + R)

    Where H is linearised measurement transformation matrix
    """

    H = EKF_LinearizedMeasurementTransformation(x)
    return (P.dot(H.T)) @ np.linalg.inv((H).dot(P).dot(H.T) + R)

