import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from utils import normalize_angle

def process_covariance(dt, sigma_vx=0.2, sigma_vy=0.2, sigma_omega=0.01):
    # process noise — models uncertainty from constant velocity assumption
    q_vx    = sigma_vx**2
    q_vy    = sigma_vy**2
    q_omega = sigma_omega**2

    # off-diagonals couple velocity uncertainty into position uncertainty via dt
    Q = np.array([
        [q_vx * dt**2,  0,             0,               q_vx * dt,  0,          0          ],
        [0,             q_vy * dt**2,  0,               0,          q_vy * dt,  0          ],
        [0,             0,             q_omega * dt**2, 0,          0,          q_omega*dt ],
        [q_vx * dt,     0,             0,               q_vx,       0,          0          ],
        [0,             q_vy * dt,     0,               0,          q_vy,       0          ],
        [0,             0,             q_omega * dt,    0,          0,          q_omega    ]
    ])
    return Q

def process_model(state, imu_gyro_bias_corrected, dt, theta_k):
    x, y, theta, vx, vy, omega = state

    omega_imu = imu_gyro_bias_corrected[2]  # yaw rate from gyroscope

    # constant velocity model — position integrates velocity
    x_est     = x + vx * dt
    y_est     = y + vy * dt
    theta_est = theta + omega_imu * dt      # heading integrates gyro
    vx_est    = vx                          # velocity held constant
    vy_est    = vy                          # GPS correction handles changes
    omega_est = omega_imu

    theta_est = normalize_angle(theta_est)  # keep theta in [-pi, pi]

    state_predicted = np.array([x_est, y_est, theta_est, vx_est, vy_est, omega_est])

    # state transition matrix — how each state at k affects state at k+1
    F = np.array([
        [1, 0, 0, dt, 0,  0],   # x(k+1) = x + vx*dt
        [0, 1, 0, 0,  dt, 0],   # y(k+1) = y + vy*dt
        [0, 0, 1, 0,  0,  dt],  # theta(k+1) = theta + omega*dt
        [0, 0, 0, 1,  0,  0],   # vx constant
        [0, 0, 0, 0,  1,  0],   # vy constant
        [0, 0, 0, 0,  0,  1]    # omega from gyro
    ])
    return state_predicted, F

def prediction_step(state, P, imu_gyro_corrected, dt):
    theta_current = state[2]
    state_pred, F = process_model(state, imu_gyro_corrected, dt, theta_current)
    Q             = process_covariance(dt)
    P_pred        = F @ P @ F.T + Q   # propagate uncertainty + add process noise
    return state_pred, P_pred