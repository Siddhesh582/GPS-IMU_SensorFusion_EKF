import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np

def compute_R(gps_x, gps_y, gps_timestamps, gps_status):
    # compute GPS measurement noise from actual data
    valid = (~np.isnan(gps_x)) & (~np.isnan(gps_y)) & (gps_status >= 0)
    gx, gy, gt = gps_x[valid], gps_y[valid], gps_timestamps[valid]

    dx = np.diff(gx)
    dy = np.diff(gy)
    dt = np.diff(gt)

    # velocities from consecutive GPS positions
    vx = dx / dt
    vy = dy / dt

    # remove outliers > 50 m/s
    v_mask = (np.abs(vx) < 50) & (np.abs(vy) < 50)

    # std of position differences = measurement noise
    sigma_x = np.std(dx[v_mask])
    sigma_y = np.std(dy[v_mask])

    R_xx = sigma_x**2
    R_yy = sigma_y**2

    print(f"GPS noise — sigma_x: {sigma_x:.3f}m  sigma_y: {sigma_y:.3f}m")
    print(f"R = diag([{R_xx:.3f}, {R_yy:.3f}])")
    return R_xx, R_yy

def measurement_model(R_xx, R_yy):
    # H selects x and y from state — GPS only measures position
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ])
    R = np.array([
        [R_xx, 0   ],
        [0,    R_yy]
    ])
    return H, R

def correction_step(state_est, P_est, gps_x, gps_y, R_xx, R_yy):
    H, R  = measurement_model(R_xx, R_yy)

    z     = np.array([gps_x, gps_y])
    z_est = H @ state_est           # predicted GPS from current state
    resid = z - z_est               # residual — correction signal

    S = H @ P_est @ H.T + R         # uncertainty in residual
    K = P_est @ H.T @ np.linalg.pinv(S)  # Kalman gain

    state_corr    = state_est + K @ resid
    state_corr[2] = np.arctan2(np.sin(state_corr[2]), np.cos(state_corr[2]))  # normalize heading

    I      = np.eye(6)
    P_corr = (I - K @ H) @ P_est @ (I - K @ H).T + K @ R @ K.T  # Joseph form
    return state_corr, P_corr, resid