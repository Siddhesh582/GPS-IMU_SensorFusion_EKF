import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from utils import get_valid_gps_mask, normalize_angle
from process_model import prediction_step
from measurement_model import correction_step, compute_R

def initialize(gps_x, gps_y, gps_timestamps, gps_status):
    # initialize state from first two valid GPS fixes
    valid_idx = np.where(get_valid_gps_mask(gps_x, gps_y, gps_status))[0]
    if len(valid_idx) < 2:
        raise ValueError("Need at least 2 valid GPS fixes to initialize")

    i0, i1 = valid_idx[0], valid_idx[1]
    x0, y0 = gps_x[i0], gps_y[i0]
    dx      = gps_x[i1] - gps_x[i0]
    dy      = gps_y[i1] - gps_y[i0]
    dt      = gps_timestamps[i1] - gps_timestamps[i0]

    theta0  = np.arctan2(dy, dx)    # heading from direction between fixes
    vx0     = dx / dt               # velocity from position difference
    vy0     = dy / dt

    state0  = np.array([x0, y0, theta0, vx0, vy0, 0.0])
    P0      = np.diag([0.0025, 0.0025, 0.1, 1.0, 1.0, 0.1])  # small pos uncertainty (GPS), large vel uncertainty

    print(f"Init position : ({x0:.2f}, {y0:.2f}) m")
    print(f"Init heading  : {np.degrees(theta0):.2f} deg")
    print(f"Init velocity : ({vx0:.2f}, {vy0:.2f}) m/s")
    return state0, P0, valid_idx[0]

def run(gps_x, gps_y, gps_timestamps, gps_status, imu_gyro_corrected, imu_timestamps):
    state, P, first_valid_gps_idx = initialize(gps_x, gps_y, gps_timestamps, gps_status)
    valid_mask = get_valid_gps_mask(gps_x, gps_y, gps_status)
    R_xx, R_yy = compute_R(gps_x, gps_y, gps_timestamps, gps_status)

    # start filter from first valid GPS timestamp
    start_time    = gps_timestamps[first_valid_gps_idx]
    start_imu_idx = np.searchsorted(imu_timestamps, start_time)

    n      = len(imu_timestamps)
    states = np.full((n, 6), np.nan)
    states[start_imu_idx] = state

    corrections = 0
    for k in range(start_imu_idx + 1, n):
        dt = imu_timestamps[k] - imu_timestamps[k-1]
        if dt <= 0 or dt > 1.0:
            dt = 0.01   # fallback for bad timestamps

        # reset P if it explodes during GPS outage — prevents singular matrix
        if np.any(np.diag(P) > 1e6):
            P = np.diag([1.0, 1.0, 0.5, 4.0, 4.0, 0.5])

        # PREDICT — IMU drives state forward at 200Hz
        state_pred, P_pred = prediction_step(state, P, imu_gyro_corrected[k], dt)

        # CORRECT — GPS correction when within 20ms sync window
        time_diff       = np.abs(gps_timestamps - imu_timestamps[k])
        closest_gps_idx = np.argmin(time_diff)

        if valid_mask[closest_gps_idx] and time_diff[closest_gps_idx] < 0.02:
            state, P, _ = correction_step(
                state_pred, P_pred,
                gps_x[closest_gps_idx],
                gps_y[closest_gps_idx],
                R_xx, R_yy
            )
            corrections += 1
        else:
            state, P = state_pred, P_pred   # no GPS — pure prediction

        states[k] = state

        if k % 50000 == 0:
            print(f"  {k:,}/{n:,} ({100*k/n:.1f}%) — corrections so far: {corrections}")

    print(f"Done — total GPS corrections: {corrections:,} ({100*corrections/np.sum(valid_mask):.1f}%)")
    return states