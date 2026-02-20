import sys
sys.path.append('/home/sid/GPS-IMU_SensorFusion/src')

import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils           import gps_to_utm, get_valid_gps_mask
from bias_estimation import estimate_bias, apply_bias, plot_bias_correction
from kalman_filter   import run

# CONFIG
DATA_PATH = '/home/sid/GPS-IMU_SensorFusion/data/sensor_data.pkl'

# LOAD DATA
print("Loading raw sensor data...")
with open(DATA_PATH, 'rb') as f:
    raw = pickle.load(f)

gps_timestamps = raw['gps']['timestamps']
gps_lat        = raw['gps']['latitude']
gps_lon        = raw['gps']['longitude']
gps_alt        = raw['gps']['altitude']
gps_status     = raw['gps']['status']
imu_timestamps = raw['imu']['timestamps']
imu_accel      = raw['imu']['accel']
imu_gyro       = raw['imu']['gyro']

duration_sec = gps_timestamps[-1] - gps_timestamps[0]
print(f"GPS  : {len(gps_timestamps):,} samples @ {len(gps_timestamps)/duration_sec:.1f} Hz")
print(f"IMU  : {len(imu_timestamps):,} samples @ {len(imu_timestamps)/duration_sec:.1f} Hz")
print(f"Duration : {duration_sec/60:.2f} minutes")

# GPS — UTM CONVERSION + DROPOUT ANALYSIS
# GPS is in lat/lon (spherical) — convert to UTM (flat, meters) for EKF
gps_x, gps_y, zone = gps_to_utm(gps_lat, gps_lon)
print(f"UTM Zone: {zone}")

nan_count   = np.sum(np.isnan(gps_lat))
first_valid = np.where(~np.isnan(gps_lat))[0][0]
print(f"GPS dropout  : {nan_count} samples ({first_valid/10:.1f} seconds)")
print(f"First valid GPS at index {first_valid} — {first_valid/10:.1f}s into recording")

valid_mask = get_valid_gps_mask(gps_x, gps_y, gps_status)

# GPS trajectories — lat/lon and UTM
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(gps_lon, gps_lat, 'b.', ms=1, alpha=0.5)
axes[0].set_title('GPS Trajectory (Lat/Lon)')
axes[0].set_xlabel('Longitude (deg)')
axes[0].set_ylabel('Latitude (deg)')

axes[1].plot(gps_x[valid_mask], gps_y[valid_mask], 'b.', ms=1, alpha=0.5)
axes[1].set_title(f'GPS Trajectory (UTM {zone})')
axes[1].set_xlabel('Easting (m)')
axes[1].set_ylabel('Northing (m)')
axes[1].axis('equal')
plt.tight_layout()
plt.savefig('../results/figures/gps_trajectory.png', dpi=150)
plt.show()

# IMU — VISUALIZE + BIAS REMOVAL
# Raw IMU plots
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axes[0].plot(imu_timestamps, imu_accel[:, 0], 'r', lw=0.3, label='X')
axes[0].plot(imu_timestamps, imu_accel[:, 1], 'g', lw=0.3, label='Y')
axes[0].plot(imu_timestamps, imu_accel[:, 2], 'b', lw=0.3, label='Z')
axes[0].set_ylabel('Accel (m/s²)')
axes[0].legend()

axes[1].plot(imu_timestamps, imu_gyro[:, 0], 'r', lw=0.3, label='Roll rate')
axes[1].plot(imu_timestamps, imu_gyro[:, 1], 'g', lw=0.3, label='Pitch rate')
axes[1].plot(imu_timestamps, imu_gyro[:, 2], 'b', lw=0.3, label='Yaw rate')
axes[1].set_ylabel('Angular Vel (rad/s)')
axes[1].legend()

accel_mag = np.linalg.norm(imu_accel, axis=1)
axes[2].plot(imu_timestamps, accel_mag, 'purple', lw=0.3)
axes[2].axhline(9.81, color='r', ls='--', alpha=0.5, label='Gravity (9.81)')
axes[2].set_ylabel('|Accel| (m/s²)')
axes[2].set_xlabel('Time (s)')
axes[2].legend()
plt.tight_layout()
plt.savefig('../results/figures/imu_raw.png', dpi=150)
plt.show()

# estimate bias from first 10s — vehicle stationary, GPS not yet acquired
accel_bias, gyro_bias = estimate_bias(imu_accel, imu_gyro, imu_timestamps, stationary_duration=10.0)
imu_accel_corr, imu_gyro_corr = apply_bias(imu_accel, imu_gyro, accel_bias, gyro_bias)

# plot raw vs corrected over stationary window — X,Y should collapse to ~0
plot_bias_correction(imu_accel, imu_gyro, imu_timestamps,
                     imu_accel_corr, imu_gyro_corr, stationary_duration=10.0)

# verify bias removal — corrected accel should be ~[0, 0, 0] during stationary window
mask = imu_timestamps < (imu_timestamps[0] + 10.0)
print("Corrected accel mean:", np.mean(imu_accel_corr[mask], axis=0))
print("Corrected accel std :", np.std(imu_accel_corr[mask],  axis=0))

# RUN EKF
states = run(gps_x, gps_y, gps_timestamps, gps_status, imu_gyro_corr, imu_timestamps)

# EXTRACT RESULTS
last_valid_time = gps_timestamps[valid_mask][-1]
imu_mask        = (imu_timestamps <= last_valid_time) & (~np.isnan(states[:, 0]))

kf_x     = states[imu_mask, 0]
kf_y     = states[imu_mask, 1]
kf_theta = states[imu_mask, 2]
kf_vx    = states[imu_mask, 3]
kf_vy    = states[imu_mask, 4]
t        = imu_timestamps[imu_mask]

# relative coordinates from first valid GPS fix
ref_x     = gps_x[np.where(valid_mask)[0][0]]
ref_y     = gps_y[np.where(valid_mask)[0][0]]
gps_x_rel = gps_x - ref_x
gps_y_rel = gps_y - ref_y
kf_x_rel  = kf_x  - ref_x
kf_y_rel  = kf_y  - ref_y

# METRICS
# interpolate GPS to IMU timestamps for error computation
gps_x_interp = np.interp(t, gps_timestamps[valid_mask], gps_x[valid_mask])
gps_y_interp = np.interp(t, gps_timestamps[valid_mask], gps_y[valid_mask])
kf_error     = np.sqrt((kf_x - gps_x_interp)**2 + (kf_y - gps_y_interp)**2)
kf_speed     = np.sqrt(kf_vx**2 + kf_vy**2)

print(f"EKF PERFORMANCE METRICS")
print(f"Mean error  : {np.mean(kf_error):.3f} m")
print(f"RMS error   : {np.sqrt(np.mean(kf_error**2)):.3f} m")
print(f"Max error   : {np.max(kf_error):.2f} m")
print(f"Mean speed  : {np.mean(kf_speed):.2f} m/s ({np.mean(kf_speed)*3.6:.1f} km/h)")
print(f"Duration    : {t[-1]-t[0]:.1f} s")
print(f"Distance    : {np.sum(np.sqrt(np.diff(kf_x)**2 + np.diff(kf_y)**2)):.1f} m")

# PLOTS
# EKF vs GPS trajectory
plt.figure(figsize=(12, 8))
plt.plot(gps_x_rel, gps_y_rel, 'b.', ms=2, alpha=0.4, label='GPS Raw')
plt.plot(kf_x_rel,  kf_y_rel,  'r-', lw=1.5,          label='EKF Fused')
plt.plot(0, 0, 'go', ms=10, label='Start')
plt.plot(kf_x_rel[-1], kf_y_rel[-1], 'rs', ms=10, label='End')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('EKF vs GPS Trajectory')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/ekf_vs_gps_trajectory.png', dpi=150)
plt.show()

# X, Y position and heading over time
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax1.plot(gps_timestamps, gps_x_rel, 'b.', ms=1, alpha=0.4, label='GPS')
ax1.plot(t, kf_x_rel, 'r-', lw=1, label='EKF')
ax1.set_ylabel('X Position (m)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(gps_timestamps, gps_y_rel, 'b.', ms=1, alpha=0.4, label='GPS')
ax2.plot(t, kf_y_rel, 'r-', lw=1, label='EKF')
ax2.set_ylabel('Y Position (m)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# GPS heading from consecutive position differences
gps_dx   = np.diff(gps_x[valid_mask])
gps_dy   = np.diff(gps_y[valid_mask])
gps_hdg  = np.degrees(np.unwrap(np.arctan2(gps_dy, gps_dx)))
gps_t_hd = (gps_timestamps[valid_mask][:-1] + gps_timestamps[valid_mask][1:]) / 2

ax3.plot(gps_t_hd, gps_hdg, 'b.', ms=1, alpha=0.4, label='GPS')
ax3.plot(t, np.degrees(np.unwrap(kf_theta)), 'r-', lw=1, label='EKF')
ax3.set_ylabel('Heading (deg)')
ax3.set_xlabel('Time (s)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/figures/position_heading.png', dpi=150)
plt.show()