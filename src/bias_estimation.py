import numpy as np
import matplotlib.pyplot as plt

def estimate_bias(imu_accel, imu_gyro, imu_timestamps, stationary_duration=10.0):
    # average first 10 seconds — vehicle stationary, GPS not yet acquired
    mask       = imu_timestamps < (imu_timestamps[0] + stationary_duration)
    accel_bias = np.mean(imu_accel[mask], axis=0)
    gyro_bias  = np.mean(imu_gyro[mask],  axis=0)
    accel_bias[2] -= 9.81   # remove gravity from Z
    print(f"Accel bias: {accel_bias}")
    print(f"Gyro  bias: {gyro_bias}")
    return accel_bias, gyro_bias

def apply_bias(imu_accel, imu_gyro, accel_bias, gyro_bias):
    return imu_accel - accel_bias, imu_gyro - gyro_bias

def plot_bias_correction(imu_accel, imu_gyro, imu_timestamps,
                         imu_accel_corr, imu_gyro_corr,
                         stationary_duration=10.0):
    # show 10s window before and after bias correction
    start_time = imu_timestamps[0]
    mask       = imu_timestamps <= start_time + stationary_duration
    t          = imu_timestamps[mask] - start_time   # offset to start at 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'IMU Bias Correction (First {stationary_duration:.0f} Seconds)', fontsize=13)

    # --- RAW ACCEL ---
    axes[0, 0].plot(t, imu_accel[mask, 0], 'r-', lw=0.6, alpha=0.8, label='X')
    axes[0, 0].plot(t, imu_accel[mask, 1], 'g-', lw=0.6, alpha=0.8, label='Y')
    axes[0, 0].plot(t, imu_accel[mask, 2], 'b-', lw=0.6, alpha=0.8, label='Z')
    axes[0, 0].set_title('Accel — Raw')
    axes[0, 0].set_ylabel('Acceleration (m/s²)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # --- CORRECTED ACCEL ---
    axes[0, 1].plot(t, imu_accel_corr[mask, 0], 'r-', lw=0.6, alpha=0.8, label='X')
    axes[0, 1].plot(t, imu_accel_corr[mask, 1], 'g-', lw=0.6, alpha=0.8, label='Y')
    axes[0, 1].plot(t, imu_accel_corr[mask, 2], 'b-', lw=0.6, alpha=0.8, label='Z')
    axes[0, 1].axhline(0, color='k', ls='--', alpha=0.4, label='Zero')
    axes[0, 1].set_title('Accel — Bias Corrected')
    axes[0, 1].set_ylabel('Acceleration (m/s²)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # --- RAW GYRO ---
    axes[1, 0].plot(t, imu_gyro[mask, 0], 'r-', lw=0.6, alpha=0.8, label='Roll rate')
    axes[1, 0].plot(t, imu_gyro[mask, 1], 'g-', lw=0.6, alpha=0.8, label='Pitch rate')
    axes[1, 0].plot(t, imu_gyro[mask, 2], 'b-', lw=0.6, alpha=0.8, label='Yaw rate')
    axes[1, 0].set_title('Gyro — Raw')
    axes[1, 0].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # --- CORRECTED GYRO ---
    axes[1, 1].plot(t, imu_gyro_corr[mask, 0], 'r-', lw=0.6, alpha=0.8, label='Roll rate')
    axes[1, 1].plot(t, imu_gyro_corr[mask, 1], 'g-', lw=0.6, alpha=0.8, label='Pitch rate')
    axes[1, 1].plot(t, imu_gyro_corr[mask, 2], 'b-', lw=0.6, alpha=0.8, label='Yaw rate')
    axes[1, 1].axhline(0, color='k', ls='--', alpha=0.4, label='Zero')
    axes[1, 1].set_title('Gyro — Bias Corrected')
    axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/figures/bias_correction.png', dpi=150)
    plt.show()