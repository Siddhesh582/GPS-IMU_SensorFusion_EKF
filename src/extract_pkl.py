"""
Extract GPS and IMU data from a ROS2 bag into a pkl file.
Run once:  python src/extract_bag.py
"""

import pickle
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# Config
DB3_PATH = '/home/sid/GPS-IMU_SensorFusion/data/2023-10-19-14-14-38-filtered/2023-10-19-14-14-38-filtered_ros2/2023-10-19-14-14-38-filtered_ros2.db3'
OUT_PATH  = 'data/sensor_data.pkl'
GPS_TOPIC = '/gps/fix'
IMU_TOPIC = '/imu/imu_uncompensated'

def get_reader(bag_path):
    storage_opts = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id='sqlite3'
    )
    converter_opts = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_opts, converter_opts)
    return reader

def extract(bag_path, gps_topic, imu_topic):
    reader   = get_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    print("Topics in bag:")
    for name, typ in type_map.items():
        print(f"  {name}  [{typ}]")

    gps_data = {'timestamps': [], 'latitude': [], 'longitude': [],
                'altitude':   [], 'status':    []}
    imu_data = {'timestamps': [], 'accel':     [], 'gyro':      [],
                'orientation': []}

    gps_count = 0
    imu_count = 0

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t_sec = t_ns * 1e-9

        if topic == gps_topic:
            msg = deserialize_message(data, get_message(type_map[topic]))
            gps_data['timestamps'].append(t_sec)
            gps_data['latitude'].append(msg.latitude)
            gps_data['longitude'].append(msg.longitude)
            gps_data['altitude'].append(msg.altitude)
            gps_data['status'].append(msg.status.status)  # -1=no fix, 0=fix
            gps_count += 1

        elif topic == imu_topic:
            msg = deserialize_message(data, get_message(type_map[topic]))
            imu_data['timestamps'].append(t_sec)
            imu_data['accel'].append([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            imu_data['gyro'].append([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            imu_data['orientation'].append([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])
            imu_count += 1

    # Warn if nothing was extracted — likely wrong topic names
    if gps_count == 0:
        print(f"\nWARNING: No GPS messages found on '{gps_topic}'")
        print("Check topic names printed above and update GPS_TOPIC in CONFIG")
    if imu_count == 0:
        print(f"\nWARNING: No IMU messages found on '{imu_topic}'")
        print("Check topic names printed above and update IMU_TOPIC in CONFIG")
    if gps_count == 0 or imu_count == 0:
        return

    # Convert to numpy
    for k in gps_data:
        gps_data[k] = np.array(gps_data[k])
    for k in imu_data:
        imu_data[k] = np.array(imu_data[k])

    raw = {'gps': gps_data, 'imu': imu_data}

    with open(OUT_PATH, 'wb') as f:
        pickle.dump(raw, f)

    duration = gps_data['timestamps'][-1] - gps_data['timestamps'][0]
    print(f"\nExtracted:")
    print(f"  GPS : {gps_count:,} samples @ {gps_count/duration:.1f} Hz")
    print(f"  IMU : {imu_count:,} samples @ {imu_count/duration:.1f} Hz")
    print(f"  Duration : {duration/60:.2f} minutes")
    print(f"  Saved to : {OUT_PATH}")

if __name__ == '__main__':
    extract(DB3_PATH, GPS_TOPIC, IMU_TOPIC)