import numpy as np
import utm

def gps_to_utm(lat, lon):
    """Convert GPS lat/lon to UTM Easting/Northing in meters."""
    gps_x = np.full_like(lat, np.nan)
    gps_y = np.full_like(lon, np.nan)
    zone_info = None
    for i, (la, lo) in enumerate(zip(lat, lon)):
        if not (np.isnan(la) or np.isnan(lo)):
            e, n, zone_num, zone_letter = utm.from_latlon(la, lo)
            gps_x[i] = e
            gps_y[i] = n
            if zone_info is None:
                zone_info = f"{zone_num}{zone_letter}"
    return gps_x, gps_y, zone_info

def normalize_angle(theta):
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(theta), np.cos(theta))

def get_valid_gps_mask(gps_x, gps_y, gps_status):
    """Boolean mask for valid GPS readings."""
    return (~np.isnan(gps_x)) & (~np.isnan(gps_y)) & (gps_status >= 0)