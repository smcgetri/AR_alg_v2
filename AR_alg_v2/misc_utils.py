"""
Miscellaneous utility functions for AR algorithm.
"""

import numpy as np
import datetime as dt
from cftime import Datetime360Day, num2date, date2num

def rename_coords(ds):
    """
    Make sure latitude and longitude in xarray dataset are named "lat" and "lon",
    so that these coordinate names can be used to subset data.
    
    Edit for CANARI to include "lat_um_atmos_grid_uv" and "lon_um_atmos_grid_uv"
    
    This function also changes very small (but nonzero) values of lat/lon to
    a value of 0 (using the helper function _fix_zero_values).
    """
    
    if ('lat' in ds.coords) and ('lon' in ds.coords):
        pass
    elif ('latitude' in ds.coords) and ('longitude' in ds.coords):
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    elif ('lat_um_atmos_grid_uv' in ds.coords) and ('lon_um_atmos_grid_uv' in ds.coords):
        ds = ds.rename({'lat_um_atmos_grid_uv': 'lat', 'lon_um_atmos_grid_uv': 'lon'})
    else:
        raise Exception('Unknown lat/lon coordinate names')

    ds = _fix_zero_values(ds)
    
    return ds



def _fix_zero_values(ds):
    """
    Change very small (but nonzero) values of lat/lon to a value of 0.
    """
    
    lats = ds.lat.data
    lons = ds.lon.data
    
    lats[np.where(np.abs(lats) < 0.0001)] = 0
    lons[np.where(np.abs(lons) < 0.0001)] = 0
    
    ds['lat'] = lats
    ds['lon'] = lons
    
    return ds


def rename_IVT_components(ds):
    """
    Make sure u- and v-IVT components in xarray dataset are named "uIVT" and "vIVT".
    
    (ERA5 pre-calculated uIVT and vIVT are named "p71.162" and "p72.162". Most
    other datasets will have variables named "uIVT" and "vIVT" as needed for
    AR ID script, because these variables are calculated "in house" by calc_IVT.py
    rather than being provided in the original dataset.)
    
    Edit for CANARI data to include viwve and viwvn
    """
    if 'p71.162' in ds and 'p72.162' in ds:
        ds = ds.rename({'p71.162': 'uIVT', 'p72.162': 'vIVT'})
    elif 'IVTx' in ds and 'IVTy' in ds:
        ds = ds.rename({'IVTx': 'uIVT', 'IVTy': 'vIVT'})
    elif 'viwve' in ds and 'viwvn' in ds:
        ds = ds.rename({'viwve': 'uIVT', 'viwvn': 'vIVT'})
        
    return ds

def generate_cftime_range(start_str, end_str, step_hours):
    """
    Converting 360 day datetime format for CANARI data
    """
    start_dt = dt.datetime.strptime(start_str, "%Y-%m-%d_%H%M")
    end_dt = dt.datetime.strptime(end_str, "%Y-%m-%d_%H%M")
    step = dt.timedelta(hours=int(step_hours))

    # Convert to Datetime360Day
    start_cf = Datetime360Day(start_dt.year, start_dt.month, min(start_dt.day, 30),
                              start_dt.hour, start_dt.minute)
    end_cf = Datetime360Day(end_dt.year, end_dt.month, min(end_dt.day, 30),
                            end_dt.hour, end_dt.minute)

    result = []
    current = start_cf

    while current <= end_cf:
        result.append(current)
        # Advance using date2num logic (step in days)
        next_num = date2num(current, 'hours since 2000-01-01 00:00:00') + step.total_seconds() / 3600
        current = num2date(next_num, 'hours since 2000-01-01 00:00:00', calendar='360_day')

    return result