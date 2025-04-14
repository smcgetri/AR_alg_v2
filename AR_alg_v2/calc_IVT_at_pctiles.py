"""
Calculate IVT at the climatological percentile rank values specified in the
configuration file.

The percentile ranks are calculated from the distribution of all values in a
31-day centered window at each analysis grid cell for the times specified by
the IVT_climatology_start_year, IVT_climatology_end_year, and
IVT_climatology_timestep_hrs in the configuration file.
"""

import argparse
import glob
import hjson
import datetime as dt
from datetime import timedelta
import xarray as xr
import numpy as np
import pandas as pd
import ast
import cftime
from pathlib import Path

def calc_IVT_at_pctiles(AR_config, start_year, end_year, timestep_hrs, start_doy, end_doy):
    """
    Calculate IVT values at a range of percentiles for each julian day of the year.
    """
    #original code used IVT_dir
    IVT_path = AR_config['IVT_path']
    ds = xr.open_dataset(IVT_path)

    #Detect calendar type
    time_data = ds.time.data
    use_cftime = isinstance(time_data[0], cftime.Datetime360Day)
    if use_cftime:
        print("Detected 360-day calendar.")
    else:
        print("Standard calendar detected.")
        ds = ds.assign_coords(time=pd.to_datetime(time_data))

    # Time filtering using attribute-based logic
    ds = ds.where(
        (ds.time.dt.year >= start_year) & 
        (ds.time.dt.year <= end_year) & 
        (ds.time.dt.hour % int(timestep_hrs) == 0),
        drop=True
    )

    # Spatial subsetting
    lats = np.arange(AR_config['min_lat'], AR_config['max_lat'] + AR_config['lat_res'], AR_config['lat_res'])
    ds = ds.sel(lat=lats, method='nearest')

    # Adjust DOY
    leap_years = np.arange(1900, end_year + 1, 4)
    doys_fixed = []
    for t in ds.time.values:
        if use_cftime:
            doy = t.dayofyr
        else:
            doy = t.timetuple().tm_yday
        if (t.year in leap_years) and doy >= 60:
            doys_fixed.append(doy - 1)
        else:
            doys_fixed.append(doy)
    ds['doy'] = ('time', doys_fixed)

    doys = np.arange(int(start_doy), int(end_doy) + 1)
    pctiles = np.array(ast.literal_eval(str(AR_config['IVT_percentiles_to_calc'])))
    IVT_data = ds['ivt_magnitude']

    output = np.empty((len(doys), len(pctiles), ds.lat.shape[0], ds.lon.shape[0]))

    for i, doy in enumerate(doys):
                # "Dummy" time will only be used to get doy of window start and end
        # - For dummy time, use a year where neither the year before or after
        #   is a leap year
        t_dummy = dt.datetime(1902, 1, 1) + timedelta(days=int(doy) - 1)
        win_start = (t_dummy - timedelta(days=15)).timetuple().tm_yday
        win_end = (t_dummy + timedelta(days=15)).timetuple().tm_yday
        
        # Handle the case where window starts at the end of one year and
        # continues into the next
        if win_start > win_end:
            window_doys = np.concatenate((np.arange(win_start, 366), np.arange(1, win_end + 1)))
        else:
            window_doys = np.arange(win_start, win_end + 1)

        ds_window = IVT_data.sel(time=ds.time.dt.dayofyear.isin(window_doys))
        q = ds_window.chunk(dict(time=-1)).quantile(pctiles / 100, dim='time', skipna=True)
        output[i] = q.to_numpy()

        print(f'Finished doy: {doy} at '+str(dt.datetime.now()))

    return doys, pctiles, ds.lat.data, ds.lon.data, output

def write_output(AR_config, start_year, end_year, timestep_hrs, doys, pctiles, lats, lons, data):
        """
    Write IVT at percentiles output file with metadata supplied by the AR ID
    configuration.
    """
    ds_out = xr.Dataset(coords=dict(doy=doys, lat=lats, lon=lons))
    for i, p in enumerate(pctiles):
        ds_out[f'IVT_pctile_{p}'] = (['doy', 'lat', 'lon'], data[:, i, :, :])
        ds_out[f'IVT_pctile_{p}'].attrs['units'] = 'kg/m/s'
    ds_out.lat.attrs['units'] = 'degrees_north'
    ds_out.lon.attrs['units'] = 'degrees_east'

    ds_out.attrs.update({
        'data_source': AR_config['data_source'],
        'IVT_data_origin': AR_config['IVT_data_origin'],
        'IVT_vert_coord': AR_config['IVT_vert_coord'],
        'IVT_climatology_start_year': start_year,
        'IVT_climatology_end_year': end_year,
        'IVT_climatology_timestep_hrs': timestep_hrs
    })

    fname = f"IVT_at_pctiles_{AR_config['data_source']}_{start_year}_{end_year}.nc"
    out_path = Path(AR_config['IVT_at_pctiles_dir']) / fname
    ds_out.to_netcdf(out_path)
    print(f"Output saved to {out_path}")

def parse_args():
    """
    Parse arguments passed to script at runtime.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'start_doy', 
        type=int, 
        help='Day of year (1-365) at which to start calculations'
    )
    parser.add_argument(
        'end_doy', 
        type=int, 
        help='Day of year (1-365) at which to end calculations'
    )
    parser.add_argument(
        'AR_ID_config_path', 
        help='Path to AR ID configuration file'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.AR_ID_config_path) as f:
        AR_config = hjson.loads(f.read())

    start_year = AR_config['IVT_climatology_start_year']
    end_year = AR_config['IVT_climatology_end_year']
    timestep_hrs = AR_config['IVT_climatology_timestep_hrs']

    doys, pctiles, lats, lons, data = calc_IVT_at_pctiles(
        AR_config, start_year, end_year, timestep_hrs, args.start_doy, args.end_doy
    )

    write_output(AR_config, start_year, end_year, timestep_hrs, doys, pctiles, lats, lons, data)

if __name__ == '__main__':
    main()
