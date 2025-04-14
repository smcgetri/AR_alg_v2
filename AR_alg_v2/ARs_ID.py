"""
Identify AR outlines by applying AR criteria to fields of IVT magnitude and
direction, and IVT at climatological percentile rank threshold.

Optionally, lower-tropospheric mean wind may be used to determine transport
direction instead of the IVT vector components.
"""

import os
import argparse
import hjson
import glob
import datetime as dt
import xarray as xr
import numpy as np
import pandas as pd
from pandas import DataFrame
from geopy.distance import great_circle
from scipy import ndimage
from skimage.measure import regionprops
import math
import itertools
import cftime 
import time
import psutil
from AR_alg_v2.misc_utils import rename_coords, rename_IVT_components, generate_cftime_range
from cftime import Datetime360Day
from scipy.ndimage import binary_erosion

def ARs_ID(AR_config, begin_time, end_time, timestep_hrs):
    """
    Main function to orchestrate AR identification for all times in the window
    defined by the begin and end times passed to the script.
    """

    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    # Verify that input data covers the entire zonal width of the earth
    check_input_zonal_extent(AR_config)
    
    # Calculate area of grid cells at each latitude of grid
    grid_cell_area_df = calc_grid_cell_areas_by_lat(AR_config)

    # Create latitude array for subset domain
    lats_subset = np.arange(AR_config['min_lat'], 
                            AR_config['max_lat']+AR_config['lat_res'], 
                            AR_config['lat_res'])

    # Create list of output data times using utils function for converting 360 day calendar
    times = generate_cftime_range(begin_time, end_time, timestep_hrs)

    # Build indexing data frame with input file names, times, and time indices
    ix_df = build_input_file_indexing_df(AR_config, times, ll_mean_wind=False)

    # Load the climatology percentiles file dynamically using glob
    pctile_files = glob.glob(os.path.join(
        AR_config['IVT_at_pctiles_dir'],
        f"IVT_at_pctiles_{AR_config['data_source']}*.nc"
    ))
    if not pctile_files:
        raise FileNotFoundError("No climatology percentile file found matching pattern.")

    IVT_at_pctiles_fpath = pctile_files[0]
    IVT_at_pctiles_ds = rename_coords(xr.open_dataset(IVT_at_pctiles_fpath))
    IVT_at_pctiles_ds = IVT_at_pctiles_ds.sel(lat=~IVT_at_pctiles_ds.get_index("lat").duplicated())
    IVT_at_pctiles_ds = IVT_at_pctiles_ds.sel(lat=lats_subset, method="nearest")

    lats = IVT_at_pctiles_ds.lat.data
    lons = IVT_at_pctiles_ds.lon.data
    climo_start_year = str(IVT_at_pctiles_ds.attrs.get('IVT_climatology_start_year', 'UNKNOWN'))
    climo_end_year = str(IVT_at_pctiles_ds.attrs.get('IVT_climatology_end_year', 'UNKNOWN'))
    climo_timestep_hrs = str(IVT_at_pctiles_ds.attrs.get('IVT_climatology_timestep_hrs', 'UNKNOWN'))

    AR_labels = np.zeros((len(times), len(lats), len(lons)), dtype=np.uint16)
    leap_years = np.arange(1900, 2100, 4)
    ivt_dataset_cache = {}

    for i, t in enumerate(times):
        t_str = f"{t.year:04d}-{t.month:02d}-{t.day:02d} {t.hour:02d}:{t.minute:02d}:{t.second:02d}"
        try:
            ix_df_t = ix_df.loc[t]
        except KeyError:
            print(f"Time {t} not found in ix_df index")
            continue

        # Calculate DOY for use in selecting percentile thresholds
        doy = (t.month - 1) * 30 + t.day  # assumes 360-day calendar
        if (t.year in leap_years) and (doy >= 60):
            doy -= 1

        IVT_at_pctiles_ds_doy = IVT_at_pctiles_ds.sel(doy=doy)

        # Build "wrap" arrays that span the width of the globe twice
        ll_mean_wind = AR_config['direction_filter_type'] == 'mean_wind_1000_700_hPa'
        label_array_prelim, IVT_wrap, IVT_at_thresh_wrap, u_wrap, v_wrap, lons_wrap = build_wrap_arrays(
            AR_config, lats_subset, ix_df_t, IVT_at_pctiles_ds_doy, ll_mean_wind, ivt_dataset_cache
        )
        # Filter potential AR features to a set of unique features that are not
        # duplicated across both "halves" of the "wrap" arrays
        label_array, feature_props_df = filter_duplicate_features(
            AR_config, label_array_prelim, IVT_wrap, lons_wrap, lats
        )

        label_array_2d = label_array[0] if label_array.ndim == 3 else label_array

        # Apply AR screening criteria to determine which features qualify as ARs,
        # and add to AR labels output array
        AR_labels_timestep, AR_count_timestep = apply_AR_criteria(
            AR_config, feature_props_df, grid_cell_area_df,
            label_array_2d, u_wrap, v_wrap, lats, lons, lons_wrap
        )
        # Renumber AR feature labels
        # - Either start at 1 and number features sequentially, or label all
        #   AR features as 1 (depending on setting in AR config)
        for j, label in enumerate(np.unique(AR_labels_timestep)[1:]):
            if AR_config['AR_labels'] == 'same_value':
                AR_labels_timestep[AR_labels_timestep == label] = 1
            elif AR_config['AR_labels'] == 'unique':
                AR_labels_timestep[AR_labels_timestep == label] = j + 1

        AR_labels[i, :AR_labels_timestep.shape[0], :AR_labels_timestep.shape[1]] = AR_labels_timestep

        now_str = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mem_mb = process.memory_info().rss / 1024**2
        elapsed = time.time() - start_time
        print(f"Processed {t_str} at {now_str} ({AR_count_timestep} ARs)")

    for ds in ivt_dataset_cache.values():
        ds.close()

    return AR_labels, times, lats, lons, climo_start_year, climo_end_year, climo_timestep_hrs



def check_input_zonal_extent(AR_config):
    """
    Verify that input data covers the entire zonal width of the earth.
    
    AR identification in narrower domains is currently not supported due
    to the "double wrapping" method used to handle antimeridian-crossing
    features.
    """
    
    # MERRA-2: for global data, min lon = -180 and max lon = 179.375
    # ERA5: for global data, min lon = -180 and max lon = 179.75
    # ** This would need to be adapted for coordinates not structured this way
    #   (e.g. if the first longitude is -180 and the last longitude is a
    #   "cyclic point" of 180)
    if ((AR_config['max_lon'] - AR_config['min_lon']) + AR_config['lon_res']) != 360:
        raise Exception('Input data must span the entire globe zonally')


def calc_grid_cell_areas_by_lat(AR_config):
    """
    Create data frame to look up grid cell areas based on latitude.
    """
    
    lat_res = AR_config['lat_res']
    lon_res = AR_config['lon_res']
    
    lats = []
    grid_cell_areas_km2 = []
    
    for lat in np.arange(AR_config['min_lat'], min(AR_config['max_lat'], 90) + lat_res, lat_res):
        if AR_config['hemisphere'] == 'NH':
            grid_cell_min_lat = lat - (lat_res/2)
            grid_cell_max_lat = min(lat + (lat_res/2), 90)  # Ensure it does not exceed 90
        elif AR_config['hemisphere'] == 'SH':
            grid_cell_max_lat = lat + (lat_res/2)
            grid_cell_min_lat = max(lat - (lat_res/2), -90)  # Ensure it does not go below -90

        # Calculate areas of sample "measurement" grid cells along the prime meridian
        grid_cell_min_lon = -(lon_res / 2)
        grid_cell_max_lon = (lon_res / 2)
        
        width_meas_pt_w = (lat, grid_cell_min_lon)
        width_meas_pt_e = (lat, grid_cell_max_lon)
        hgt_meas_pt_s = (grid_cell_min_lat, 0)
        hgt_meas_pt_n = (grid_cell_max_lat, 0)
        
        approx_grid_cell_width_m = great_circle(width_meas_pt_w, width_meas_pt_e).meters
        approx_grid_cell_hgt_m = great_circle(hgt_meas_pt_s, hgt_meas_pt_n).meters
        approx_grid_cell_area_m2 = approx_grid_cell_width_m * approx_grid_cell_hgt_m
        
        approx_grid_cell_area_km2 = approx_grid_cell_area_m2 * 1e-6
    
        lats.append(lat)
        grid_cell_areas_km2.append(approx_grid_cell_area_km2)
    
    grid_cell_area_df = DataFrame({'lat':lats, 'grid_cell_area_km2':grid_cell_areas_km2})

    return grid_cell_area_df



def build_input_file_indexing_df(AR_config, times, ll_mean_wind=False):
    """
    Build data frame with file path, time, and time index for each timestep of
    the input IVT files.
    
    If ll_mean_wind is True, then the indexing information for the input low-level
    mean wind files is also included in the data frame. Note that the time span
    and time indexing of the low-level wind files must be *exactly* the same as
    the IVT files.
    """
    
    IVT_fpaths = _sift_fpaths(AR_config, AR_config['IVT_dir'],
                              times[0], times[-1])
    IVT_fpaths_alltimes = []
    time_ixs = []
    file_times = []
    
    if ll_mean_wind:
        ll_mean_wind_fpaths_alltimes = []
    
    for fpath in IVT_fpaths:
        ds = xr.open_dataset(fpath)
        
        IVT_fpaths_alltimes.extend([fpath for i in range(len(ds.time))])
        time_ixs.extend(list(np.arange(0,len(ds.time),1)))
        if isinstance(ds.time.data[0], cftime.Datetime360Day):
            file_times.extend(ds.time.data)  # Keep CFTime format
        else:
            file_times.extend(pd.to_datetime(ds.time.data))  # Standard calendar
        

        if ll_mean_wind:
            ll_mean_wind_fpath = AR_config['wind_1000_700_mean_dir'] + \
                'mean_wind_1000_700_hPa' + \
                os.path.basename(fpath).split('IVT')[1]
            ll_mean_wind_fpaths_alltimes.extend([ll_mean_wind_fpath for i in range(len(ds.time))])
        
    ix_df = DataFrame({'IVT_fpath':IVT_fpaths_alltimes, 'time_ix':time_ixs, 't':file_times},
                      index=file_times)
    if ll_mean_wind:
        ix_df['ll_mean_wind_fpath'] = ll_mean_wind_fpaths_alltimes
    
    return ix_df
        
def _sift_fpaths(AR_config, data_dir, begin_dt, end_dt):
    """
    Adjusted helper function to locate IVT files based on ensemble and year. This has been changed from the original to reflect the data structure I am using. It is quite specific, so it will need to be updated if using different input files.
    """
    analysis_yrs = np.arange(begin_dt.year, end_dt.year + 1, 1)
    fpaths_all = []

    for ensemble in range(1, 2):  # Adjust range if needed #change
        for year in analysis_yrs:
            search_pattern = os.path.join(
                data_dir, f"ATM/yearly/{year}/ivt/ivt_tcwv_wspeed_ensemble{ensemble}_year{year}.nc"
            )
            matching_files = glob.glob(search_pattern)
            
            if matching_files:
                fpaths_all.extend(matching_files)
                print(f"Found IVT file: {matching_files[0]}")
            else:
                print(f"No IVT file found for ensemble {ensemble}, year {year}")

    return sorted(fpaths_all)


def build_wrap_arrays(AR_config, lats_subset, ix_df_t, IVT_at_pctiles_ds_doy, ll_mean_wind, ivt_dataset_cache):
    """
    Create "wrap arrays" that encircle the entire zonal width of the globe *twice*,
    so that features that cross the antimeridian and/or a pole can be handled
    as contiguous features by the image processing functions.
    
    Arrays created are:
    - label arrays of unique potential AR ID "features"
    - u/v arrays used for filtering final AR objects according to AR ID criteria
      (either u/v-IVT or low-level mean u/v-wind)
   
    edited to use cached open datasets to reduce file I/O overhead instead of opening the file for every timestep. Removed mean wind as not needed for this case
    """
    # Extract IVT file path from index
    if isinstance(ix_df_t, pd.Series):
        IVT_fpath = str(ix_df_t['IVT_fpath'])
    else:
        IVT_fpath = str(ix_df_t['IVT_fpath'].iloc[0])

    # Use cached dataset or open if not already in cache
    if IVT_fpath not in ivt_dataset_cache:
        ivt_dataset_cache[IVT_fpath] = xr.open_dataset(IVT_fpath, engine="netcdf4")

    dataset = ivt_dataset_cache[IVT_fpath]

    # Time slicing
    if isinstance(dataset.time.data[0], cftime.Datetime360Day):
        ix_df_t_times = xr.CFTimeIndex([ix_df_t.t])
    else:
        ix_df_t_times = pd.to_datetime(ix_df_t.t)

    IVT_ds = rename_IVT_components(rename_coords(dataset)).sel(
        time=ix_df_t_times, lat=lats_subset, method="nearest"
    )

    # Wrap longitude
    lons_wrap = np.concatenate((IVT_ds.lon, IVT_ds.lon))
    lons_wrap[np.where(np.abs(lons_wrap) < 0.0001)] = 0

    IVT = IVT_ds.ivt_magnitude
    IVT_wrap = np.concatenate((IVT, IVT), axis=2)

    IVT_at_thresh = IVT_at_pctiles_ds_doy['IVT_pctile_' + str(AR_config['IVT_PR_thresh'])]
    if IVT_at_thresh.shape[0] != IVT_wrap.shape[1]:
        IVT_at_thresh = IVT_at_thresh.sel(lat=lats_subset, method="nearest")
        if IVT_at_thresh.shape[0] != IVT_wrap.shape[1]:
            IVT_at_thresh = IVT_at_thresh.interp(lat=lats_subset, method="linear")

    IVT_at_thresh_expanded = np.expand_dims(IVT_at_thresh, axis=0)
    IVT_at_thresh_expanded = np.repeat(IVT_at_thresh_expanded, IVT_wrap.shape[0], axis=0)
    IVT_at_thresh_wrap = np.concatenate((IVT_at_thresh_expanded, IVT_at_thresh_expanded), axis=2)

    if IVT_at_thresh_wrap.shape != IVT_wrap.shape:
        IVT_at_thresh_wrap = np.broadcast_to(IVT_at_thresh_wrap, IVT_wrap.shape)

    thresh_array_wrap = np.where(
        (IVT_wrap >= AR_config['IVT_thresh']) & (IVT_wrap >= IVT_at_thresh_wrap),
        1, 0
    )
    label_array_prelim, _ = ndimage.label(thresh_array_wrap)

    if ll_mean_wind:
        raise NotImplementedError("mean_wind_1000_700_hPa not implemented in caching fix")
    else:
        if "uIVT" in IVT_ds and "vIVT" in IVT_ds:
            u_wrap = np.concatenate((IVT_ds.uIVT, IVT_ds.uIVT), axis=2)
            v_wrap = np.concatenate((IVT_ds.vIVT, IVT_ds.vIVT), axis=2)
        else:
            raise ValueError("uIVT and vIVT not found in dataset")

    return label_array_prelim, IVT_wrap, IVT_at_thresh_wrap, u_wrap, v_wrap, lons_wrap



def filter_duplicate_features(AR_config, label_array_prelim, IVT_wrap, lons_wrap, lats):
    """
    Apply two tests to ensure that potential AR features in "wrap array" are not
    duplicated across the two globe-encircling "halves" of the "wrap array":
    (1) Check if any of the labeled features wrap zonally around the entire hemisphere
        (common near the North Pole; not sure about South Pole). If so, change the
        label array values to 0 for the second "wrap" of the array so that the entire
        720-degree feature in the "double wrapped" label array is not labeled as
        one feature.
    (2) Check if mean IVT is exactly the same within any two features. (Indicating
        that the same feature is present in both "halves" of the "wrap" array spanning
        the hemisphere twice.)
        
    Return:
    - label_array with potential AR features that wrap zonally around the entire
      hemisphere filtered out
    - A data frame with attributes of potential AR features that are not
      duplicated across both "halves" of the "wrap array", and haven't yet been
      filtered into the *final* AR features by the direction, size, and shape criteria.
    """
    
    label_array, labels_prelim = _full_zonal_wraps_filter(AR_config,
                                                          label_array_prelim, lons_wrap, lats)
    feature_props_IVT = regionprops(label_array, intensity_image=IVT_wrap)
    # Create data frame containing info on each feature:
    # - feature label
    # - the feature's image processing "regionprops" object
    # - mean IVT within feature
    # - feature number of grid cells
    feature_props_df = DataFrame({
        'label':labels_prelim[1:],
        'feature_props_IVT':feature_props_IVT,
        'feature_mean_IVT':[feature.mean_intensity for feature in feature_props_IVT],
        'feature_num_grid_cells':[len(feature.coords) for feature in feature_props_IVT]
        })
    
    feature_props_df_IVT_filtered = _identical_IVT_filter(feature_props_df)    

    return label_array, feature_props_df_IVT_filtered

def _full_zonal_wraps_filter(AR_config, label_array_prelim, lons_wrap, lats):
    """
    Helper to filter_duplicate_features.
    
    Check if any of the labeled features wrap zonally around the entire hemisphere
    (common near the North Pole; not sure about South Pole). If so, change the
    label array values to 0 for the second "wrap" of the array so that the entire
    720-degree feature in the "double wrapped" label array is not labeled as
    one feature.

    Modified from original to support 3D arrays and include basic error handling.
    Size check removed as it's already done in apply_AR_criteria (need to confirm).
    """
    
    labels_prelim = np.unique(label_array_prelim)
    for label in labels_prelim:
        binary_array_label = np.where((label_array_prelim == label), 1, 0)


        for lat_ix in range(min(lats.shape[0], binary_array_label.shape[1])):  
            if lat_ix >= binary_array_label.shape[1]:  
                print(f"Skipping lat_ix={lat_ix}, out of range (max={binary_array_label.shape[1]-1})")
                continue

            if np.sum(binary_array_label[:, lat_ix, :]) == lons_wrap.shape[0]:  
                # Make label array in second wrap equal to 0
                label_array_replace_ixs_full = np.where(label_array_prelim == label)
                label_array_second_wrap_ixs = np.where(label_array_replace_ixs_full[2] >= int(lons_wrap.shape[0] / 2))

                label_array_replace_ixs_second_wrap = (
                    label_array_replace_ixs_full[0][label_array_second_wrap_ixs],  # Time indices
                    label_array_replace_ixs_full[1][label_array_second_wrap_ixs],  # Latitude indices
                    label_array_replace_ixs_full[2][label_array_second_wrap_ixs],  # Longitude indices
                )
                label_array_prelim[label_array_replace_ixs_second_wrap] = 0
                break

    return label_array_prelim, labels_prelim


def _identical_IVT_filter(feature_props_df):
    """
    Helper to filter_duplicate_features.
    
    Check if mean IVT is exactly the same within any two features.
    
    If there are two or more features with exactly the same mean IVT value,
    then remove all but the *first* feature from the data frame of potential
    AR features.
    """
    
    labels_to_remove = []
    for unique_IVT_value in feature_props_df.feature_mean_IVT.unique():        
        df_at_value = feature_props_df[feature_props_df.feature_mean_IVT == unique_IVT_value]
        if len(df_at_value) > 0:
            # Remove all but the first duplicated feature from the data frame
            min_label_at_value = np.min(df_at_value.label)
            labels_to_remove_at_value = list(df_at_value.label)
            del labels_to_remove_at_value[labels_to_remove_at_value.index(min_label_at_value)]
            labels_to_remove.extend(labels_to_remove_at_value)
    
    unique_labels_to_remove = list(set(labels_to_remove))
    feature_props_df_IVT_filtered = feature_props_df.drop(\
        feature_props_df[feature_props_df.label.isin(unique_labels_to_remove)].index)

    return feature_props_df_IVT_filtered

def apply_AR_criteria(AR_config, feature_props_df, grid_cell_area_df, 
                      label_array, u_wrap, v_wrap,
                      lats, lons, lons_wrap):
    """
    Apply final AR criteria to potential AR features, returning an array containing
    the unique labels for each AR.
    - Initial filter to remove very small features that are well below AR area threshold
    - Direction filters (v transport poleward if centroid not in polar region;
      u transport from west to east if centroid in tropics / subtropics)
    - Length and length-to-width ratio filters
    """

    # Output AR labels array. The features are translated from the "wrap" array to this
    # array (which only spans the globe zonally once) by manually placing AR labels
    # at the lat/lon coordinates found to be part of actual ARs in the "wrap" array.
    AR_labels_timestep = np.zeros((len(lats), len(lons)), dtype=np.uint16)
    AR_count_timestep = 0

    # Normalise lons
    lons_mod = lons % 360

    for _, row in feature_props_df.iterrows():
        label = row['label']
        feature_area = row['feature_num_grid_cells']
        feature_props = row['feature_props_IVT']

        # Initial filter of very small features (to avoid unnecessary processing time)
        # - For this and subsequent tests, continue to the next feature in the loop
        #   if the current feature "fails" the test
        if feature_area < AR_config['min_num_grid_pts']:
            continue

        # Calculate feature centroid
        centroid_lat = _calc_centroid_lat(feature_props, lats)
        
        # simpler binary mask than np.where, changed for functionality. 
        feature_array = (label_array == label).astype(int)
        
        # Test if feature is transporting moisture poleward (if not centered in
        # polar latitudes), and is not an east-to-west tropical/subtropical
        # moisture transport plume (if centered in tropical/subtropical latitudes)
        direction_flags = _check_direction(AR_config, centroid_lat, feature_array, u_wrap, v_wrap)
        if direction_flags > 0:
            continue

        # Test if feature meets length and length-to-width ratio criteria
        length_and_shape_flags = _check_length_and_shape(
            AR_config, grid_cell_area_df, feature_array, feature_props, lats, lons_wrap
        )
        if length_and_shape_flags > 0:
            continue
        
        # If the feature passes all checks, assign its grid cells to the AR label array.
        # - For each point in the feature, we match the wrapped longitude back to the
        #   output grid using the closest longitude index.
        # - Unlike the original version, this doesn't try to resolve conflicts where
        #   two features might overlap, whichever one gets assigned first stays.
        # - It assumes overlapping features are rare or already handled earlier,
        #   this may need to be adjusted but seems to be working fine in tests
        ys, xs = np.where(feature_array == 1)
        n_assigned = 0

        for lat_ix, lon_ix_wrap in zip(ys, xs):
            lon_wrap = lons_wrap[lon_ix_wrap] % 360
            lon_ix = np.argmin(np.abs(lons_mod - lon_wrap))

            if 0 <= lat_ix < len(lats):
                AR_labels_timestep[lat_ix, lon_ix] = label
                n_assigned += 1

        if n_assigned > 0:
            AR_count_timestep += 1

    return AR_labels_timestep, AR_count_timestep



def _calc_centroid_lat(feature_props, lats):
    """
    Helper to apply_AR_criteria.
    
    Calculate feature centroid lat/lon.
    """
    
    cent_ixs_float = feature_props.centroid
    lat_ix = \
        math.ceil(cent_ixs_float[0]) if ((math.ceil(cent_ixs_float[0]) - cent_ixs_float[0]) <= 0.5) \
        else math.floor(cent_ixs_float[0])
    centroid_lat = lats[lat_ix]
    
    return centroid_lat

def _check_direction(AR_config, centroid_lat, feature_array, u_wrap, v_wrap):
    """
    Helper to apply_AR_criteria. Simplified from original version. 
    
    Directional filtering for ARs in Northern Hemisphere above 40°N.

    Filters out features that have equatorward v-wind if south of the Arctic Circle.
    """
    feature_ixs = np.where(feature_array == 1)
    
    vwind_equatorward_flag = 0
    v_mean = np.nanmean(v_wrap[0][feature_ixs])
    
    # Only applies to NH above 40N
    if (v_mean < AR_config['v_thresh']) and (centroid_lat < AR_config['v_poleward_cutoff_lat']):
        vwind_equatorward_flag += 1

    return vwind_equatorward_flag

def _check_length_and_shape(AR_config, grid_cell_area_df, feature_array, feature_props, lats, lons_wrap):
    """
    Helper to apply_AR_criteria.
    
    Filter out features that don't meet the length and length-to-width ratio
    criteria.
    """
        
    # Calculate feature area
    feature_grid_cell_areas = []
    for ixs in feature_props.coords:
        lat = lats[ixs[0]]
        area_df_at_lat = grid_cell_area_df[grid_cell_area_df.lat == lat]
        feature_grid_cell_areas.append(list(area_df_at_lat['grid_cell_area_km2'])[0])
    feature_area_km2 = np.nansum(feature_grid_cell_areas)
    
    # Calculate feature perimeter
    feature_array_eroded = ndimage.binary_erosion(feature_array, border_value=0)
    feature_perim_array = feature_array - feature_array_eroded
    feature_perim_ixs = np.where(feature_perim_array == 1)
    perim_pts = []
    for (lat_ix, lon_ix) in zip(feature_perim_ixs[0], feature_perim_ixs[1]):
        perim_pts.append((lats[lat_ix], lons_wrap[lon_ix]))

    # Get the 2 perimeter points with the maximum great circle distance from one
    # another and use these to calculate maximum perimeter great circle distance
    # (proxy for length) in km
    perim_pt1s = []
    perim_pt2s = []
    perim_great_circle_distances = []
    for perim_pt1, perim_pt2 in itertools.combinations(perim_pts, 2):
        perim_pt1s.append(perim_pt1)
        perim_pt2s.append(perim_pt2)
        
        perim_great_circle_distance = great_circle(perim_pt1, perim_pt2).kilometers
        perim_great_circle_distances.append(perim_great_circle_distance)
    
    feature_max_perim_great_circle_distance = np.max(perim_great_circle_distances)
    # "Width" = the effective earth surface width of the feature               
    feature_effective_width_km = feature_area_km2 / feature_max_perim_great_circle_distance
    feature_length_width_ratio = feature_max_perim_great_circle_distance / feature_effective_width_km
                            
    if feature_max_perim_great_circle_distance < AR_config['min_length']:
        length_flag = 1
    else:
        length_flag = 0
        
    if feature_length_width_ratio < AR_config['min_length_width_ratio']:
        length_width_ratio_flag = 1
    else:
        length_width_ratio_flag = 0
    
    length_and_shape_flags = length_flag + length_width_ratio_flag
    
    return length_and_shape_flags



def write_AR_labels_file(AR_config, begin_t, end_t, timestep_hrs,
                         AR_labels, times, lats, lons,
                         climo_start_year, climo_end_year, climo_timestep_hrs):
    """
    Write AR labels output file, with detailed metadata supplied by AR_ID_config.hjson.
    """
    
    # Change data types of AR labels, lats, and lons to save disk space
    ARs_ds = xr.Dataset(
        {
         'AR_labels':(('time','lat','lon'), AR_labels.astype('uint16'))
        },
        coords={
            'time':times,
            'lat':lats.astype('float32'),
            'lon':lons.astype('float32')
        }
    )

    ARs_ds.lat.attrs['units'] = 'degrees_north'
    ARs_ds.lon.attrs['units'] = 'degrees_east'
    
    ARs_ds.attrs['data_source'] = AR_config['data_source']
    ARs_ds.attrs['IVT_data_origin'] = AR_config['IVT_data_origin']
    ARs_ds.attrs['IVT_vert_coordinate'] = AR_config['IVT_vert_coord']
    
    ARs_ds.attrs['IVT_31day_centered_climatology_start_year'] = int(climo_start_year)
    ARs_ds.attrs['IVT_31day_centered_climatology_end_year'] = int(climo_end_year)
    ARs_ds.attrs['IVT_31day_centered_climatology_timestep_hrs'] = int(climo_timestep_hrs)

    ARs_ds.attrs['AR_IVT_minimum_threshold'] = AR_config['IVT_thresh']
    ARs_ds.attrs['AR_IVT_minimum_percentile_rank'] = AR_config['IVT_PR_thresh']
    ARs_ds.attrs['AR_min_length'] = AR_config['min_length']
    ARs_ds.attrs['AR_min_length_width_ratio'] = AR_config['min_length_width_ratio']
    ARs_ds.attrs['AR_direction_filter_type'] = AR_config['direction_filter_type']
    ARs_ds.attrs['AR_poleward_transport_requirement_cutoff_lat'] = AR_config['v_poleward_cutoff_lat']
    ARs_ds.attrs['AR_subtropical_westerly_transport_requirement_cutoff_lat'] = AR_config['subtrop_bound_lat']
    if AR_config['direction_filter_type'] == 'IVT':
        ARs_ds.attrs['AR_poleward_transport_threshold_value_kg_m-1_s-1'] = AR_config['v_thresh']
        ARs_ds.attrs['AR_subtropical_westerly_transport_requirement_value_kg_m-1_s-1'] = AR_config['subtrop_u_thresh']
    elif AR_config['direction_filter_type'] == 'mean_wind_1000_700_hPa':
        ARs_ds.attrs['AR_poleward_transport_threshold_value_m_s-1'] = AR_config['v_thresh']
        ARs_ds.attrs['AR_subtropical_westerly_transport_requirement_value_m_s-1'] = AR_config['subtrop_u_thresh']

    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        ARs_ds.attrs['IVT_calc_pressure_levels'] = str(AR_config['IVT_calc_plevs'])
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        ARs_ds.attrs['IVT_calc_model_levels'] = str(AR_config['IVT_calc_mlevs'])

    t_begin_str = times[0].strftime('%Y%m%d%H%M')
    t_end_str = times[-1].strftime('%Y%m%d%H%M')

    minlat = AR_config['min_lat']
    maxlat = AR_config['max_lat']
    data_source = AR_config['data_source']
    
    # If output data grid covers the 10-90 degree band in the given hemisphere,
    # then label output file as "NH" or "SH". Otherwise, note the latitude band of
    # AR data in the file name.
    if (minlat == 10) and (maxlat == 90):
        fname = f'ARs_{data_source}_NH_{timestep_hrs}hr_{t_begin_str}_{t_end_str}.nc'
    elif (minlat == -90) and (maxlat == -10):
        fname = f'ARs_{data_source}_SH_{timestep_hrs}hr_{t_begin_str}_{t_end_str}.nc'  
    else:
        fname = f'ARs_{data_source}_lat_{minlat}_{maxlat}_{timestep_hrs}hr_{t_begin_str}_{t_end_str}.nc'  

    ARs_ds.to_netcdf(AR_config['AR_output_dir']+fname)
    return fname

def parse_args():
    """
    Parse arguments passed to script at runtime.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('begin_time', help='Begin time in the format YYYY-MM-DD_HHMM')
    parser.add_argument('end_time', help='End time in the format YYYY-MM-DD_HHMM')
    parser.add_argument('timestep_hrs', help='Timestep as integer number of hours (e.g. 3)')
    parser.add_argument('AR_ID_config_path', help='Path to AR ID configuration file')
    args = parser.parse_args()
    
    return args.begin_time, args.end_time, args.timestep_hrs, args.AR_ID_config_path


def main():
    """
    Main block to control reading inputs from command line, ingesting AR ID
    configuration, calculating ARs, and writing AR output file.
    """
    print(f"Script started at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    begin_time, end_time, timestep_hrs, AR_ID_config_path = parse_args()
    
    with open(AR_ID_config_path) as f:
        AR_config = hjson.loads(f.read())
    
    AR_labels, times, lats, lons, climo_start_year, climo_end_year, climo_timestep_hrs = \
        ARs_ID(AR_config, begin_time, end_time, timestep_hrs)

    fname = write_AR_labels_file(AR_config, begin_time, end_time, timestep_hrs,
                                 AR_labels, times, lats, lons,
                                 climo_start_year, climo_end_year, climo_timestep_hrs)
    
    print(f"AR labels saved to: {AR_config['AR_output_dir'] + fname}")

if __name__ == '__main__':
    main()