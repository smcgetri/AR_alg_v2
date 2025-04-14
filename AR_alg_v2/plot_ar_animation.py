import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# === FILE PATHS ===
ivt_file = "/gws/nopw/j04/canari/users/singet75/large-ensemble/derived/HIST2_new/1/ATM/yearly/1950/ivt/ivt_tcwv_wspeed_ensemble1_year1950.nc"
pctile_file = "/gws/nopw/j04/canari/users/singet75/AR_IVT_percentiles/IVT_at_pctiles_CANARI_1950_2014.nc"
ar_file = "/gws/nopw/j04/canari/users/singet75/AR_output/ARs_CANARI_lat_40_89.445_6hr_195001010600_195012301800.nc"

# === LOAD DATASETS ===
ds_ivt = xr.open_dataset(ivt_file)
ds_pctile = xr.open_dataset(pctile_file)
ds_ar = xr.open_dataset(ar_file)

# === TIME RANGE FOR JUNE 1950 ===
times = ds_ivt.time.sel(time=slice("1950-01-01", "1950-12-30")).values

# === FIRST FRAME DATA ===
time_sel = times[0]
ivt_t = ds_ivt.ivt_magnitude.sel(time=time_sel).rename({'lat_um_atmos_grid_uv': 'lat', 'lon_um_atmos_grid_uv': 'lon'})
doy = (ivt_t.time.dt.month - 1) * 30 + ivt_t.time.dt.day
ivt_thresh = ds_pctile["IVT_pctile_85"].sel(doy=int(doy))
ar_t = ds_ar.AR_labels.sel(time=time_sel)
ar_data = ar_t.where(ar_t > 0)

vmin = float(min(ivt_t.min(), ivt_thresh.min()))
vmax = float(max(ivt_t.max(), ivt_thresh.max()))

# === FIGURE SETUP ===
proj = ccrs.NorthPolarStereo()
data_crs = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': proj})
titles = ["IVT Magnitude", "IVT Threshold (85th percentile)", "AR Labels (label > 0)"]

for ax in axes:
    ax.set_extent([-180, 180, 40, 90], crs=data_crs)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.3, linestyle='--', alpha=0.5)

# === Add Arctic Circle ===
def draw_arctic_circle(ax, radius_lat=66.56, **kwargs):
    lons = np.linspace(-180, 180, 360)
    lats = np.full_like(lons, radius_lat)
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)

for ax in axes:
    draw_arctic_circle(ax, radius_lat=66.56, color='red', linestyle='--', linewidth=1.2)

# === STATIC PLOTS ===
ivt_img = axes[0].pcolormesh(ivt_t.lon, ivt_t.lat, ivt_t, transform=data_crs,
                             cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
fig.colorbar(ivt_img, ax=axes[0])
axes[0].set_title(titles[0])

thresh_img = axes[1].pcolormesh(ivt_thresh.lon, ivt_thresh.lat, ivt_thresh, transform=data_crs,
                                cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
fig.colorbar(thresh_img, ax=axes[1])
axes[1].set_title(titles[1])

# === AR LABELS: single color for presence ===
cmap_ar = mcolors.ListedColormap(["none", "tab:blue"])
norm_ar = mcolors.BoundaryNorm([0, 0.5, 1.5], cmap_ar.N)

label_img = axes[2].pcolormesh(ar_t.lon, ar_t.lat, ar_data, transform=data_crs,
                               cmap=cmap_ar, norm=norm_ar, shading="auto")
fig.colorbar(label_img, ax=axes[2], ticks=[1], label="AR Present")
axes[2].set_title(titles[2])


# === UPDATE FUNCTION ===
def update(frame_idx):
    time_sel = times[frame_idx]
    ivt_t = ds_ivt.ivt_magnitude.sel(time=time_sel).rename({'lat_um_atmos_grid_uv': 'lat', 'lon_um_atmos_grid_uv': 'lon'})
    doy = (ivt_t.time.dt.month - 1) * 30 + ivt_t.time.dt.day
    ivt_thresh = ds_pctile["IVT_pctile_85"].sel(doy=int(doy))
    ar_t = ds_ar.AR_labels.sel(time=time_sel)
    ar_data = ar_t.where(ar_t > 0).fillna(0)

    # Update arrays
    ivt_img.set_array(ivt_t.values.ravel())
    thresh_img.set_array(ivt_thresh.values.ravel())
    label_img.set_array(ar_data.values.ravel() if np.any(ar_data > 0) else np.full_like(ar_data.values.ravel(), np.nan))

    # Title
    fig.suptitle(f"IVT + AR Detection – {time_sel.strftime('%Y-%m-%d %H:%M')}", fontsize=16)

# === MAKE ANIMATION ===
ani = animation.FuncAnimation(fig, update, frames=len(times), interval=200)
ani.save("ar_animation_full_year1950.mp4", writer="ffmpeg", fps=5)
print("Animation saved as ar_animation_full_year1950.mp4")
