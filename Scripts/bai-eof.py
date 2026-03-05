import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.linalg import svd


def detrend_gridwise(data, ax):
    
    detrended = detrend(data, axis=ax, type='linear')
    
    return xr.DataArray(detrended, coords=data.coords, dims=data.dims, attrs=data.attrs)


def perform_eof_analysis(dat, lati, loni, flag=2, nmodes=10):
    """
    Perform EOF analysis on the provided dataset.

    Parameters:
    dat: np.ndarray
        The 3D dataset to analyze, with dimensions [time, lat, lon].
    lati: np.ndarray
        The latitude array corresponding to the dataset's grid.
    loni: np.ndarray
        The longitude array corresponding to the dataset's grid.
    flag: int, optional (default=2)
        Determines whether to perform PCA on the covariance matrix (flag=1) or on the correlation matrix (flag=2).
    nmodes: int, optional (default=10)
        The number of principal components (PCs) or EOF modes to calculate.

    Returns:
    score: np.ndarray
        The time series of each PC in a column.
    latent: np.ndarray
        The eigenvalue of each PC normalized to sum to 1.
    loading: np.ndarray
        The loading matrix of EOFs, with shape [nmodes, nlat, nlon].
    north: np.ndarray
        The North et al. criterion values for the first nmodes.
    """
    
    # Extract the dimensions of the data: time (nt), latitude (nlat), longitude (nlon)
    nt, nlat, nlon = dat.shape

    # Convert 1D latitude/longitude to 2D grids if needed
    if lati.ndim == 1:
        print("Converting lati and loni to 2D grids.")
        loni, lati = np.meshgrid(loni, lati)

    # Latitude weighting factor (cosine of latitude)
    wf = np.sqrt(np.cos(np.deg2rad(lati)))

    # Repeat the weighting factor across the time dimension
    wf = np.tile(wf, (nt, 1, 1))

    # Apply the latitude weighting to the data
    dat = dat * wf

    # Reshape the data into [time x (lat * lon)] for PCA
    dat = dat.reshape(nt, -1)

    # Center the data by removing the mean
    dat -= np.mean(dat, axis=0)

    # Standardize the data if performing PCA on the correlation matrix (flag == 2)
    if abs(flag) == 2:
        s = np.std(dat, axis=0)
        s[s == 0] = 1  # Prevent division by zero
        dat /= s

    # Perform Singular Value Decomposition (SVD)
    U, s, Vt = svd(dat / np.sqrt(nt - 1), full_matrices=False)

    # North et al. criterion for determining significant modes
    gg = s ** 2
    north = np.zeros((nmodes + 1, 4))
    north[:, 0] = gg[:nmodes + 1]  # Eigenvalues
    north[:, 1] = gg[:nmodes + 1] * np.sqrt(2 / nt)  # Uncertainty

    # Compute upper and lower bounds for the North criterion
    check = np.zeros((nmodes + 1, 2))
    check[:, 0] = gg[:nmodes + 1] + north[:, 1]
    check[:, 1] = gg[:nmodes + 1] - north[:, 1]

    # Apply the North criterion
    north[0, 2] = check[0, 1] > check[1, 0]
    for i in range(1, nmodes):
        north[i, 2] = (check[i, 0] < check[i - 1, 1]) & (check[i, 1] > check[i + 1, 0])

    # Calculate differences for North criterion
    delta = np.diff(north[:, 0])
    north[:-1, 3] = np.abs(delta / north[:-1, 1])

    # Project data onto the principal components (EOFs)
    score = dat @ Vt.T

    # Eigenvalues (latent) normalized to sum to 1
    latent_raw = s ** 2
    latent = latent_raw / np.sum(latent_raw)

    # Reshape the first nmodes of EOFs into [lat x lon]
    loading = np.full((nmodes, nlat, nlon), np.nan)
    for i in range(min(nmodes, Vt.shape[0])):
        loading[i, :] = Vt[i, :].reshape(nlat, nlon)

    return score, latent, loading, north


def analyze_data(model, ssp, g, years):
    path = f'{ssp}/zg_Amon_{model}_{ssp}_r1i1p1f1_{g}_{years}.nc'
    
    ds = xr.open_dataset(path)
    # ds['date'] = pd.to_datetime(ds['date'],format='%Y%m%d')
    full_ds = ds.where(ds['time.season']=='MAM').groupby('time.year').mean()
    ds = ds.sel(lat = slice(35, 65), lon = slice(170, 340))
    ds = ds.where(ds['time.season']=='MAM').groupby('time.year').mean()

    ds = ds['zg'].squeeze()
    full_ds = full_ds['zg'].squeeze()

    ds = detrend_gridwise(ds, 0)

    lat = ds['lat'].values
    lon = ds['lon'].values

    score, latent, loading, north = perform_eof_analysis(ds.to_numpy(), lat, lon, nmodes=2)
    
    corr_coef = xr.corr(full_ds, xr.DataArray(score[:, 1], dims='year'), dim='year')
    print(score.shape)
    print(full_ds.shape)
    print(corr_coef.shape)
    
    return lat, lon, corr_coef, loading


def plot_data(lat, lon, loading, score, model, ssp):
    # fig = plt.figure(figsize=(8, 3))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.set_title(f"EOF2 {model}")
    # cb = ax.contourf(lon, lat, loading[1, :, :], levels=30, cmap='RdBu')
    # ax.set_xticks(range(-170, -20, 15))
    # ax.set_yticks(range(0, 80, 15))
    # ax.set_xbound(-180, -20)
    # ax.set_ybound(10, 80)
    # ax.coastlines()

    # plt.colorbar(cb, label="EOF2", orientation="vertical")

    # plt.show()
    
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_title(f"PC2 Corr {model}-{ssp}")
    cb = ax.contourf(score['lon'], score['lat'], score, levels=30, cmap='RdBu')
    ax.set_xticks(range(-170, -20, 15))
    ax.set_yticks(range(0, 80, 15))
    ax.set_xbound(-180, -20)
    ax.set_ybound(10, 80)
    ax.coastlines()
    ax.grid()

    plt.colorbar(cb, label="EOF2", orientation="vertical")
    plt.savefig(f'{model}-{ssp}.png')
 

models = [['ACCESS-CM2', 'gn'],#0
          ['INM-CM4-8', 'gr1'],#1
          ['INM-CM5-0', 'gr1'],#2
          ['MIROC6', 'gn'],#3
          ['NorESM2-MM', 'gn'],#4
          ['TaiESM1', 'gn'],#5
          ['FIO-ESM-2-0', 'gn'],#6
          ['BCC-CSM2-MR', 'gn'],#7
          ['MPI-ESM1-2-HR', 'gn'],#8
          ['ACCESS-ESM1-5', 'gn'],#9
          ['CESM2-WACCM', 'gn'],#10
          ['FGOALS-f3-L', 'gr']]#11

for i in range(len(models)):
    
    print(f'Analyzing {models[i][0]}-ssp245')
    
    if i != 8 and i != 9 and i != 10 and i != 11:
        lat245, lon245, score245, loading245 = analyze_data(models[i][0], 'ssp245', models[i][1], '20150116-21001216')
        plot_data(lat245, lon245, loading245, score245, models[i][0], 'ssp245')
    
    if i != 6 and i != 8 and i != 9 and i != 10:
        print(f'Analyzing {models[i][0]}-ssp370')
        lat370, lon370, score370, loading370 = analyze_data(models[i][0], 'ssp370', models[i][1], '20150116-21001216')
        plot_data(lat370, lon370, loading370, score370, models[i][0], 'ssp370')
        
    if i == 10:
        print(f'Analyzing {models[i][0]}-ssp370')
        lat370, lon370, score370, loading370 = analyze_data(models[i][0], 'ssp370', models[i][1], '20150115-21001215')
        plot_data(lat370, lon370, loading370, score370, models[i][0], 'ssp370')
        
        print(f'Analyzing {models[i][0]}-historical')
        lathist, lonhist, scorehist, loadinghist = analyze_data(models[i][0], 'historical', models[i][1], '18500115-20141215')
        plot_data(lathist, lonhist, loadinghist, scorehist, models[i][0], 'historical')
    
    if i != 7 and i != 10:
        print(f'Analyzing {models[i][0]}-historical')
        lathist, lonhist, scorehist, loadinghist = analyze_data(models[i][0], 'historical', models[i][1], '18500116-20141216')
        plot_data(lathist, lonhist, loadinghist, scorehist, models[i][0], 'historical')
