import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.linalg import svd

lat_grid = np.linspace(35, 65, 13)
lon_grid = np.linspace(170, 340, 69)

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


def analyze_data(model, ssp, g, years, season):
    path = f'{model}/zg_Amon_{model}_{ssp}_r1i1p1f1_{g}_{years}.nc'
    
    ds = xr.open_dataset(path)
    # ds['date'] = pd.to_datetime(ds['date'],format='%Y%m%d')
    full_ds = ds.where(ds['time.season']==season).groupby('time.year').mean()
    ds = ds.sel(lat = slice(35, 65), lon = slice(170, 340))
    ds = ds.where(ds['time.season']==season).groupby('time.year').mean()

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
    print(f'loading shape: {loading.shape}')
    
    return corr_coef, loading[1, :, :], lat, lon
 
 
def analyze_era(path):
    ds = xr.open_dataset(path)
    full_ds = ds.where(ds['valid_time.season']=='JJA').groupby('valid_time.year').mean()
    ds = ds.sel(latitude = slice(65, 35), longitude = slice(170, 340))
    ds = ds.where(ds['valid_time.season']=='JJA').groupby('valid_time.year').mean()

    full_ds = full_ds.sel(year=slice(1950, 2019))
    ds = ds.sel(year=slice(1950, 2019))

    ds = ds['z'].squeeze()
    full_ds = full_ds['z'].squeeze()

    ds = detrend_gridwise(ds, 0)

    lat = ds['latitude'].values
    lon = ds['longitude'].values

    score, latent, loading, north = perform_eof_analysis(ds.to_numpy(), lat, lon, nmodes=2)

    corr_coef = xr.corr(full_ds, xr.DataArray(score[:, 1], dims='year'), dim='year')
    print(score.shape)
    print(full_ds.shape)
    print(corr_coef.shape)
    print(f'loading shape: {loading.shape}')

    
    return corr_coef, loading[1, :, :], lat, lon


models = [['INM-CM4-8', 'gr1'],#0
          ['NorESM2-MM', 'gn'],#1
          ['MIROC6', 'gn'],#2
          ['TaiESM1', 'gn'],#3
          ['ACCESS-CM2', 'gn'],#4
          ['INM-CM5-0', 'gr1']]#5

# models = [['ACCESS-CM2', 'gn']]

years = ['18500116-20141216', '20150116-21001216']

score_era, loadingera, lat_era, lon_era = analyze_era(r'ERA5-data\data_stream-moda_stepType-avgua.nc')

loadingera = xr.DataArray(loadingera, dims=['lat', 'lon'])
loadingera = loadingera.assign_coords(lat=lat_era, lon=lon_era)
loadingera = loadingera.isel(lat=slice(None, None, -1))
loadingera = loadingera.interp(lat=lat_era, lon=lon_era)

for i in range(len(models)):
    model = models[i][0]
    
    # if i == 8:
    #     scorehist, loadinghist, lat_hist, lon_hist = analyze_data(model, 'ssp245', models[i][1], '20150115-21001215', 'JJA')
    
    scorehist, loadinghist, lat_hist, lon_hist = analyze_data(model, 'historical', models[i][1], years[0], 'JJA')
    
    loadinghist = xr.DataArray(loadinghist, dims=['lat', 'lon'])
    loadinghist = loadinghist.assign_coords(lat=lat_hist, lon=lon_hist)

    # print(loadinghist)
    # print(loadingera)

    loadinghist = loadinghist.interp(lat=lat_era, lon=lon_era)

    print(f'585 loading shape: {loadinghist.shape}')
    print(f'era loading shape: {loadingera.shape}')

    corr = xr.corr(loadinghist, loadingera)
    
    with open('data-normalhist-MAM-asdf.txt', 'a') as f:
        f.write(f'370 Corr {model}: {corr.values}\n')
    
    print(f'370 Corr {model}: {corr.values}')