import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.linalg import svd

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12})


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
    # full_ds = detrend_gridwise(full_ds, 0)

    lat = ds['lat'].values
    lon = ds['lon'].values

    score, latent, loading, north = perform_eof_analysis(ds.to_numpy(), lat, lon, nmodes=2)
    
    corr_coef = xr.corr(full_ds, xr.DataArray(score[:, 1], dims='year'), dim='year')
    print(score.shape)
    print(full_ds.shape)
    print(corr_coef.shape)
    
    return corr_coef, loading[1], lat, lon
 
 
def many_datas(model, g, years):
    score245_JJA, loading245_JJA, lat, lon = analyze_data(model, 'ssp245', g, years[1], 'JJA')
    score370_JJA, loading370_JJA, lat, lon = analyze_data(model, 'ssp370', g, years[1], 'JJA')
    score585_JJA, loading585_JJA, lat, lon = analyze_data(model, 'ssp585', g, years[1], 'JJA')
    scorehist_JJA, loadinghist_JJA, lat, lon = analyze_data(model, 'historical', g, years[0], 'JJA')

    score245_MAM, loading245_MAM, lat, lon = analyze_data(model, 'ssp245', g, years[1], 'MAM')
    score370_MAM, loading370_MAM, lat, lon = analyze_data(model, 'ssp370', g, years[1], 'MAM')
    score585_MAM, loading585_MAM, lat, lon = analyze_data(model, 'ssp585', g, years[1], 'MAM')
    scorehist_MAM, loadinghist_MAM, lat, lon = analyze_data(model, 'historical', g, years[0], 'MAM')
    
    scores = [[loadinghist_MAM, loading245_MAM, loading370_MAM, loading585_MAM], [loadinghist_JJA, loading245_JJA, loading370_JJA, loading585_JJA]]
    return scores, lat, lon


def plot_many(scores, model, lat, lon):
    
    ssps = ['HIST', 'SSP245', 'SSP370', 'SSP585']
    seasons = ['MAM', 'JJA']

    fig, ax = plt.subplots(2, 4, figsize=(18, 7), subplot_kw={'projection' : ccrs.PlateCarree()})
    fig.suptitle(f'Model Circulation Pattern of {model} PC2')
    
    for j in range(2):
        for i in range(4):
            score = scores[j][i]
            im = ax[j, i].contourf(lon, lat, score, levels=30, cmap='RdBu_r')
            title = ''
            title += seasons[j] + ' '
            title += ssps[i]
            plt.colorbar(im, ax=ax[j, i], fraction=0.022, pad=0.025)
            ax[j, i].coastlines()
            ax[j, i].set_xticks(range(-180, -20, 20))
            ax[j, i].set_yticks(range(10, 80, 20))
            ax[j, i].set_xbound(-180, -20)
            ax[j, i].set_ybound(10, 80)
            # ax[j, i].set_xlabel('Longitude')
            # ax[j, i].set_ylabel('Latitude')
            ax[j, i].grid()
            ax[j, i].set_title(title)
            
    fig.tight_layout()
    
    plt.show()
 
years = ['18500116-20141216', '20150116-21001216']

models = [['ACCESS-CM2', 'gn'],#0
          ['INM-CM4-8', 'gr1'],#1
          ['INM-CM5-0', 'gr1'],#2
          ['MIROC6', 'gn'],#3
          ['NorESM2-MM', 'gn'],#4
          ['TaiESM1', 'gn']]#5

for i in range(len(models)):
    model = models[i][0]
    print(f'analysing {model}')
    scores, lat, lon = many_datas(model, models[i][1], years)
    plot_many(scores, model, lat, lon)
