import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.linalg import svd

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 25})


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
    full_ds = detrend_gridwise(full_ds, 0)

    lat = ds['lat'].values
    lon = ds['lon'].values

    score, latent, loading, north = perform_eof_analysis(ds.to_numpy(), lat, lon, nmodes=2)
    
    corr_coef = xr.corr(full_ds, xr.DataArray(score[:, 1], dims='year'), dim='year')
    print(score.shape)
    print(full_ds.shape)
    print(corr_coef.shape)
    
    return corr_coef, loading
 
 
def many_datas(model, g, years):
    score245_JJA, loading245 = analyze_data(model, 'ssp245', g, years[1], 'JJA')
    score370_JJA, loading370 = analyze_data(model, 'ssp370', g, years[1], 'JJA')
    scorehist_JJA, loadinghist = analyze_data(model, 'historical', g, years[0], 'JJA')

    score245_MAM, loading245 = analyze_data(model, 'ssp245', g, years[1], 'MAM')
    score370_MAM, loading370 = analyze_data(model, 'ssp370', g, years[1], 'MAM')
    scorehist_MAM, loadinghist = analyze_data(model, 'historical', g, years[0], 'MAM')
    
    # scorehist_MAM *= -1
    score370_MAM *= -1
    # score245_JJA *= -1
    
    scores = [[scorehist_MAM, score245_MAM, score370_MAM], [scorehist_JJA, score245_JJA, score370_JJA]]
    return scores


def plot_many(scores, model):
        
    ssps = ['HIST', 'SSP245', 'SSP370', 'SSP585']
    seasons = ['MAM', 'JJA']
    
    # fig = plt.figure(figsize=(17, 6))
    # fig.suptitle(f'Model Circulation Pattern of {model} PC2')
    # gs = GridSpec(2, 5)
    
    # for j in range(2):
    #     for i in range(4):
    #         score = scores[j][i]
    #         ax = fig.add_subplot(gs[j, i], projection=ccrs.PlateCarree())
    #         im = ax.contourf(score['lon'], score['lat'], score, levels=30, cmap='RdBu_r')
    #         title = ''
    #         title += seasons[j] + ' '
    #         title += ssps[i]
    #         ax.coastlines()
    #         ax.set_xticks(range(-180, -20, 20))
    #         ax.set_yticks(range(10, 80, 20))
    #         ax.set_xbound(-180, -20)
    #         ax.set_ybound(10, 80)
    #         # ax[j, i].set_xlabel('Longitude')
    #         # ax[j, i].set_ylabel('Latitude')
    #         ax.grid()
    #         ax.set_title(title)
            

    fig, ax = plt.subplots(2, 3, figsize=(17, 6), subplot_kw={'projection' : ccrs.PlateCarree()})
    
    for j in range(2):
        for i in range(3):
            score = scores[j][i]
            im = ax[j, i].contourf(score['lon'], score['lat'], score, levels=np.linspace(-.9,.9, 30), cmap='RdBu_r')
            
            title = ''
            
            if i == 0:
                title += 'Historical: '
            elif i == 1:
                title += 'SSP245: '
            else:
                title += 'SSP370: '

            if j == 0:
                title += 'Spring'
            else:
                title += 'Summer'
            
            # plt.colorbar(im, ax=ax[j, i], fraction=0.022, pad=0.025)
            ax[j, i].coastlines()
            ax[j, i].set_xticks(range(-170, -20, 30))
            ax[j, i].set_yticks(range(10, 80, 20))
            ax[j, i].set_xbound(-180, -20)
            ax[j, i].set_ybound(10, 80)
            # ax[j, i].set_xlabel('Longitude')
            # ax[j, i].set_ylabel('Latitude')
            ax[j, i].grid()
            ax[j, i].set_title(title)
            
    # cb_ax = fig.add_subplot(gs[:, 4])
    # cb_ax.set_position([0, 0, .01, .01])
    # fig.colorbar(im, cax=cb_ax)
            
    fig.tight_layout()
    
    # # plt.savefig(f'{model}-pc2.png')
    plt.show()
 
years = ['18500116-20141216', '20150116-21001216']

# models = [['ACCESS-CM2', 'gn'],#0
#           ['INM-CM4-8', 'gr1'],#1
#           ['INM-CM5-0', 'gr1'],#2
#           ['MIROC6', 'gn'],#3
#           ['NorESM2-MM', 'gn'],#4
#           ['TaiESM1', 'gn']]#5

# for i in range(len(models)):
#     model = models[i][0]
#     print(f'analysing {model}')
#     scores = many_datas(model, models[i][1], years)
#     plot_many(scores, model)

scores = many_datas('INM-CM5-0', 'gr1', years)
plot_many(scores, 'INM-CM5-0')

# score, loading, lat, lon = analyze_data('BCC-CSM2-MR', 'ssp585', 'gn', years[1], 'MAM')

# fig = plt.figure(figsize=(8, 3))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# cb = ax.contourf(score['lon'], score['lat'], score, levels=30, cmap='RdBu')
# ax.coastlines()
# ax.set_xticks(range(-180, -20, 20))
# ax.set_yticks(range(10, 80, 20))
# ax.set_xbound(-180, -20)
# ax.set_ybound(10, 80)
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.grid()

# cb_ax = fig.add_axes()
# fig.colorbar(cb, cax=cb_ax)

# plt.show()
