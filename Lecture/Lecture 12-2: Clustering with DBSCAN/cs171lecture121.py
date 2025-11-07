import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from matplotlib.colors import ListedColormap

def read_ocean_properties_dataset():
    ds = nc4.Dataset('Ocean_Surface_Properties.nc')
    temp = ds.variables['Temperature'][:,:]
    salt = ds.variables['Salinity'][:,:]
    o2 = ds.variables['Oxygen'][:,:]
    no3 = ds.variables['Nitrate'][:,:]
    po4 = ds.variables['Phosphate'][:,:]
    fe = ds.variables['Iron'][:,:]
    PP = ds.variables['Primary_Productivity'][:,:]
    sunlight = ds.variables['Sunlight'][:,:]
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    ds.close()
    return(lon, lat, temp, salt, o2, no3, po4, fe, PP, sunlight)

def plot_ocean_properties_dataset(lon, lat, temp, salt, o2, no3, po4, fe, PP, sunlight):
    var_names = ['temp', 'salt', 'o2', 'no3', 'po4', 'fe', 'PP', 'sunlight']
    metadata = {'temp':['Temperature','$^{\\circ}$C',-2,32,'turbo'],
                'salt':['Salinity','p.s.u.',28,36,'viridis'],
                'o2':['Oxygen','$\\mu$M',200,400,'PuRd'],
                'no3':['Nitrate','$\\mu$M',0,30,'PuRd'],
                'po4':['Phosphate','$\\mu$M',0,2,'PuRd'],
                'fe':['Iron','$\\mu$M',0,0.001,'YlOrBr'],
                'PP':['Primary Productivity','mmol C/m$^3$/s',0,0.02,'BuGn'],
                'sunlight':['Sunlight','$\\mu$Ein/m$^2$/s',-50,450,'YlOrRd']}
    var_grids = [temp, salt, o2, no3, po4, fe, PP*1000, sunlight]
    mask = temp==0
    fig = plt.figure(figsize=(10,12))
    for v, var_name in enumerate(var_names):
        plt.subplot(4,2,v+1)
        ax = plt.gca()
        C = ax.pcolormesh(lon,lat,var_grids[v],
                          vmin=metadata[var_name][2],
                          vmax=metadata[var_name][3],
                          cmap=metadata[var_name][4])
        ax.contourf(lon, lat, temp==0, levels=[0.01,1], cmap="Greys")
        plt.colorbar(C, label=metadata[var_name][1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylabel(metadata[var_name][0])
    plt.show()

def plot_biome_classification(lon,lat,classification, temp):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(8,5))
    biomes = np.copy(temp.ravel())
    biomes[temp.ravel()!=0]= classification+1
    biomes = biomes.reshape(np.shape(temp))
    for c in range(1,np.max(classification)+2):
        biome_masked = np.ma.masked_where(biomes!=c, biomes)
        biome_masked = np.ma.masked_where(temp==0, biome_masked)
        single_color_cmap = ListedColormap([colors[(c-1)%10]])
        plt.pcolormesh(lon,lat,biome_masked,cmap=single_color_cmap)
        plt.plot(-1000,-1000,'s',color=colors[(c-1)%10], label='Biome '+str(c))
    plt.xlim([np.min(lon),np.max(lon)])
    plt.ylim([np.min(lat),np.max(lat)])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
