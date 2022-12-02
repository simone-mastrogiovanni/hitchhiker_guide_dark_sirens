import os, sys
sys.path.append('../')
from gal4H0 import *
import pickle

np.random.seed(0)

filename='LOS_451_30perc.p'
galaxies_list = np.genfromtxt('../MICECAT_LOS/micecat_451.csv',skip_header=1)
galaxies_list = replicate_galaxies(galaxies_list)

Ngw=200
sigma_dl=0.3
zcut_rate=1.4
dl_thr=1550
H0_array=np.linspace(40,120,300)
Nrep=100

output={'H0_grid':H0_array,
       'single_pos':[],
       'true_H0':np.zeros(Nrep)}

sigmaz=0.013*np.power(1+galaxies_list,3.)
sigmaz[sigmaz>0.015]=0.015
z_obs=np.random.randn(len(galaxies_list))*sigmaz+galaxies_list
zinterpo,zinterpolant=build_interpolant(z_obs,sigmaz,zcut_rate)

for ii in tqdm(range(Nrep)):
    output['true_H0'][ii]=np.random.uniform(40,120,size=1)
    true_cosmology = FlatLambdaCDM(H0=output['true_H0'][ii],Om0=0.25)
    gw_obs_dl,_,_,std_dl=draw_gw_events(Ngw,sigma_dl,dl_thr,galaxies_list,true_cosmology,zcut_rate)
    posterior_matrix, combined=galaxy_catalog_analysis_photo_redshift(H0_array,zinterpo,gw_obs_dl,sigma_dl,dl_thr)
    output['single_pos'].append(posterior_matrix)
    
pickle.dump(output,open(filename,'wb'))