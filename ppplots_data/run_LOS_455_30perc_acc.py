import os, sys
sys.path.append('../')
from gal4H0 import *
import pickle

np.random.seed(0)

filename='LOS_455_30perc_acc.p'
galaxies_list = np.genfromtxt('../MICECAT_LOS/micecat_455.csv',skip_header=1)

Ngw=200
sigma_dl=0.3
zcut_rate=1.4
dl_thr=1550
H0_array=np.linspace(40,120,400)
Nrep=100

output={'H0_grid':H0_array,
       'single_pos':[],
       'true_H0':np.zeros(Nrep)}

for ii in tqdm(range(Nrep)):
    output['true_H0'][ii]=np.random.uniform(40,120,size=1)
    true_cosmology = FlatLambdaCDM(H0=output['true_H0'][ii],Om0=0.25)
    gw_obs_dl,_,_,std_dl=draw_gw_events(Ngw,sigma_dl,dl_thr,galaxies_list,true_cosmology,zcut_rate)
    posterior_matrix,combined=galaxy_catalog_analysis_accurate_redshift(H0_array,galaxies_list,zcut_rate,gw_obs_dl,sigma_dl,dl_thr)
    output['single_pos'].append(posterior_matrix)
    
pickle.dump(output,open(filename,'wb'))