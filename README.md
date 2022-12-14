This repository contains the data and code distribution for  the "The Hitchhiker's guide to the galaxy catalog approach for gravitational wave cosmology". You are free to clone and use the software as you like.

To run the notebooks you will need a working copy of Python3 with standard packages such as numpy and scipy.

## Short description of the content

* `gal4H0.py`: Python module collecting all the function used for simulating mock data and calculating the hierarchical likelihood as function of H0.

* `plotting_101.ipynb`: It contains some plots of the GW detection probability and the galaxy density interpolant.

* `ppplots.ipynb`: It produces the PP-plots (Parameter-Parameter plots). Unfortunately git does not have the possibility of storing very heavy files. To produce the files needed for this notebook, you will need to run the python scripts in the folder `ppplots_data`. The files will be stored there.

* `H0analysis.ipynb`: Do the analysis for 200 GW events exploring various cases for the GW likelihood.

* `H0analysis_perfect_z.ipynb`: Do the analysis assumming that the galaxy redshift is perfectly known.

* `H0analysis_realuniform.ipynb`: Do the analysis simulating a uniform in comoving volume distribution of galaxies. 

* `H0analysis_OneGal.ipynb`: Do the analysis with the full-likelihood formalism, in the case that there are multiple GW events in the same galaxies.

* `H0analysis_4inconsistencies.ipynb`: Do the analysis exploring some of the possible inconsistencies when generating mock data and using the catalog statistical method.
