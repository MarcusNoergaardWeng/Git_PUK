import sys
sys.path.append('../scripts')
from common_settings import n_sites_OH, n_sites_O, n_metals, sites_OH, sites_O, metals
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import itertools as it

path_DFT = '../DFT_data'

# Load DFT adsorption energies and features for *OH
data = np.loadtxt(f'{path_DFT}/OH_features.csv', delimiter=',', skiprows=1)
features_OH = data[:, :-3].astype(int)
Gs_OH = data[:, -3]

# Load DFT adsorption energies and features for O*
data = np.loadtxt(f'{path_DFT}/O_features.csv', delimiter=',', skiprows=1)
features_O = data[:, :-3].astype(int)
Gs_O = data[:, -3]

# Train linear regressor on the features and adsorption energies
zone_labels = ['1a', '2a', '3b']
n_atoms_zones = [6, 3, 3]
for ontop_idx in range(n_metals):
	
	# Get indices of the current adsorption site
	ids = np.nonzero(features_OH[:, ontop_idx] == 1)[0]
	
	# Train linear regressor on this OH on-top site
	reg = LinearRegression(fit_intercept=True).fit(features_OH[ids, 5:], Gs_OH[ids])
	
	# Get fitted slopes
	slopes = reg.coef_
	intercept = reg.intercept_
	slopes_init = deepcopy(slopes)
	
	for zone_idx, n_atoms in enumerate(n_atoms_zones):
		
		# Set the pure metal slope to zero
		ref_val = slopes[zone_idx*n_metals + ontop_idx]
		slopes[zone_idx*n_metals : (zone_idx+1)*n_metals] -= ref_val
		
		# Update intercept parameter to make up for the change in slopes
		intercept += ref_val*n_atoms

	# Write parameters to file
	with open(f'OH_{sites_OH[ontop_idx]}.csv', 'w') as file_:
		
		# Write intercept to file
		file_.write(f'intercept,{intercept:>9.6f}')
		
		# Write slopes to file
		for (label, metal), slope in zip(it.product(zone_labels, metals), slopes):
			file_.write(f'\n{label}{metal},{slope:>9.6f}')

# Train linear O* regressor	
reg_O = LinearRegression(fit_intercept=False).fit(features_O, Gs_O)

# Set initial slopes and intercept
slopes = reg_O.coef_
intercept = 0.

# Get the index of Pt, whose parameters will be set to zero
ref_idx = metals.index('Pt')

zone_labels = ['1a', '1b', '2a', '2b']
n_atoms_zones = [3, 6, 3, 3]
for zone_idx, n_atoms in enumerate(n_atoms_zones):
	
	# Get value of reference metal parameter in the current zone
	start_idx = n_sites_O + zone_idx*n_metals
	ref_val = slopes[start_idx + ref_idx]
	
	# Update slopes and intercept
	stop_idx = n_sites_O + (zone_idx+1)*n_metals
	slopes[start_idx : stop_idx] -= ref_val
	intercept += ref_val*n_atoms

# Subtract the intercept from the adsorption site ensemble parameters,
# since they are one hot encoded only one will be active at a time
slopes[:n_sites_O] += intercept

# Write parameters to file
with open(f'O.csv', 'w') as file_:
	
	# Write ensemble energies to file
	for site, param in zip(sites_O, slopes):
		file_.write(f'{site},{param:>9.6f}\n')
	
	# Write slopes to file
	for (label, metal), slope in zip(it.product(zone_labels, metals), slopes[n_sites_O:]):
		file_.write(f'{label}{metal},{slope:>9.6f}\n')
