import itertools as it
import numpy as np
import re
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt

# Define metals in alloy
metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
n_metals = len(metals)

# Define colors of metals
metal_colors = dict(Ag = 'silver',
					Au = 'gold',
					Cu = 'darkorange',
					Pd = 'dodgerblue',
					Pt = 'lime',					
					H = 'white',
					O = 'red')

# Get *OH on-top and O* fcc hollow site labels
sites_OH = [''.join(comb) for comb in it.combinations_with_replacement(metals, 1)]
n_sites_OH = len(sites_OH)
sites_O = [''.join(comb) for comb in it.combinations_with_replacement(metals, 3)]
n_sites_O = len(sites_O)

# Define number of electrons involved in the reduction of O* and *OH to H2O
n_elec_OH = 1
n_elec_O = 2

# Get potentials to start and stop at (V vs. RHE)
U_start = 0.25
U_stop = 1.6
U_step = 0.05

# Define adsorption energies for plots of adsorption energy distributions
potentials = np.arange(U_start, U_stop+U_step, step=U_step)
G_OHs = n_elec_OH * potentials
G_Os = n_elec_O * potentials

# Define adsorption energy steps
G_step_OH = n_elec_OH*U_step
G_step_O = n_elec_O*U_step

# Define adsorption energy bins corresponding to the locations of the
# 'analytical' (i.e. non-binned) adsorption energies
bins_OH = np.concatenate((G_OHs - 0.5*G_step_OH, [G_OHs[-1] + 0.5*G_step_OH]))
bins_O = np.concatenate((G_Os - 0.5*G_step_O, [G_Os[-1] + 0.5*G_step_O]))

# Get colors of on-top and fcc sites as the average RGB values of the constituent elements
site_colors = {}
for site in list(sites_OH) + list(sites_O):
	site_metals = re.findall('[A-Z][^A-Z]*', site)
	site_colors[site] = np.mean(np.array([to_rgb(metal_colors[m]) for m in site_metals]), axis=0)

# Define ordering (z-order and label order) of
# the distributions in the plot (back --> top)
metals_order = ['Pt', 'Cu', 'Pd', 'Au', 'Ag']
site_order = {}
for n_atoms_site in [1, 3]:
	for order_idx, comb in enumerate(it.combinations_with_replacement(metals_order, n_atoms_site), start=1):
		site = ''.join(sorted(comb))
		site_order[site] = order_idx

# Set histogram parameters
total_hist_kw = dict(histtype='step', lw=0.7)
site_hist_kw = dict(histtype='stepfilled', edgecolor='black', lw=0.5, alpha=0.6)

# Set x-axis limits
G_OH_lims = n_elec_OH*np.array([U_start, U_stop])
G_O_lims = n_elec_O*np.array([U_start, U_stop])

def get_time_stamp(dt):
	'''
	Return the elapsed time in a nice format.
	
	Parameters
	----------
	dt: float
		Elapsed time in seconds.
		
	Return
	------
	string
		Elapsed time in a neat human-radable format.
	'''
	dt = int(dt)
	if dt < 60:
		return '{}s'.format(dt)
	elif 60 < dt < 3600:
		mins = dt//60
		secs = dt%60
		return '{:d}min{:d}s'.format(mins, secs)
	else:
		hs = dt//3600
		mins = dt//60 - hs*60
		secs = dt%60
		return '{:d}h{:d}min{:d}s'.format(hs, mins, secs)

# Define Boltzmann's constant
kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kBT = kB*300 # eV

def get_activity(energies, E_opt, n_atoms_surface, eU, jD=1., ps=1., front_factor=1.):
    '''
    Return the activity per surface atom calculated using the
    Angewandte Chemie equations 2-4 (doi: 10.1002/anie.202014374)
    '''
    jki = front_factor * np.exp((-np.abs(energies - E_opt) + 0.86 - eU) / kBT)
    return np.sum(ps * (1. / (1. / jki + 1./jD))) / n_atoms_surface

n_save = 5

# Make figure for plotting adsorption energies
stop_auto_plot_in_notebook = True
if not stop_auto_plot_in_notebook:
    fig, axes = plt.subplots(figsize=(2*n_save+1, 4), nrows=2, ncols=n_save,
                             gridspec_kw=dict(hspace=0.4, wspace=0.03),
                             sharey='row', sharex='row')

    # Make coverage axes on the right axis
    coverage_axes = np.array([[axes[row, col].twinx() for col in range(n_save)] for row in range(2)])
    coverage_kw = dict(lw=0.9, color='darkgreen', zorder=100)

def prepare_and_save_plot(filename, fig, axes, coverage_axes):

	# Set x-ticks
	axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
	axes[0, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
	axes[0, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

	axes[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(1.0))
	axes[1, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
	axes[1, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

	for row in range(2):
		for col in range(n_save):
		
			# Set y-axis limits on coverage axis
			coverage_axes[row, col].set_ylim(0.0, 1.0)
			
			# Set y-axis ticks on coverage axis
			coverage_axes[row, col].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
			coverage_axes[row, col].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
			
			# Remove left y-axis labels on all axes
			axes[row, col].tick_params(which='both', left=False, labelleft=False)
			
			# Remove right y-axis labels on all but the right-most coverage axis
			if col < n_save-1:
				coverage_axes[row, col].tick_params(which='both', right=False, labelright=False)
				
			# Color ticks and tick labels on the right-most y-axis
			else:
				coverage_axes[row, col].tick_params(which='both', axis='y', colors='darkgreen')	

			# Remove first tick label from x axis
			plt.setp(axes[row, col].get_xticklabels()[0], visible=False)
			
			# Remove last tick label from x axis
			plt.setp(axes[row, col].get_xticklabels()[-2], visible=False)
			
	# Make custom legend
	custom_lines = []
	for site in metals_order:
		custom_lines.append(Patch(facecolor=site_colors[site], edgecolor='black', label=site))
	custom_lines.append(Line2D([0], [0], lw=1., color='black', label='total'))
	y_top = axes[0,0].get_position().y1
	fig.legend(handles=custom_lines, bbox_to_anchor=(0.5, y_top), loc='lower center', shadow=True,
			   ncol=len(sites_OH)+1, fontsize=14)

	# Write x-axis labels
	text_kw = dict(x=0.5, fontsize=14, transform=fig.transFigure, ha='center')
	fig.text(y=0.45, s='$\\rm \Delta G_{^{\\ast} OH}$ (eV)', **text_kw)
	fig.text(y=0.00, s='$\\rm \Delta G_{O^{\\ast}}$ (eV)', **text_kw)
			
	# Write left y-axis labels
	axes[0, 0].set_ylabel('Frequency', fontsize=14)
	axes[1, 0].set_ylabel('Frequency', fontsize=14)

	# Write right y-axis labels
	coverage_axes[0, -1].set_ylabel('Coverage', fontsize=14)
	coverage_axes[1, -1].set_ylabel('Coverage', fontsize=14)

	# Save figure
	print('Saving figure...')
	t0 = time()
	fig.savefig(filename, bbox_inches='tight', dpi=300)
	t1 = time()
	print(f'[SAVED] {filename} ({get_time_stamp(t1-t0)})')
	plt.close(fig)

def get_molar_fraction_color(f, metals):
	'Return color corresponding to the average RGB values of the molar fraction'
	return np.sum(np.array([f0*np.asarray(to_rgb(metal_colors[m])) for f0, m in zip(f, metals)]), axis=0)

def get_composition(f, metals, return_latex=False, saferound=True):
	
	# Make into numpy and convert to atomic percent
	f = np.asarray(f)*100
	
	if saferound:
		# Round while maintaining the sum, the iteround module may need
		# to be installed manually from pypi: "pip3 install iteround"
		import iteround
		f = iteround.saferound(f, 0)
	
	if return_latex:
		# Return string in latex format with numbers as subscripts
		return ''.join(['$\\rm {0}_{{{1}}}$'.format(m,f0) for m,f0 in\
			zip(metals, map('{:.0f}'.format, f)) if float(f0) > 0.])
	else:
		# Return composition as plain text
		return ''.join([''.join([m, f0]) for m,f0 in\
			zip(metals, map('{:.0f}'.format, f)) if float(f0) > 0.])
