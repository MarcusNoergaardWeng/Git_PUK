import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import matplotlib.ticker as ticker
import numpy as np
import sys
sys.path.append('../scripts')
from linearmodel_settings import regs_OH, reg_O
from common_settings import sites_OH, sites_O, n_sites_OH, n_sites_O,\
							site_colors, G_OH_lims, G_O_lims

path_DFT = '../DFT_data'

# Load DFT adsorption energies and features for *OH
data = np.loadtxt(f'{path_DFT}/OH_features.csv', delimiter=',', skiprows=1)
features_OH = data[:, :-3].astype(int)
Gs_OH = data[:, -3]

# Load DFT adsorption energies and features for O*
data = np.loadtxt(f'{path_DFT}/O_features.csv', delimiter=',', skiprows=1)
features_O = data[:, :-3].astype(int)
Gs_O = data[:, -3]

calc_train = [Gs_OH, Gs_O]

site_ids_OH = np.nonzero(features_OH[:, :n_sites_OH])[1]
preds_OH = np.zeros_like(Gs_OH)
for site_idx, site in enumerate(sites_OH):

	mask = site_ids_OH == site_idx
	preds_OH[mask] = regs_OH[site].predict(features_OH[mask, n_sites_OH:])

site_ids_O = np.nonzero(features_O[:, :n_sites_O])[1]
preds_O = reg_O.predict(features_O)

pred_train = [preds_OH, preds_O]

fontsize=14
pm=0.1
lw=0.5
filename='parity_plot.png'
adsorbates=['OH', 'O']
calc_test = [[], []]
pred_test = [[], []]
	
# Get the number of plots to make
n_ads = len(adsorbates)

# Make figure
fig, axes = plt.subplots(figsize=(4*n_ads, 3), ncols=n_ads, gridspec_kw=dict(wspace=0.3))

# If a single axis is made, then make it iterable
# for the below loop to work
try:
	ax = iter(axes)
except TypeError:
	axes = [axes]

# Iterate through adsorbates and axes
for ads_idx, (ads, ax, calc_train_, pred_train_, calc_test_, pred_test_, sites, site_ids, lims)\
	in enumerate(zip(adsorbates, axes, calc_train, pred_train, calc_test, pred_test, [sites_OH, sites_O], [site_ids_OH, site_ids_O], [G_OH_lims, G_O_lims])):

	# Set axis labels
	if ads.endswith('H'):
		xlabel = '$\\rm \Delta G_{{\\rm ^{{\\ast}}{0:s}}}^{{\\rm DFT}}$ (eV)'.format(ads)
		ylabel = '$\\rm \Delta G_{{\\rm ^{{\\ast}}{0:s}}}^{{\\rm pred}}$ (eV)'.format(ads)
		
	else:
		xlabel = '$\\rm \Delta G_{{\\rm {0:s}^{{\\ast}}}}^{{\\rm DFT}}$ (eV)'.format(ads) 
		ylabel = '$\\rm \Delta G_{{\\rm {0:s}^{{\\ast}}}}^{{\\rm pred}}$ (eV)'.format(ads)
	
	ax.set_xlabel(xlabel, fontsize=fontsize-2)
	ax.set_ylabel(ylabel, fontsize=fontsize-2)
	
	# Make inset axis showing the prediction error as a histogram
	ax_inset = inset_axes(ax, width=0, height=0)
	margin = 0.015
	scale = 0.85
	width = 0.4*scale
	height = 0.3*scale
	pos = InsetPosition(ax,
		[margin, 1.0-height-margin, width, height])
	ax_inset.set_axes_locator(pos)
	
	# Make plus/minus 0.1 eV lines in inset axis
	ax_inset.axvline(pm, color='black', ls='--', dashes=(5, 5), lw=lw)
	ax_inset.axvline(-pm, color='black', ls='--', dashes=(5, 5), lw=lw)
	
	# Set x-tick label fontsize in inset axis
	ax_inset.tick_params(axis='x', which='major', labelsize=fontsize-6)
	
	# Remove y-ticks in inset axis
	ax_inset.tick_params(axis='y', which='major', left=False, labelleft=False)

	# Set x-tick locations in inset axis		
	ax_inset.xaxis.set_major_locator(ticker.MultipleLocator(0.50))
	ax_inset.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

	# Remove the all but the bottom spines of the inset axis
	for side in ['top', 'right', 'left']:
		ax_inset.spines[side].set_visible(False)
	
	# Make the background transparent in the inset axis
	ax_inset.patch.set_alpha(0.0)
	
	# Print 'pred-calc' below inset axis
	ax_inset.text(0.5, -0.5,
				  '$pred - DFT$ (eV)',
				  ha='center',
				  transform=ax_inset.transAxes,
				  fontsize=fontsize-7)
	
	# Iterate through training and test sets
	for calc, pred, color, y, label, marker, ms in zip([calc_train_, calc_test_], [pred_train_, pred_test_],
																   ['deepskyblue', 'lightcoral'], [0.8, 0.6],
																   ['train', 'test'], ['o', 'X'], [10, 20]):
		
		if len(calc) == 0:
			continue

		# Get the number of data points
		n_samples = len(calc)
		
		for site_idx, site in enumerate(sites):

			mask = site_ids == site_idx
			
			# Make scatter parity plot
			ax.scatter(calc[mask], pred[mask],
					   marker=marker,
					   s=ms,
					   c=[site_colors[site]]*np.sum(mask).astype(int),
					   edgecolors='black',
					   linewidths=0.1,
					   zorder=2)#,
					   #label='{} ({:d} points)'.format(label, n_samples))
		
		# Get prediction errors
		errors = pred - calc
		
		# Make histogram of distribution of errors
		ax_inset.hist(errors,
				 	  bins=np.arange(-0.6, 0.6, 0.05),
				 	  color=color,
				 	  density=True,
				 	  alpha=0.7,
				 	  histtype='stepfilled',
				 	  ec='black',
				 	  lw=lw)
		
		# Print mean absolute error in plot
		mae = np.mean(np.absolute(errors))
		ax_inset.text(0.75, y,
					  'MAE({})={:.3f} eV'.format(label, mae),
					  ha='left',
					  color=color,
					  fontweight='bold',
					  transform=ax_inset.transAxes,
					  fontsize=fontsize-7)
				  
	# Set tick locations
	major_tick = 1.0
	ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25 * major_tick))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(major_tick))
	ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25 * major_tick))
	
	# Format tick labels
	ax.xaxis.set_major_formatter('{x:.1f}')
	ax.yaxis.set_major_formatter('{x:.1f}')
	
	# Set tick label fontsize
	ax.tick_params(axis='both', which='major', labelsize=fontsize-5)
	ax.tick_params(axis='both', which='both', right=True, top=True, direction='in')
	
	# Set x and y limits
	ax.set_xlim(lims)
	ax.set_ylim(lims)
	
	# Make central and plus/minus 0.1 eV lines in scatter plot
	ax.plot(lims, lims,
			lw=lw, color='black', zorder=1,
			label=r'$\rm \Delta G_{pred} = \Delta G_{DFT}$')
	
	# Make plus/minus 0.1 eV lines around y = x
	ax.plot(lims, [lims[0]+pm, lims[1]+pm],
			lw=lw, ls='--', dashes=(5, 5), color='black', zorder=1,
			label=r'$\rm \pm$ {:.1f} eV'.format(pm))
			
	ax.plot([lims[0], lims[1]], [lims[0]-pm, lims[1]-pm],
			lw=lw, ls='--', dashes=(5, 5), color='black', zorder=1)
	
	# Make legend
	ax.legend(frameon=False,
			  bbox_to_anchor=[0.45, 0.0],
			  loc='lower left',
			  handletextpad=0.2,
			  handlelength=1.0,
			  labelspacing=0.2,
			  borderaxespad=0.1,
			  markerscale=1.5,
			  fontsize=fontsize-5)

fig.savefig(filename, dpi=300, bbox_inches='tight')
