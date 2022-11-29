import sys
sys.path.append('../feature_reader')
from Slab import expand_triangle, Slab, inside_triangle
from FeatureReader import OntopStandard111, FccStandard111
from ase.db import connect
from ase.visualize import view
import numpy as np

# Specify metals
metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
alloy = ''.join(metals)

# Specify name of databases
db_name_OH = 'OH_out.db'
db_name_O = 'O_out.db'

# Set free energies of Pt(111) *OH and O* adsorption
G_OH_Pt111 = 0.76
G_O_Pt111 = 2*0.76

# Initiate dictionary with to convert DFT adsorption energies to free energies
ref = {}

# Load Pt(111) databases
with connect(f'db_1.db') as db_slab,\
	 connect(f'pure_metals_OH_out.db') as db_OH,\
	 connect(f'pure_metals_O_out.db') as db_O:
	
	# Get DFT energies of pure slab, slab with *OH, and slab with O*
	E_slab = db_slab.get('energy', Pt=20, C=0, H=0, O=0).energy
	E_OH = db_OH.get('energy', Pt=20).energy
	E_O = db_O.get('energy', Pt=20).energy
	
	# Set references for each adsorbate
	ref['OH'] = -(E_OH - E_slab) + G_OH_Pt111
	ref['O'] = -(E_O - E_slab) + G_O_Pt111

# Initiate feature readers
reader_OH = OntopStandard111(metals)
reader_O = FccStandard111(metals)

# Initiate counters of rejected samples
rejected_O = 0
rejected_OH = 0

# Specify atom indices of the hollow site that O* is initially put in.
# If O* is outside this hollow site (with some buffer) then it will not be considered
# as it is most likely in a neighboring hcp site
site_ids_O = [16, 17, 18]

# Writer headers to files
with open('OH_features.csv', 'w') as file_OH:
	file_OH.write(f'Features, G_ads (eV), slab db row, {db_name_OH} row')

with open('O_features.csv', 'w') as file_O:
	file_O.write(f'Features, G_ads (eV), slab db row, {db_name_O} row')

# Load HEA(111) databases
with connect(f'{db_name_OH}') as db_OH,\
	 connect(f'{db_name_O}') as db_O,\
	 connect('slabs_out.db') as db_slab,\
	 open('OH_features.csv', 'a') as file_OH,\
	 open('O_features.csv', 'a') as file_O:
	
	# Iterate through slabs without adsorbates
	for row_slab in db_slab.select('energy', H=0, C=0, O=0):
		
		# Iterate through the two adsorbates
		for ads in ['OH', 'O']:
			
			# Set adsorbate-specific parameters
			if ads == 'OH':
				db = db_OH
				kw = {'O': 1, 'H': 1}
				db_name = db_name_OH
				out_file = file_OH
			
			elif ads == 'O':
				db = db_O
				kw = {'O': 1, 'H': 0}
				db_name = db_name_O
				out_file = file_O
			
			# Set counter of matched slabs between the databases to zero
			n_matched = 0
			
			# Get the corresponding slab with adsorbate
			for row in db.select('energy', **kw, **row_slab.count_atoms()):
				
				# If symbols match up
				if row.symbols[:-len(ads)] == row_slab.symbols:
					
					# Increment the counter of matched structures
					n_matched += 1
					
					# Get atoms object
					atoms = db.get_atoms(row.id)
					
					# Make slab instance
					slab = Slab(atoms, ads=ads, ads_atom='O')
					
					# If the adsorbate is *OH
					if ads == 'OH':
						
						# Get adsorption site elements as neighbors within a radius
						site_elems, site = slab.get_adsorption_site(radius=2.6, hollow_radius=2.6)
						
						# If the site does not consist of exactly one atom, then skip this sample
						# as the *OH has moved too far away from an on-top site
						try:
							if len(site_elems) !=1:
								rejected_OH += 1
								#slab.view()
								continue
						except TypeError:
							print(site_elems, site)
							print(row_slab.id, row.id)
							slab.view()
							exit()
							
						# Get features of structure
						features = reader_OH.get_features(slab, radius=2.6)
					
					# Else, if the adsorbate is O*
					elif ads == 'O':
					
						# Get hollow site planar corner coordinates
						site_atoms_pos_orig = atoms.positions[site_ids_O, :2]
						
						# Get expanded triangle vertices
						site_atoms_pos = expand_triangle(site_atoms_pos_orig, expansion=1.45)
						
						# Get position of adsorbate atom (with atom index XXX 20 XXX)
						ads_pos = atoms.positions[20][:2]
						
						# If the oxygen is outside the expanded fcc triangle,
						# then it is most likely in an hcp site, that is not
						# being modeled
						if not inside_triangle(ads_pos, site_atoms_pos):
							rejected_O += 1
							continue
						
						# Get features of structure
						features = reader_O.get_features(slab, radius=2.6, site_ids=site_ids_O)
						
					# Get adsorption energy
					E_ads = row.energy - row_slab.energy + ref[ads]
					
					# Write output to file
					features = ','.join(map(str, features))
					out_file.write(f'\n{features},{E_ads:.6f},{row_slab.id},{row.id}')
				
			# Print a message if more than one slabs were matched. This probably means that
			# the same slab has accidentally been saved multiple to the database
			if n_matched > 1:
				print(f'[INFO] {n_matched} {ads} and slab matched for row {row_slab.id} in {db_name_slab}')
			
			# Print a message if no slabs were matched. This probably means that the DFT calculation
			# did not converge and was left out
			#elif n_matched == 0:
				#print(f'[INFO] No match found in {db_name} for row {row_slab.id} in {db_name_slab}')

# Print the number of rejected samples to screen
print('rejected OH samples: ', rejected_OH)
print('rejected O samples: ', rejected_O)
