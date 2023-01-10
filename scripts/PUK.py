# This is a library of functions used for my Project in Practice about Cyclic Voltammograms

# Imports
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import copy

# Functions

# Make a 100x100x3 surface with an even distribution of the five metals
def simulate_surface(dim_x, dim_y): #Is still random - could be used with a seed in the name of reproduceability
    #dim_x, dim_y, dim_z = 100, 100, 3 #Specify dimensions
    dim_z = 3
    surface_list = np.array([int(dim_x*dim_y*dim_z/len(metals))*[metals[metal_number]] for metal_number in range(len(metals))]).flatten() #Jack had a way shorter way of doing this, but I think it was random drawing instead of ensuring a perfectly even split
    np.random.shuffle(surface_list) #Shuffle list
    surface = np.reshape(surface_list, (dim_x, dim_y, dim_z)) #Reshape list to the
    return surface

def on_top_site_vector(surface, site_x, site_y):
    
    site1 = [surface[site_x, site_y, 0]]# Make a one-hot encoded vector of the very site here! Add at the beginning
    site1_count = [site1.count(metals[n]) for n in range(len(metals))]
    
    top6 = [surface[site_x % 100, (site_y-1) % 100, 0], surface[site_x % 100, (site_y+1) % 100, 0], surface[(site_x-1) % 100, site_y % 100, 0], surface[(site_x+1) % 100, site_y % 100, 0], surface[(site_x-1) % 100, (site_y+1) % 100, 0], surface[(site_x+1) % 100, (site_y-1) % 100, 0]]
    top6_count = [top6.count(metals[n]) for n in range(len(metals))]
    
    mid3 = [surface[(site_x-1) % 100, (site_y-1) % 100,1], surface[site_x % 100, (site_y-1) % 100,1], surface[(site_x-1) % 100, site_y % 100,1]]
    mid3_count = [mid3.count(metals[n]) for n in range(len(metals))]
    
    bot3 = [surface[(site_x-1) % 100, (site_y-1) % 100, 2], surface[(site_x-1) % 100, (site_y+1) % 100, 2], surface[(site_x+1) % 100, (site_y-1) % 100, 2]]
    bot3_count = [bot3.count(metals[n]) for n in range(len(metals))]
    
    return site1_count + top6_count + mid3_count + bot3_count

metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
three_metals_combinations = [] #List of possible combinations of the three
# Der skal være 35, ikke 125

for a in metals:
    for b in metals:
        for c in metals:
            three_metals_combinations.append(''.join(sorted([a, b, c])))
            
# Remove duplicates
three_metals_combinations = list(dict.fromkeys(three_metals_combinations)) # Let's encode it in a better way later

def hollow_site_vector(surface, site_x, site_y, adsorbate): #Now with adsorbate encoding
    
    if adsorbate == "H":
        ads = [0]
    elif adsorbate == "O":
        ads = [1]
    
    # First encode the 3 neighbours
    blues = [surface[(site_x+1) % 100, site_y, 0], surface[site_x, (site_y+1) % 100, 0], surface[(site_x+1) % 100, (site_y+1) % 100, 0]]
    blues = "".join(sorted(blues))
    idx = three_metals_combinations.index(blues)
    blues = 35*[0]
    blues[idx] = 1
    
    # Then the next neighbours (green)
    greens = [surface[(site_x+2) % 100, site_y, 0], surface[site_x, (site_y+2) % 100, 0], surface[site_x, site_y, 0]]
    greens_count = [greens.count(metals[n]) for n in range(len(metals))]
    
    # Then the next neighbours (brown) # Kunne gøres smartere med list comprehension og to lister med +- zipped
    browns = [surface[(site_x + a) % 100, (site_y + b) % 100, c] for a, b, c in zip([1, 2, 2, 1, -1, -1], [2, 1, -1, -1, 1, 2], [0, 0, 0, 0, 0, 0])]
    browns_count = [browns.count(metals[n]) for n in range(len(metals))]
    
    # Then the three downstairs neighbours
    yellows = [surface[(site_x + a) % 100, (site_y + b) % 100, c] for a, b, c in zip([0, 1, 0], [0, 0, 1], [1, 1, 1])]
    yellows_count = [yellows.count(metals[n]) for n in range(len(metals))]
    
    # Then the purples downstairs
    purples = [surface[(site_x + a) % 100, (site_y + b) % 100, c] for a, b, c in zip([1, -1, 1], [-1, 1, 1], [1, 1, 1])]
    purples_count = [purples.count(metals[n]) for n in range(len(metals))]
    
    return ads + blues + greens_count + browns_count + yellows_count + purples_count

def pandas_to_DMatrix(df):
    label = pd.DataFrame(np.random.randint(2, size=len(df)))
    DMatrix = xgb.DMatrix(df, label=label)
    return DMatrix

def pandas_to_DMatrix(df):#, label):
    label = pd.DataFrame(np.random.randint(2, size=len(df)))
    DMatrix = xgb.DMatrix(df)#, label=label)
    return DMatrix

# Surface to energies #prob the hardest one
def surface_to_energies(surface, on_top_site_model, hollow_site_model):
    t1 = time.time()
    dim_x, dim_y = np.shape(surface)[0], np.shape(surface)[1]
    # Find all sites in x,y
    sites_list = []
    for site_x in range(dim_x): #Looping through all on top sites
        for site_y in range(dim_y):
            sites_list.append([site_x, site_y])

    # Make all feature vectors for all on-top and hollow sites
    all_on_top_OH_df = pd.DataFrame([on_top_site_vector(surface, site_x, site_y) for site_x, site_y in sites_list], columns = [f"feature{n}" for n in range(20)])
    all_hollow_H_df  = pd.DataFrame([hollow_site_vector(surface, site_x, site_y, adsorbate = "H") for site_x, site_y in sites_list], columns = [f"feature{n}" for n in range(56)])
    all_hollow_O_df  = pd.DataFrame([hollow_site_vector(surface, site_x, site_y, adsorbate = "O") for site_x, site_y in sites_list], columns = [f"feature{n}" for n in range(56)])
    # I forgot to add 0 for H and 1 for O in front of the vector
    all_on_top_OH_DM = pandas_to_DMatrix(all_on_top_OH_df)
    all_hollow_H_DM  = pandas_to_DMatrix(all_hollow_H_df)
    all_hollow_O_DM  = pandas_to_DMatrix(all_hollow_O_df)
    
    ## Predict energies of all sites
    OH_Energies = on_top_site_model.predict(all_on_top_OH_DM)
    H_Energies = hollow_site_model.predict(all_hollow_H_DM)
    O_Energies = hollow_site_model.predict(all_hollow_O_DM)
    All_Energies = np.concatenate((OH_Energies, H_Energies, O_Energies))
    print(f"I predicted the energy of every adsorbate on every feasible site on a {dim_x}x{dim_y} surface, resulting in {dim_x*dim_y*3} energies. It took {time.time() - t1:.2f} seconds")
    
    all_energies = {"H": H_Energies, "OH": OH_Energies, "O": O_Energies}
    return all_energies, H_Energies, OH_Energies, O_Energies

# Energies to Charge isoterm - going to be quite big
def energies_to_charge_isoterm(all_energies): #In order to process more adsorbates just add them in the dicts and energy list
    # Number of slope PD's
    sum_diff_pd = np.zeros(1000) #I have no idea why I set it to 10.000 initially. It seems excessive
    slopes = {"H": 1, "OH": -1, "O": -2}
    adsorbates = ["H", "OH", "O"]
    
    for adsorbate in adsorbates:
        slope = slopes[adsorbate]
        energies = all_energies[adsorbate]
        
        for single_energy in tqdm(energies):
    
            V_cut = single_energy# / elementary_charge # Find the V, where the energy hits 0 # Maybe the elementary charge shouldn't be there unless the delta G's are also corrected. They arent truly at just around 0 to 3. Maybe they should be in electron volts. Ask Jack about the units:)
            V_list = np.linspace(-0.25, 2.9, 1000)
            if adsorbate == "H":
                diff_PD = [slopes[adsorbate] if V < V_cut else 0 for V in V_list]
            if adsorbate == "OH" or adsorbate == "O":
                diff_PD = [slopes[adsorbate] if V > V_cut else 0 for V in V_list]
            sum_diff_pd += diff_PD
    return V_list, sum_diff_pd

# Charge isoterm to CV
def charge_isoterm_to_CV(charge_isoterm):
    return -np.gradient(charge_isoterm)

# Smoothing
def rolling_average_smoothing(data, kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')
    
# All combined
def surface_to_CV(surface, on_top_site_model, hollow_site_model):
    all_energies, H_Energies, OH_Energies, O_Energies = surface_to_energies(surface, on_top_site_model, hollow_site_model)
    V_list, charge_isoterm = energies_to_charge_isoterm(all_energies)
    CV = charge_isoterm_to_CV(charge_isoterm)
    CV_smooth = rolling_average_smoothing(CV, 5)
    return V_list, all_energies, H_Energies, OH_Energies, O_Energies, charge_isoterm, CV, CV_smooth

# Make surface composition dictionary and chemical composition string for matplotlib
def surface_composition(metals, surface):
    composition = {}
    num_atoms = np.shape(surface)[0]*np.shape(surface)[1]*np.shape(surface)[2]
    
    for metal in metals:
        composition[metal] = np.count_nonzero(surface == metal) / num_atoms
        
    # Lav en kemisk formel for overfladen og lav subskript med deres fractions
    composition_string = ""

    for metal in metals:
        if composition[metal] > 0:
            composition_string += str(metal) + f"$_{{{composition[metal]:.2f}}}$"
    return composition, composition_string

# Plot charge isoterm + CV
def super_surface_plot(metals, surface, H_Energies, OH_Energies, O_Energies, All_Energies, V_list, sum_diff_pd, movavg_CV, **filename):

    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize = (14, 10))
    
    ### SURFACE COMPOSITION ###
    composition, composition_string = surface_composition(metals, surface)
    ymax = max(composition.values()) * 1.1
    ax_a.set_ylim(0, ymax)
    ax_a.set_title("Surface composition: " + composition_string)
    ax_a.set_yticks([])
    #ax.set_xticks([])
    bar1 = ax_a.bar(composition.keys(), composition.values(), alpha = 0.9, color = ["silver", "gold", "darkorange", "lightsteelblue", "cornflowerblue"])
    
    for idx, rect in enumerate(bar1):
        height = rect.get_height()
        fraction = list(composition.values())[idx]*100
        ax_a.text(rect.get_x() + rect.get_width() / 2.0, height, s = f"{fraction:.2f}" + " %", ha='center', va='bottom')
    
    ### ENERGY HISTOGRAMS ###
    # Plot histograms
    ax_b.hist(H_Energies , bins=np.arange(-0.2, 2.8, 0.019), ec = "black", color = "tab:green", histtype = "stepfilled", density=False, alpha=0.7,label = "H")
    ax_b.hist(OH_Energies, bins=np.arange(-0.2, 2.8, 0.019), ec = "black", color = "tab:blue" , histtype = "stepfilled", density=False, alpha=0.7,label = "OH")
    ax_b.hist(O_Energies , bins=np.arange(-0.2, 2.8, 0.019), ec = "black", color = "tab:red"  , histtype = "stepfilled", density=False, alpha=0.7,label = "O")
    ax_b.hist(All_Energies , bins=np.arange(-0.2, 2.8, 0.019), ec = "black", color = "tab:red"  , histtype = "step", density=False, alpha=1.0,label = "All adsorbates")
    
    ax_b.legend()
    
    ax_b.set_title("Predicted $\Delta G_{*Adsorbate}^{Pred} (eV)$ for all sites and three adsorbates\n on simulated 100x100 HEA surface")
    ax_b.set_xlabel("$\Delta G_{Adsorbate}^{Pred} (eV)$")
    ax_b.set_ylabel("Frequency")
    
    ax_b.tick_params(axis='y', which='major', left=False, labelleft=False)
    
    # Make text at each peak to show the reaction happening
    if False:
        ax_b.text(x, y, "$H^{}$")
    
    ### CHARGE ISOTERM ###    
        
    ax_c.set_title("Charge isoterm (summed slopes of phase diagrams)")
    ax_c.scatter(V_list, sum_diff_pd, c = "black", s = 1)
    ax_c.set_ylabel(r"$\frac{dE}{dV}$ (eV)")
    ax_c.set_xlabel("V")
    ax_c.set_yticks([])
        
    ### VOLTAMMOGRAM ###
    kernel_size = 5
    ax_d.set_title(f"Voltammogram (smoothed, kernel size: {kernel_size})")
    ax_d.plot(V_list, movavg_CV, c = "black") #, s = 5
    ax_d.set_yticks([])
    ax_d.set_ylabel("Current density")
    ax_d.set_xlabel("V")
    
    if filename:
        #print("The filename is: ", filename["filename"])
        fig.savefig("../figures/Multiplots/"+filename["filename"]+".png", dpi = 300, bbox_inches = "tight")
    fig.show()
    return None

def create_surface(metals, dim_x, dim_y, split):
    dim_z = 3
    num_atoms = dim_x*dim_y*dim_z
    if split == "Even":
        proba = [1.0 / len(metals) for n in range(len(metals))] 
        surface = np.random.choice(metals, num_atoms, p=proba)
    else:
        surface = np.random.choice(metals, num_atoms, p=split)
    surface = np.reshape(surface, (dim_x, dim_y, dim_z)) #Reshape list to the
    return surface

def plot_voltammograms(V_list, voltammograms_list):
    metal_colours = ["silver", "gold", "darkorange", "lightsteelblue", "cornflowerblue"]
    fig, ax = plt.subplots(1, 1, figsize = (7, 5))
    for idx, voltammogram in enumerate(voltammograms_list):
        ax.plot(V_list, voltammogram, c = metal_colours[idx], linewidth = 2, label = f"Excluded metal: {metals[idx]}")
    
    ### VOLTAMMOGRAM ###
    kernel_size = 5
    ax.set_title(f"Voltammogram (smoothed, kernel size: 5)")
    ax.set_yticks([])
    ax.set_ylabel("Current density")
    ax.set_xlabel("V")
    ax.legend()
    #plt.savefig("../figures/CV/combined_voltammogram.png", dpi = 300, bbox_inches = "tight")
    fig.show()
    return

def create_empty_adsorbates(dim_x, dim_y):
    adsorbates_hollow = np.reshape(["empty" for n in range(dim_x*dim_y)], (dim_x, dim_y))
    adsorbates_on_top = np.reshape(["empty" for n in range(dim_x*dim_y)], (dim_x, dim_y))
    return adsorbates_hollow, adsorbates_on_top

def surface_to_energies_matrices(surface, potential, on_top_site_model, hollow_site_model):
    t1 = time.time()
    dim_x, dim_y = np.shape(surface)[0], np.shape(surface)[1]
    # Find all sites in x,y
    sites_list = []
    for site_x in range(dim_x): #Looping through all on top sites
        for site_y in range(dim_y):
            sites_list.append([site_x, site_y])

    # Make all feature vectors for all on-top and hollow sites
    all_on_top_OH_df = pd.DataFrame([on_top_site_vector(surface, site_x, site_y) for site_x, site_y in sites_list], columns = [f"feature{n}" for n in range(20)])
    all_hollow_H_df  = pd.DataFrame([hollow_site_vector(surface, site_x, site_y, adsorbate = "H") for site_x, site_y in sites_list], columns = [f"feature{n}" for n in range(56)])
    all_hollow_O_df  = pd.DataFrame([hollow_site_vector(surface, site_x, site_y, adsorbate = "O") for site_x, site_y in sites_list], columns = [f"feature{n}" for n in range(56)])
    # I forgot to add 0 for H and 1 for O in front of the vector
    all_on_top_OH_DM = pandas_to_DMatrix(all_on_top_OH_df)
    all_hollow_H_DM  = pandas_to_DMatrix(all_hollow_H_df)
    all_hollow_O_DM  = pandas_to_DMatrix(all_hollow_O_df)
    
    ## Predict energies of all sites
    OH_Energies= on_top_site_model.predict(all_on_top_OH_DM)
    H_Energies = hollow_site_model.predict(all_hollow_H_DM)
    O_Energies = hollow_site_model.predict(all_hollow_O_DM)
    
    ## The potential affects the delta G for each adsorbate
    H_Energies += 1 * potential
    OH_Energies -= 1 * potential
    O_Energies -= 2 * potential # Skal O kunne sætte sig direkte eller skal den først laves om fra OH?
    
    #print(f"I predicted the energy of every adsorbate on every feasible site on a {dim_x}x{dim_y} surface, resulting in {dim_x*dim_y*3} energies. It took {time.time() - t1:.2f} seconds")

    #all_energies = {"H": H_Energies, "OH": OH_Energies, "O": O_Energies}
    return np.reshape(H_Energies, (dim_x, dim_y)), np.reshape(OH_Energies, (dim_x, dim_y)), np.reshape(O_Energies, (dim_x, dim_y))

def argmin2d(A):
    return np.unravel_index(A.argmin(), A.shape)

def find_smallest_energy(Energies_matrix):
    smallest_energy = 10000
    smallest_adsorbate = "bong"
    for adsorbate in ["H", "OH", "O"]:
        energy = np.min(Energies_matrix[adsorbate])
        x, y = argmin2d(Energies_matrix[adsorbate])
        if energy < smallest_energy:
            #print("I found an energy smaller than 100")
            smallest_energy = energy
            smallest_adsorbate = adsorbate
            small_idx_x = x
            small_idx_y = y
    return small_idx_x, small_idx_y, smallest_energy, smallest_adsorbate

def count_adsorbates(adsorbates_hollow, adsorbates_on_top):
    unique, counts = np.unique(np.concatenate((adsorbates_hollow, adsorbates_on_top)), return_counts=True)
    return dict(zip(unique, counts))

def count_adsorbates_full(adsorbates_hollow, adsorbates_on_top):
    adsorbates_all = np.concatenate((adsorbates_hollow, adsorbates_on_top))
    adsorbate_dict = {"H": 0, "OH": 0, "O": 0, "empty": 0}
    for adsorbate in ["H", "OH", "O", "empty"]:
        adsorbate_dict[adsorbate] = np.count_nonzero(adsorbates_all == adsorbate)
    return adsorbate_dict

def make_voltage_range(V_start, V_max, V_min, num_rounds, points_per_round):
    # Make a single round first
    #Figure out the ratios
    total_length = 2*(V_max-V_min)
    
    a = list(np.linspace(V_start, V_max, int(points_per_round * (V_max-V_start) / total_length)))
    b = list(np.linspace(V_max, V_min, int(points_per_round * (V_max-V_min) / total_length)))
    c = list(np.linspace(V_min, V_start, int(points_per_round * (V_start-V_min) / total_length)))
    d = a+b+c
    d *= num_rounds
    return d

def steric_hindrance(steric_bonus_energy, idx_x, idx_y, dim_x, dim_y, adsorbate, adsorbates_hollow, adsorbates_on_top):
    
    # Hvilken type site sidder adsorbatet på?
    if adsorbate == "OH":
        site_type = "on-top"
    else:
        site_type = "hollow"
    
    if site_type == "on-top":
        # Tæl naboer på samme type site (der er 6 af dem) TJEK LAV MODULO
        on_top_neighbours = [adsorbates_on_top[(idx_x+site_x)%dim_x, (idx_y+site_y)%dim_y] for site_x, site_y in zip([-1, 1, 0, 0, -1, 1], [0, 0, -1, 1, 1, -1])] #Lav en zip-ting der list comprehender sites of interest så vi får en liste med de steders indhold. Ligesom med vektorerne. Kig der.
        # Tæl naboer på den anden type site (der er 3 af dem)
        hollow_neighbours = [adsorbates_hollow[(idx_x+site_x)%dim_x, (idx_y+site_y)%dim_y] for site_x, site_y in zip([0, -1, 0], [0, 0, -1])]
        
    else: # Hollow site
        # Tæl naboer på samme type site (der er 6 af dem)
        hollow_neighbours = [adsorbates_hollow[(idx_x+site_x)%dim_x, (idx_y+site_y)%dim_y] for site_x, site_y in zip([-1, 1, 0, 0, -1, 1], [0, 0, -1, 1, 1, -1])] #Lav en zip-ting der list comprehender sites of interest så vi får en liste med de steders indhold. Ligesom med vektorerne. Kig der.
        # Tæl naboer på den anden type site (der er 3 af dem)
        on_top_neighbours = [adsorbates_on_top[(idx_x+site_x)%dim_x, (idx_y+site_y)%dim_y] for site_x, site_y in zip([0, 1, 0], [0, 0, 1])]
    
    # Combine and count
    all_neighbours = on_top_neighbours + hollow_neighbours
    neighbours = 9 - all_neighbours.count("empty")
    steric_energy = neighbours * steric_bonus_energy
    return steric_energy

def plot_CV(metals, surface, voltage_range, sum_delta_G_log, adsorbates_log, steric_bonus_energy, hysteresis_threshold, **filename):
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize = (18, 5))
    
    ### TITLE ###
    fig.suptitle(f"Simulated cyclic voltammogram \n steric hindrance parameter: {steric_bonus_energy} (eV), hysteresis barrier {hysteresis_threshold} (eV) \n", x = 0.5, y = 1.05,fontsize = 18)
    
    ### SURFACE COMPOSITION ###
    composition, composition_string = surface_composition(metals, surface)
    ymax = max(composition.values()) * 1.1
    ax_a.set_ylim(0, ymax)
    ax_a.set_title("Surface composition: " + composition_string)
    ax_a.set_yticks([])
    #ax.set_xticks([])
    bar1 = ax_a.bar(composition.keys(), composition.values(), alpha = 0.9, color = ["silver", "gold", "darkorange", "lightsteelblue", "cornflowerblue"])
    
    for idx, rect in enumerate(bar1):
        height = rect.get_height()
        fraction = list(composition.values())[idx]*100
        ax_a.text(rect.get_x() + rect.get_width() / 2.0, height, s = f"{fraction:.2f}" + " %", ha='center', va='bottom')
        
    ### ADSORBATES ON SURFACE ###
    adsorbate_colours = {"H": "tab:green", "OH": "tab:blue", "O": "tab:red", "empty": "Grey"}
    for adsorbate in ["H", "OH", "O"]:
        ax_b.plot(voltage_range, [adsorbates_log[n][adsorbate]/100 for n in range(len(adsorbates_log))], c = adsorbate_colours[adsorbate], label = adsorbate)
    
    ax_b.set_title("Adsorbates on the surface sites")
    ax_b.set_xlabel("Voltage (V)")
    ax_b.set_ylabel("Occupation (%)")
    
    ax_b.legend()
    
    ### CYCLIC VOLTAMMOGRAM ###
    
    ax_c.plot(voltage_range, -np.gradient(np.gradient(sum_delta_G_log)))
    
    ax_c.set_title("Cyclic voltammogram")
    ax_c.set_xlabel("Voltage (V)")
    ax_c.set_yticks([])
    
    if filename:
        #print("The filename is: ", filename["filename"])
        fig.savefig("../figures/Sim_CV/"+filename["filename"]+".png", dpi = 300, bbox_inches = "tight")
    fig.show()
    return None


def simulate_CV(surface, voltage_range, hysteresis_threshold, steric_bonus_energy, dim_x, dim_y, adsorbates_hollow, adsorbates_on_top, hollow_site_model, on_top_site_model):
    adsorbates_log = []
    sum_delta_G_log = []
    
    for potential in tqdm(voltage_range):
        ### Print the voltage ###
        #print(f"Potential: {potential:.2f} V")
        
        ### Calculate all energies on the surface for all adsorbates ###
        H_matrix, OH_matrix, O_matrix = surface_to_energies_matrices(surface, potential, on_top_site_model, hollow_site_model)
        Energies_matrix = {"H": H_matrix, "OH": OH_matrix, "O": O_matrix}
        Energies_matrix_count = copy.deepcopy(Energies_matrix)
        ### REMOVE ADSORBATES WITH POSITIVE ENERGY ###
        # Look through all adsorbate sites (hollow and on-top)
        # If occupied, remove at positive energy (or above hysteresis barrier)
        for idx_x in range(dim_x):
            for idx_y in range(dim_y):
                
                ### REMOVE HOLLOW SITE ADSORBATES ###
                hollow_adsorbate = adsorbates_hollow[idx_x, idx_y] # Find the adsorbate
                if hollow_adsorbate != "empty": # If the spot is not empty
                    hollow_adsorbate_energy = Energies_matrix[hollow_adsorbate][idx_x, idx_y] #Check the energy at that spot with that adsorbate
                    
                    # Korriger for nabo-interaktioner? Ja. Læg noget til energien baseret på antallet af naboer
                    hollow_adsorbate_energy += steric_hindrance(steric_bonus_energy, idx_x, idx_y, dim_x, dim_y, "H", adsorbates_hollow, adsorbates_on_top)
                    
                    if hollow_adsorbate_energy > hysteresis_threshold: # Hysteresis could be inserted here
                        #print(f"Removing adsorbate {hollow_adsorbate} because the energy is {hollow_adsorbate_energy}")
                        adsorbates_hollow[idx_x, idx_y] = "empty" # The adsorbate is removed
                        #print(f"The hollow_adsorbate at this place is now {adsorbates_hollow[idx_x, idx_y]}")
                
                ### REMOVE ON-TOP SITE ADSORBATES ###
                on_top_adsorbate = adsorbates_on_top[idx_x, idx_y] # Find the adsorbate
                if on_top_adsorbate != "empty": # If the spot is not empty
                    on_top_adsorbate_energy = Energies_matrix[on_top_adsorbate][idx_x, idx_y] #Check the energy at that spot with that adsorbate
                    
                    # Korriger for nabo-interaktioner?
                    on_top_adsorbate_energy += steric_hindrance(steric_bonus_energy, idx_x, idx_y, dim_x, dim_y, "OH", adsorbates_hollow, adsorbates_on_top)
                    
                    if on_top_adsorbate_energy > hysteresis_threshold: # Hysteresis could be inserted here
                        adsorbates_on_top[idx_x, idx_y] = "empty" # The adsorbate is removed
        
        
        # Kig igennem energierne ved de frie sites, læg nabo-interaktion-energier til, hvis energien stadig er 
        # negativ, så sæt den på. Ellers, så erstat energien med et "no", så man ikke finder samme energi igen
        
        # Find smallest energy in all three matrices
        smallest_energy = -1 #Just initially to get the while loop going
        # The absolute maximum number of smallest energies is dim_x * dim_y * 3. One energy is checked, if under 0 AND the spot is free -> attach the adsorbate, then set the energy to 100. If the smallest energy is not under 0 -> Exit the loop  
        #for iteration in range(dim_x*dim_y*3): #Maksimalt antal energier der skal tjekkes hvis alle er negative. Hvis man overskrider den her er der noget galt
        counter = 0
        while smallest_energy < 0:
            counter += 1
            if counter > dim_x*dim_y*3:
                raise ValueError("Exceeded the maxmimum amount of negative energies. Some energies keep getting found again and again. Make sure all checked energies are set to 100.")
            idx_x, idx_y, smallest_energy, adsorbate = find_smallest_energy(Energies_matrix)
            
            #Check if the position is occupied:
            if ((adsorbate == "H" or adsorbate == "O") and adsorbates_hollow[idx_x, idx_y] != "empty") or (adsorbate == "OH" and adsorbates_on_top[idx_x, idx_y] != "empty"):
                # The position was occupied. Remove the energies at this position
                if adsorbate == "H" or adsorbate == "O":
                    Energies_matrix["H"][idx_x, idx_y] = 100
                    Energies_matrix["O"][idx_x, idx_y] = 100
                
                if adsorbate == "OH":
                    Energies_matrix["OH"][idx_x, idx_y] = 100
                continue # Ahh, hvis pladsen der hører til den laveste energi er optaget, så sæt energien på denne plads FOR SAMME ADSORBAT til 100. Så skal man ikke tjekke senere. Ellers kunne man også bare tjekke om pladsen er ledig for hver energi. Det burde være simplere
            
            # Add neighbor-interaction
            smallest_energy += steric_hindrance(steric_bonus_energy, idx_x, idx_y, dim_x, dim_y, adsorbate, adsorbates_hollow, adsorbates_on_top)
            
            # Remove that energy and put "checked" instead no matter what happened before
            if smallest_energy < 0: # ADSORPTION
                if adsorbate == "H" or adsorbate == "O":
                    # Put the adsorbate on the adsorbates matrix
                    adsorbates_hollow[idx_x, idx_y] = adsorbate
                    
                    # Remove the energies at this position
                    Energies_matrix["H"][idx_x, idx_y] = 100
                    Energies_matrix["O"][idx_x, idx_y] = 100
                
                if adsorbate == "OH":
                    adsorbates_on_top[idx_x, idx_y] = adsorbate
                    # Remove the energies at this position
                    Energies_matrix["OH"][idx_x, idx_y] = 100
        
        
        ### SUM ADSORBATE ENERGIES ### 
        
        # Loop through all sites, add the energy of the adsorbate if one is found
        sum_delta_G = 0
        for idx_x in range(dim_x): #Perhaps I should add the 
            for idx_y in range(dim_y):
                # HOLLOW SITES
                if adsorbates_hollow[idx_x, idx_y] == "H":
                    sum_delta_G += Energies_matrix_count["H"][idx_x, idx_y]
                    sum_delta_G += steric_hindrance(steric_bonus_energy, idx_x, idx_y, dim_x, dim_y, "H", adsorbates_hollow, adsorbates_on_top)
                    
                if adsorbates_hollow[idx_x, idx_y] == "O":
                    sum_delta_G += Energies_matrix_count["O"][idx_x, idx_y]
                    sum_delta_G += steric_hindrance(steric_bonus_energy, idx_x, idx_y, dim_x, dim_y, "O", adsorbates_hollow, adsorbates_on_top)
                    
                if adsorbates_on_top[idx_x, idx_y] == "OH":
                    sum_delta_G += Energies_matrix_count["OH"][idx_x, idx_y]
                    sum_delta_G += steric_hindrance(steric_bonus_energy, idx_x, idx_y, dim_x, dim_y, "OH", adsorbates_hollow, adsorbates_on_top)
                    
        sum_delta_G_log.append(sum_delta_G / (dim_x*dim_y))
        
        adsorbates_log.append(count_adsorbates_full(adsorbates_hollow, adsorbates_on_top))
    
    return sum_delta_G_log, adsorbates_log
















