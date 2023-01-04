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
    return small_idx_x, small_idx_y, smallest_energy, adsorbate

def count_adsorbates(adsorbates_hollow, adsorbates_on_top):
    unique, counts = np.unique(np.concatenate((adsorbates_hollow, adsorbates_on_top)), return_counts=True)
    return dict(zip(unique, counts))





















