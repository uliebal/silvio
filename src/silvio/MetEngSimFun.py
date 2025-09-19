import cobra
from cobra.io import read_sbml_model
import os
import wget
import numpy as np
import random
import matplotlib.pyplot as plt

def load_model():
    '''
    This function loads the E. coli core model
    args:
    None

    returns:
    model: cobra model
    '''
    # define boths paths where the model can be stored
    ModelFile = os.path.join('..', 'Data', 'e_coli_core.xml.gz')
    model = None

    # load the model from the first path where it is found
    if os.path.isfile(ModelFile):
            print(f'Loading existing file e_coli_core.xml.gz')
            # model = read_sbml_model(ModelFile)
            # return model
    else:
        print('Download of file e_coli_core.xml.gz from BIGG')
        # download the file from BIGG and save it in the `Data` directory
        wget.download('http://bigg.ucsd.edu/static/models/e_coli_core.xml.gz')
        # move the file to the `Data` directory
        os.rename('e_coli_core.xml.gz', ModelFile)
    model = read_sbml_model(ModelFile)
    return model

# substrates and products with carbon
def get_metabolites(model):
    '''
    This function gets the metabolites with carbon from the exchange reactions
    args:
    model: cobra model

    returns:
    metabolites: list of metabolites with carbon
    '''
    metabolites = []
    for rct in model.exchanges:
        for met in rct.metabolites:
            if 'C' in met.formula:
                metabolites.append(met.id)
            # # if met is co2 remove from list --> schönere Lösung finden!
            if met.id == 'co2_e':
                metabolites.remove(met.id)
    return metabolites

# check if reaction has positive product flux
def check_product(model, substrate, product):
    '''
    This function checks if the reaction has positive product flux
    args:
    model: cobra model
    substrate: str, substrate id
    product: str, product id

    returns:
    product: float, product flux
    '''
    with model:
        model.reactions.get_by_id('EX_glc__D_e').bounds = 0, 0
        model.reactions.get_by_id(f'EX_{substrate}').bounds = -10, 0
        # model objective to product
        model.objective = f'EX_{product}'
        product_max = model.slim_optimize()
        return product_max

# function to get sub and pro that fulfill the conditions
def get_sub_pro_pair(model, metabolites):
    '''
    This function gets a pair of metabolites from the exchange reactions with carbon, 
    where the corresponding reactions have positive growth rate and product flux.
    
    args:
    model: cobra model
    metabolites: list of metabolite IDs

    returns:
    tuple: pair of metabolite IDs (met_id1, met_id2) that fulfill the conditions
    '''
    # Select a random pair of metabolites from metabolites
    search_pair = True
    while search_pair:
        selected_pair = random.sample(metabolites, 2)
        # new pair if amount of carbon is the same
        diff_id = model.metabolites.get_by_id(selected_pair[0]).formula != model.metabolites.get_by_id(selected_pair[1]).formula
        # pair with more than 1 Carbon
        more_C = np.min([model.metabolites.get_by_id(selected_pair[0]).elements['C'], model.metabolites.get_by_id(selected_pair[1]).elements['C']]) > 1
        # check if product flux is positive
        product_bool = check_product(model, selected_pair[0], selected_pair[1]) > 0
        # search pair False, if all conditions are met
        if diff_id and more_C and product_bool:
            search_pair = False
        return selected_pair 
    else:
        print("Not enough metabolites found to form a pair.")  # Debugging output
        return None
    
# function to limit one random reaction in selected_pair by setting upper bound lower product flux
def limit_Rct(model, selected_pair, multiplier = 0.5):
    '''
    This function limits one radom reaction by setting the upper bound lower product flux
    args:
    model: cobra model
    selected_pair: list of met ids
    product: float, the product flux
    multiplier: float, the fraction of the product flux

    returns:
    model: cobra model with limited reaction
    '''
    # save reaction ID of all reactions from sub to pro in list
    my_Rct = f'EX_{selected_pair[0]}', f'EX_{selected_pair[1]}'  # Use f-string to get the reaction ID
    model.reactions.get_by_id(my_Rct[0]).lower_bound = -10
    model.reactions.get_by_id('EX_glc__D_e').bounds = 0, 0
    model.objective = my_Rct[1]
    solution = model.optimize()
    Rct = np.where(abs(round(solution.fluxes)) >= round(solution.objective_value))[0]
    EX_idx = [index for index, reaction in enumerate(model.reactions) if 'EX_' in reaction.id]
    valid_rct = list(set(Rct) - set(EX_idx))
    rct = random.choice(valid_rct)
    rct_id = model.reactions[rct].id
    if solution.fluxes[rct] > 0:
        model.reactions[rct].upper_bound = multiplier*solution.fluxes[rct]
    else:
        model.reactions[rct].lower_bound = multiplier*solution.fluxes[rct]
    return model, rct_id

# function to combine the functions above
def make_metabolite_combination(Student_ID):
    '''
    This function selects a pair of substrates and products from exchange metabolites and limits one reaction in the pathway. 
    args:
    Student_ID: int, the student ID
    
    returns:
    model: cobra model with limited reaction
    selected_pair: list of met ids
    product_max: float, maximum ofproduct flux
    product_lim: float, limited product flux
    rct: 
    '''
    random.seed(Student_ID)
    model = load_model()
    metabolites = get_metabolites(model)
    selected_pair = get_sub_pro_pair(model, metabolites)
    # variable with max product flux
    product = round(check_product(model, selected_pair[0], selected_pair[1]),2)
    # divide product_max by itself to get 1
    product_max = product / product
    model, rct = limit_Rct(model, selected_pair)
    # divide new product rate by product to get a value between 0 and 1
    product_lim = [round(model.slim_optimize(),2) / product]
    all_reactions = [f'{selected_pair[0]} --> {selected_pair[1]} limited']
    # product_min = [product_lim]
    return model, selected_pair, all_reactions, product, product_lim, rct

# function to set limited reaction back to original upper bounds in steps by factor 1.5
# beliebig oft durchführbar und wird immer um faktor 1.5 hochgesetzt
def optimize_reaction(model, rct_id, increase_factor = 1.5):
    '''
    This function sets the limited reaction back to the original upper bounds in steps by factor 1.5
    args:
    model: cobra model
    rct: str, reaction ID
    selected_pair: list of met ids

    returns: 
    model: cobra model with bounds set back
    '''
    solution = model.optimize()
    # increase upper bound 
    if solution.fluxes[model.reactions.get_by_id(rct_id).id] > 0:
        model.reactions.get_by_id(rct_id).upper_bound = model.reactions.get_by_id(rct_id).upper_bound * increase_factor
    # relax lower bound 
    else:
        model.reactions.get_by_id(rct_id).lower_bound = model.reactions.get_by_id(rct_id).lower_bound * increase_factor
    return model

# function to create bar chart of substrate-product pair and associated product flux    
def create_bar_chart(all_reactions, product_lim): # target_reaction):
    '''
    This function creates a bar chart of product flux
    args: 
    all_reactions: list of reaction IDs
    product_min: list of product fluxes
    
    returns:
    bar chart
    '''
    idx = range(len(all_reactions))
    plt.bar(idx, product_lim)
    plt.title('Product Flux for Substrate-Product Pair')
    plt.xlabel('Substrate-Product Combination')
    plt.ylabel('Product Flux [mmol/gDW/h]')
    plt.xticks(idx, all_reactions, rotation=90)
    plt.show()