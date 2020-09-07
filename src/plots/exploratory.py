import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import *

def plot_pca(a):
    all_data = deepcopy(a)
    # We'll let value num 3 = hard constraint rider
    # And value num 5 = hard constraint driver

    all_data = [i for i in all_data if not('lambda' in i['settings'] and i['settings']['lambda']<0)]

    for i in all_data:
        if 'pickup_delay' in i['settings']:
            i['settings']['value_num'] = 3
        elif 'add_constraints' in i['settings'] and i['settings']['add_constraints'] == 'min':
            i['settings']['value_num'] = 5
    
    all_coords = [get_coords(i)+[i['settings']['value_num']] for i in all_data]
    pca = PCA(n_components=2)
    data_df = pd.DataFrame(all_coords,
                           columns=['profit','gini','delay','request completion','location distro','value num'])

    
    principalComponents = pca.fit_transform(data_df)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    
    finalDf = pd.concat([principalDf, data_df[['value num']]], axis = 1)

    targets = [1,2,4,3,5,7,8,9,10,11,12,13,14]
    
    colors = ['b','b','b','g','r','y','y','m','m','k','k','c','c']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['value num'] == target
        plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    plt.legend(targets)

    print("Explained variance {}".format(pca.explained_variance_ratio_))


def move_from_unknown():
    all_files = glob.glob("../logs/epoch_data/unknown/*.pkl")
    all_pickles = [get_pickle(i) for i in all_files]

    for i in range(len(all_pickles)):
        settings = all_pickles[i]["settings"]
        file_name = all_files[i].split("\\")[-1]
        if settings["value_num"] in [7,8]:
            print("mv {} ../entropy".format(file_name))
        elif settings["value_num"] in [9,10]:
            print("mv {} ../variance".format(file_name))
        elif settings["value_num"] == 2 and "add_constraints" in settings and settings["add_constraints"] == "min":
            print("mv {} ../income_hard".format(file_name))
        elif settings["value_num"] in [1,2,4] and "add_constraints" not in settings and "pickup_delay" not in settings:
            print("mv {} ../baseline".format(file_name))
        elif "pickup_delay" in settings:
            print("mv {} ../waittime".format(file_name))
        elif settings["value_num"] in [11,12]:
            print("mv {} ../entropy_rider".format(file_name))
        elif settings["value_num"] in [13,14]:
            print("mv {} ../variance_rider".format(file_name))

plot_over_newyork()
