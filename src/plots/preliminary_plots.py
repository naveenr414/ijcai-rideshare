import matplotlib.pyplot as plt
import numpy as np

from util import *
from timeplots import *
from pareto import *
from driverside import *
from riderside import *
from fractions import Fraction

data = get_data()

def plot_1():
    # Plot gini after 3 days for each lambda for 8
    labels = []
    first_data = [i for i in data if(i['settings']['value_num'] in [1,2,7,8,9,10])]
    first_data = [i for i in first_data if 'lambda' not in i['settings'] or i['settings']['lambda']>=0]
    first_data = [i for i in first_data if i['settings']['training_days'] == 3 or i['settings']['value_num'] in [2,7,9]]
    first_data = [i for i in first_data if 'pickup_delay' not in i['settings'] and 'add_constraints' not in i['settings']]

    # Compare how gini changes with lambda
    lambda_vals = sorted(list(set([i['settings']['lambda'] for i in first_data if 'lambda' in i['settings'] ])))

    corresponding_gini = []
    corresponding_profit = []
    for k in [8,10]:
        for i in lambda_vals:
            for j in first_data:
                if 'lambda' in j['settings'] and j['settings']['lambda'] == i and j['settings']['value_num'] == k and 'nn_inputs' not in j['settings']:
                    corresponding_gini.append(gini(j))
                    corresponding_profit.append(total_profit(j))
                    if k == 8:
                        labels.append("Entropy {}".format(i))
                    else:
                        labels.append("Variance {}".format(round(i,2)))
    plot_bar(corresponding_gini,labels)
    plt.title("Gini for Neural Income Policies",fontsize=20)
    plt.ylabel("Gini",fontsize=20)
    plt.xlabel("Policy",fontsize=20)
    plt.show()

    plot_bar(corresponding_profit,labels)
    plt.title("Profit for Neural Income Policies",fontsize=20)
    plt.ylabel("Profit",fontsize=20)
    plt.xlabel("Policy",fontsize=20)
    plt.show()
    
def plot_2():
    # Plot two (1,2,7,8)
    second_data = [i for i in data if(i['settings']['value_num'] in [1,2,8,10]
                               and (i['settings']['value_num'] in [2] or i['settings']['training_days']==3)
                               and ('lambda' not in i['settings'] or i['settings']['lambda']>=0))
                                and (i['settings']['value_num'] in [1,2] or 'nn_inputs' in i['settings'])
            and ('pickup_delay' not in i['settings'] and 'add_constraints' not in i['settings'])]

    def sort_by(o):
        if 'lambda' not in o['settings']:
            return o['settings']['value_num']
        return o['settings']['value_num']+o['settings']['lambda']/10000

    second_data = sorted(second_data,key=lambda x:sort_by(x))[::-1]

    names = []
    for i in second_data:
        neural = ""
        if i['settings']['value_num'] == 8:
            names.append("Entropy {}".format(i['settings']['lambda']))
        elif i['settings']['value_num'] == 10:
            
            names.append("Variance {}/3".format(round(i['settings']['lambda']*3)))
        elif i['settings']['value_num'] == 1:
            names.append("NeurADP")
        else:
            names.append("Baseline")



    incomes = [list(payment_by_driver(i).values()) for i in second_data]
    plt.figure(figsize=(10,6))
    plot_income_distro(incomes,names)
    plt.xticks(fontsize=14)
    plt.xlim((2000,5000))
    plt.show()

def plot_25():
    colors = []
    labels = []
    for i in data:
        if i['settings']['value_num'] == 1:
            colors.append('r')
            labels.append("NeurADP")
        elif i['settings']['value_num'] in [7,8]:
            colors.append('b')
            labels.append("Entropy Driver")
        elif i['settings']['value_num'] in [9,10]:
            colors.append('y')
            labels.append("Variance Driver")
        elif i['settings']['value_num'] in [11,12]:
            colors.append('k')
            labels.append("Entropy Rider")
        elif i['settings']['value_num'] in [13,14]:
            colors.append('g')
            labels.append("Variance Rider")
        else:
            colors.append('c')
            labels.append("Other")

    plot_min_income(data,labels,colors)
    plt.show()

def plot_3():
    second_data = [i for i in data if(i['settings']['value_num'] in [1,2,8,10]
                               and (i['settings']['value_num'] in [2] or i['settings']['training_days']==3)
                               and ('lambda' not in i['settings'] or i['settings']['lambda']>=0))
                                and (i['settings']['value_num'] in [1,2] or 'nn_inputs' in i['settings'])
            and ('pickup_delay' not in i['settings'] and 'add_constraints' not in i['settings'])]

    colors = []
    labels = []
    for i in second_data:
        if i['settings']['value_num'] == 1:
            colors.append('r')
            labels.append("NeurADP")
        elif i['settings']['value_num'] in [7,8]:
            colors.append('b')
            labels.append("Entropy Driver")
        elif i['settings']['value_num'] in [9,10]:
            colors.append('y')
            labels.append("Variance Driver")
        elif i['settings']['value_num'] in [11,12]:
            colors.append('k')
            labels.append("Entropy Rider")
        elif i['settings']['value_num'] in [13,14]:
            colors.append('g')
            labels.append("Variance Rider")
        else:
            colors.append('c')
            labels.append("Other")

    plot_min_income(data,labels,colors)
    plt.show()


def plot_4():
    real_data_1 = [i for i in data if i['settings']['value_num'] in [11,12]
                           and i['settings']['lambda']>=0 and (i['settings']['training_days'] == 3 or i['settings']['lambda'] == 50000) and 'nn_inputs' not in i['settings']]
    labels_1 = []
    for i in real_data_1:
        if i['settings']['value_num'] == 1:
            labels_1.append('NeurADP training {}'.format(i['settings']['training_days']))
        elif i['settings']['value_num'] == 11:
            labels_1.append("Non-Neural Entropy, lamb {}".format(i['settings']['lambda']))
        else:
            labels_1.append("Neural Entropy, lamb {} training {}".format(i['settings']['lambda'],i['settings']['training_days']))

    plot_num_min_request(real_data_1,labels_1)
    plt.show()

    real_data_2 = [i for i in data if i['settings']['value_num'] in [11,12]
                           and i['settings']['lambda']>=0 and (i['settings']['training_days'] == 3 or i['settings']['lambda'] == 5*10**8) and 'nn_inputs' not in i['settings']]

    labels_2 = []
    for i in real_data_2:
        if i['settings']['value_num'] == 1:
            labels_2.append('NeurADP training {}'.format(i['settings']['training_days']))
        elif i['settings']['value_num'] == 13:
            labels_2.append("Non-Neural Variance, lamb {}".format(i['settings']['lambda']))
        else:
            labels_2.append("Neural Variance, lamb {} training {}".format(i['settings']['lambda'],i['settings']['training_days']))

    plot_num_min_request(real_data_2,labels_2)
    plt.show()

def plot_5():
    plot_num_min_request(data,['' for i in data])
    plt.show()

def plot_6():
    colors = []
    labels = []
    for i in data:
        if i['settings']['value_num'] == 1:
            colors.append('r')
            labels.append("NeurADP")
        elif i['settings']['value_num'] in [7,8]:
            colors.append('b')
            labels.append("Entropy Driver")
        elif i['settings']['value_num'] in [9,10]:
            colors.append('y')
            labels.append("Variance Driver")
        elif i['settings']['value_num'] in [11,12]:
            colors.append('k')
            labels.append("Entropy Rider")
        elif i['settings']['value_num'] in [13,14]:
            colors.append('g')
            labels.append("Variance Rider")
        else:
            colors.append('c')
            labels.append("Other")

    plot_num_min_request(data,labels,colors)
    plt.show()




plot_6()

