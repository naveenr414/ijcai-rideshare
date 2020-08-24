import matplotlib.pyplot as plt
import numpy as np

from util import *
from timeplots import *
from pareto import *
from driverside import *
from riderside import *

data = get_data()

def plot_2():
    # Plot one
    box_whisker_income(data,2000,5000)
    plt.show()

    # Plot two (1,2,7,8)
    second_data = [i for i in data if(i['settings']['value_num'] in [1,2,7,8]
                               and (i['settings']['value_num'] in [2] or i['settings']['training_days']>=2)
                               and ('lambda' not in i['settings'] or i['settings']['lambda']>=0))
                                and (i['settings']['value_num'] in [1,2] or 'nn_inputs' in i['settings'])
            and ('pickup_delay' not in i['settings'])]

    names = []
    for i in second_data:
        neural = ""
        if i['settings']['value_num'] == 8:
            neural = " neural"
        lamb = ""
        if 'lambda' in i['settings']:
            lamb=" lambda: {}".format(i['settings']['lambda'])
        train = ""
        if i['settings']['training_days']>0:
            train = " {}".format(i['settings']['training_days'])
            
        names.append(get_nice_name(i)+neural+lamb)
            


    incomes = [list(payment_by_driver(i).values()) for i in second_data]
    plot_income_distro(incomes,names)
    plt.xlim((2000,5000))
    plt.show()

    # Plot third (1,2,7,8)
    third_data = [i for i in data if(i['settings']['value_num'] in [1,2,9,10]
                               and (i['settings']['value_num'] in [2] or i['settings']['training_days']>=2)
                               and ('lambda' not in i['settings'] or i['settings']['lambda']>=0))
                                and (i['settings']['value_num'] in [1,2] or 'nn_inputs' in i['settings'])
            and ('pickup_delay' not in i['settings'])]

    names = []
    for i in third_data:
        neural = ""
        if i['settings']['value_num'] == 10:
            neural = " neural"
        lamb = ""
        if 'lambda' in i['settings']:
            lamb=" lambda: {}".format(i['settings']['lambda'])
        train = ""
        if i['settings']['training_days']>0:
            train = " {}".format(i['settings']['training_days'])
            
        names.append(get_nice_name(i)+neural+lamb)
            


    incomes = [list(payment_by_driver(i).values()) for i in third_data]
    plot_income_distro(incomes,names)
    plt.xlim((2000,5000))
    plt.show()

def plot_3():
    real_data = [i for i in data if(i['settings']['value_num'] in [1,2,7,8,9,10]
                               and (i['settings']['value_num'] in [2] or i['settings']['training_days']>=2)
                               and ('lambda' not in i['settings'] or i['settings']['lambda']>=0))
                                and (i['settings']['value_num'] in [1,2] or 'nn_inputs' in i['settings'])
            and ('pickup_delay' not in i['settings'])]

    plot_std_income(real_data)
    plt.show()

def plot_4():
    real_data_1 = [i for i in data if i['settings']['value_num'] in [1,11,12]
                           and ('lambda' not in i['settings'] or i['settings']['lambda']>=0)
                   and i['settings']['training_days'] != 1]
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

    real_data_2 = [i for i in data if i['settings']['value_num'] in [1,13,14]
                           and ('lambda' not in i['settings'] or i['settings']['lambda']>=0)
                   and i['settings']['training_days'] != 1]

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

