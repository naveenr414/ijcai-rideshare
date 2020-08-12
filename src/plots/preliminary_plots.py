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

plot_3()

