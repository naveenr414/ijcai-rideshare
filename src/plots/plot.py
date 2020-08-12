import matplotlib.pyplot as plt
import numpy as np

from util import *
from timeplots import *
from pareto import *
from driverside import *
from riderside import *

data = get_data()
real_data = [i for i in data if(i['settings']['value_num'] in [1,2,7,8,9,10]
                               and (i['settings']['value_num'] in [2] or i['settings']['training_days']>=2)
                               and ('lambda' not in i['settings'] or i['settings']['lambda']>=0))
                                and (i['settings']['value_num'] in [1,2] or 'nn_inputs' in i['settings'])
            and ('pickup_delay' not in i['settings'])]

plot_std_income(real_data)
plt.show()

