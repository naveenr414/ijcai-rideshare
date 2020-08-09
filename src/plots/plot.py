import matplotlib.pyplot as plt
import numpy as np

from util import *
from timeplots import *
from pareto import *
from driverside import *
from riderside import *
    
data = get_data()
plot_income_distro([[1,2,3,4],[1,3,4,5]],['a','b'])
plt.show()
