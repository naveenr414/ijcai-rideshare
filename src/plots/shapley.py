from util import *
from driverside import *
import matplotlib.pyplot as plt

data = get_shapley()

def plot_1():
    for i in data:
        if i['settings']['value_num'] == 2:
            labels = ['random_shapley','truncated_shapley','one_permutation_shapley']
            shapley = [i[j] for j in labels]
            labels+=['actual_profit']
            shapley.append(list(payment_by_driver(i).values()))
            plot_income_distro(shapley,labels)
            plt.title(i['settings']['num_agents'])
            plt.show()
            plt.clf()
def plot_2():
    for i in data:
        if i['settings']['value_num'] != 2:
            labels = ['random_shapley','truncated_shapley','one_permutation_shapley']
            shapley = [i[j] for j in labels]
            labels+=['actual_profit']
            shapley.append(list(payment_by_driver(i).values()))
            plot_income_distro(shapley,labels)
            plt.title(i['settings']['value_num'])
            plt.show()
            plt.clf()

def plot_3():
    plots = [[],[],[],[]]
    xvals = []
    real_labels = ['random_shapley','truncated_shapley','one_permutation_shapley','actual_profit']
    for i in data:
        if i['settings']['value_num'] == 2:
            labels = ['random_shapley','truncated_shapley','one_permutation_shapley']
            shapley = [i[j] for j in labels]
            labels+=['actual_profit']
            shapley.append(list(payment_by_driver(i).values()))
            
            for j in range(len(shapley)):
                plots[j].append(np.std(shapley[j]))
            xvals.append(i['settings']['num_agents'])
    for i in range(len(real_labels)):
        plt.plot(xvals,plots[i],label=real_labels[i])
    plt.legend()
    plt.show()

def plot_4():
    plots = [[],[],[],[]]
    xvals = []
    real_labels = ['random_shapley','truncated_shapley','one_permutation_shapley','actual_profit']
    for i in data:
        if i['settings']['value_num'] == 2:
            labels = ['random_shapley','truncated_shapley','one_permutation_shapley']
            shapley = [i[j] for j in labels]
            labels+=['actual_profit']
            shapley.append(list(payment_by_driver(i).values()))
            
            for j in range(len(shapley)):
                plots[j].append(np.std(shapley[j])/np.std(shapley[-1]))
            xvals.append(i['settings']['num_agents'])
    for i in range(len(real_labels)):
        plt.plot(xvals,plots[i],label=real_labels[i])
    plt.legend()
    plt.show()

def plot_5():
    distros = []
    labels = []
    for i in data:
        if i['settings']['value_num'] == 2 and i['settings']['num_agents'] == 100:
            shapley_vals = i['truncated_shapley']
            real_payment = list(payment_by_driver(i).values())
            uniform = [sum(real_payment)/len(real_payment) for j in range(len(real_payment))]

            for r in [0,0.5,0.9,1]:
                p_1 = [r*real_payment[j] for j in range(len(real_payment))]
                p_2 = [r*real_payment[j] for j in range(len(real_payment))]
                p_3 = [r*shapley_vals[j] for j in range(len(shapley_vals))]

                r_1 = sum(real_payment)-sum(p_1)
                r_2 = sum(real_payment)-sum(p_2)
                r_3 = sum(real_payment)-sum(p_3)

                t_1 = sum([max(0,uniform[j]-p_1[j]) for j in range(len(p_1))])
                t_2 = sum([max(0,shapley_vals[j]-p_2[j]) for j in range(len(p_2))])
                t_3 = sum([max(0,uniform[j]-p_3[j]) for j in range(len(p_3))])

                for j in range(len(p_1)):
                    p_1[j]+=max(0,uniform[j]-p_1[j])/t_1 * r_1

                for j in range(len(p_2)):
                    p_2[j]+=max(0,shapley_vals[j]-p_2[j])/t_2 * r_2

                for j in range(len(p_3)):
                    p_3[j]+=max(0,uniform[j]-p_3[j])/t_3 * r_3
                if r == 0:
                    distros+=[real_payment]
                    labels+=['Real distribution']
                distros+=[p_1,p_2,p_3]
                labels+=["Distro 1 {}".format(r),"Distro 2 {}".format(r),"Distro 3 {}".format(r)]
                
    plot_income_distro(distros[::-1],labels[::-1])
    plt.show()

plot_1()
plot_2()
plot_3()
plot_4()
