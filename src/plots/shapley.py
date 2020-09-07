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
        if i['settings']['value_num'] == 1 and i['settings']['num_agents'] == 100:
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
                distros+=[p_1,p_3]
                labels+=["Income {}".format(r),"Shapley {}".format(r)]

    plot_income_distro(distros[::-1],labels[::-1])
    plt.title("Redistribution of Income",fontsize=24)
    plt.show()

def plot_6():
    for p in [1,2,10,14,15]:
        distros = []
        labels = []
        for i in data:
            if i['settings']['value_num'] == p and i['settings']['num_agents'] == 100:
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
                    distros+=[p_1,p_3]
                    labels+=["Income {}".format(r),"Shapley 2 {}".format(r)]

        plot_income_distro(distros[::-1],labels[::-1])
        plt.title("Redistribution of Income",fontsize=24)
        plt.show()

def plot_65():
    for p in [10,50,100,200]:
        distros = []
        labels = []
        for i in data:
            if i['settings']['value_num'] == 2 and i['settings']['num_agents'] == p:
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
                    distros+=[p_1,p_3]
                    labels+=["Income {}".format(r),"Shapley 2 {}".format(r)]

        plot_income_distro(distros[::-1],labels[::-1])
        plt.title("Redistribution of Income",fontsize=24)
        plt.show()

def plot_7():
    for i in data:
        if i['settings']['value_num'] == 2 and i['settings']['num_agents'] == 100:
            shapley_vals = i['truncated_shapley']
            real_payment = list(payment_by_driver(i).values())
            uniform = [sum(real_payment)/len(real_payment) for j in range(len(real_payment))]
            median_payment = sorted(real_payment)[0]
            median_actor = -1

            amount_more = []
            x_vals = [j/100 for j in range(101)]
            
            for j in range(len(real_payment)):
                if real_payment[j] == median_payment:
                    median_actor = j

            print(real_payment[median_actor])
            print()

            for r in range(101):
                real_r = r/100

                
                p_1 = [real_r*real_payment[j] for j in range(len(real_payment))]

                r_1 = sum(real_payment)-sum(p_1)

                t_1 = sum([max(0,uniform[j]-p_1[j]) for j in range(len(p_1))])

                for j in range(len(p_1)):
                    p_1[j]+=max(0,uniform[j]-p_1[j])/t_1 * r_1
            
                initial_pay = p_1[median_actor]
                real_payment[median_actor]*=2
                p_1 = [real_r*real_payment[j] for j in range(len(real_payment))]

                r_1 = sum(real_payment)-sum(p_1)

                t_1 = sum([max(0,uniform[j]-p_1[j]) for j in range(len(p_1))])

                for j in range(len(p_1)):
                    p_1[j]+=max(0,uniform[j]-p_1[j])/t_1 * r_1

                later_pay = p_1[median_actor]
                print(later_pay,initial_pay)
                real_payment[median_actor]/=2
                amount_more.append(later_pay/initial_pay)
            plt.title("Gain vs. R",fontsize=24)
            plt.plot(x_vals,amount_more)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("R",fontsize=16)
            plt.ylabel("Gain",fontsize=16)
            plt.show()

def plot_8():
    for i in data:
        if i['settings']['value_num'] == 2 and i['settings']['num_agents'] == 100:
            shapley_vals = i['truncated_shapley']
            real_payment = list(payment_by_driver(i).values())
            uniform = [sum(real_payment)/len(real_payment) for j in range(len(real_payment))]
            median_payment = sorted(real_payment)[len(real_payment)//2]
            median_actor = -1

            amount_more = []
            x_vals = [j/100 for j in range(101)]
            
            for j in range(len(real_payment)):
                if real_payment[j] == median_payment:
                    median_actor = j

            for r in range(101):
                real_r = r/100

                
                p_1 = [real_r*real_payment[j] for j in range(len(real_payment))]

                r_1 = sum(real_payment)-sum(p_1)

                t_1 = sum([max(0,uniform[j]-p_1[j]) for j in range(len(p_1))])

                for j in range(len(p_1)):
                    p_1[j]+=max(0,uniform[j]-p_1[j])/t_1 * r_1
            
                amount_more.append(np.std(p_1))

            print(amount_more)
            plt.title("Standard Deviation vs. R",fontsize=24)
            plt.plot(x_vals,amount_more)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("R",fontsize=16)
            plt.ylabel("Standard Deviation",fontsize=16)
            plt.show()
def plot_9():
    for i in data:
        if i['settings']['value_num'] == 2 and i['settings']['num_agents'] == 100:
            shapley_vals = i['truncated_shapley']
            real_payment = list(payment_by_driver(i).values())
            uniform = [sum(real_payment)/len(real_payment) for j in range(len(real_payment))]
            median_payment = sorted(real_payment)[len(real_payment)//2]
            median_actor = -1

            amount_more = []
            x_vals = [j/100 for j in range(101)]
            
            for j in range(len(real_payment)):
                if real_payment[j] == median_payment:
                    median_actor = j

            for r in range(101):
                real_r = r/100

                
                p_1 = [real_r*real_payment[j] for j in range(len(real_payment))]

                r_1 = sum(real_payment)-sum(p_1)

                t_1 = sum([max(0,uniform[j]-p_1[j]) for j in range(len(p_1))])

                for j in range(len(p_1)):
                    p_1[j]+=max(0,uniform[j]-p_1[j])/t_1 * r_1
            
                amount_more.append(sorted(p_1)[len(p_1)//4])

            print(amount_more)
            plt.title("25th percentile of Income",fontsize=24)
            plt.plot(x_vals,amount_more)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("R Value",fontsize=16)
            plt.ylabel("25th Percentile Income",fontsize=16)
            plt.show()
plot_7()    
