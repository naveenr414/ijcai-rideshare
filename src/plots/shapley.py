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
    print(len(data))
    for p in [1,2,10,14,15]:
        distros = []
        labels = []
        for i in data:
            if i['settings']['value_num'] == p and i['settings']['num_agents'] == 100:
                print(p)
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
            median_payment = sorted(real_payment)[len(real_payment)//2]
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
            plt.xlabel("r",fontsize=16)
            plt.ylabel("Gain",fontsize=16)
            plt.show()

def plot_8():
    for i in data:
        if i['settings']['value_num'] == 1 and i['settings']['num_agents'] == 100:
            shapley_vals = i['truncated_shapley']
            real_payment = list(payment_by_driver(i).values())
            uniform = shapley_vals#[sum(real_payment)/len(real_payment) for j in range(len(real_payment))]

            
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

                value_percent = [p_1[j]/uniform[j] for j in range(len(p_1))]
            
                amount_more.append(np.std(value_percent))
            plt.figure(figsize=(6,4))
            plt.title("Standard Deviation vs. r",fontsize=16)
            plt.plot(x_vals,amount_more)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("r",fontsize=16)
            plt.ylabel("Standard Deviation",fontsize=16)
            plt.tight_layout()
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
            plt.xlabel("r Value",fontsize=16)
            plt.ylabel("25th Percentile Income",fontsize=16)
            plt.show()

def plot_10():
    for i in data:
        if i['settings']['value_num'] == 1 and i['settings']['num_agents'] == 100:
            shapley_vals = i['truncated_shapley']
            real_payment = list(payment_by_driver(i).values())
            uniform = shapley_vals#[sum(real_payment)/len(real_payment) for j in range(len(real_payment))]
            median_payment = sorted(real_payment)[0]
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

                gain_metric = 0
                for q in range(len(p_1)):
                    initial_pay = p_1[q]
                    real_payment[q]*=2
                    p_1 = [real_r*real_payment[j] for j in range(len(real_payment))]

                    r_1 = sum(real_payment)-sum(p_1)

                    t_1 = sum([max(0,uniform[j]-p_1[j]) for j in range(len(p_1))])

                    for j in range(len(p_1)):
                        p_1[j]+=max(0,uniform[j]-p_1[j])/t_1 * r_1

                    later_pay = p_1[q]
                    real_payment[q]/=2
                    gain_metric+=(later_pay/initial_pay-1)
                gain_metric/=100
                amount_more.append(gain_metric)
            plt.figure(figsize=(6,4))
            plt.title("Gain vs. r",fontsize=16)
            plt.plot(x_vals,amount_more)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("r",fontsize=16)
            plt.ylabel("Gain",fontsize=16)
            plt.tight_layout()
            plt.show()
def plot_11():
    for i in data:
        if i['settings']['value_num'] == 1 and i['settings']['num_agents'] == 100:
            shapley_vals = i['truncated_shapley']
            real_payment = list(payment_by_driver(i).values())
            uniform = [sum(real_payment)/len(real_payment) for j in range(len(real_payment))]
            median_payment = sorted(real_payment)[len(real_payment)//4]
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
            plt.title("Gain vs. R for 25th Percentile Driver",fontsize=24)
            plt.plot(x_vals,amount_more)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("r",fontsize=16)
            plt.ylabel("Gain",fontsize=16)
            plt.show()

def plot_12():
    for i in data:
        if i['settings']['value_num'] == 1 and i['settings']['num_agents'] == 100:
            shapley_vals = i['truncated_shapley']
            real_payment = list(payment_by_driver(i).values())
            uniform = shapley_vals#[sum(real_payment)/len(real_payment) for j in range(len(real_payment))]
            min_vals = []
            bounds = []

            
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

                value_percent = [p_1[j]/uniform[j] for j in range(len(p_1))]
                k = np.argmin(p_1)
                min_vals.append(p_1[k])
                bounds.append(min(real_r*uniform[k],(1-real_r)*uniform[k]))
            
                amount_more.append(np.std(value_percent))
            plt.figure(figsize=(6,4))
            plt.title("Min Income vs. Bound",fontsize=16)
            plt.plot(x_vals,min_vals,label="Min Income")
            plt.plot(x_vals,bounds,label="Bound on Income")
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("r",fontsize=16)
            plt.ylabel("Income",fontsize=16)
            plt.legend()
            plt.tight_layout()
            plt.show()

def plot_13():
    for i in data:
        if i['settings']['value_num'] == 1 and i['settings']['num_agents'] == 100:
            shapley_vals = i['truncated_shapley']
            real_payment = list(payment_by_driver(i).values())
            uniform = [sum(real_payment)/len(real_payment) for j in range(len(real_payment))]
            min_vals = []
            bounds = []

            incomes = []
            labels = []

            
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

                value_percent = [p_1[j]/uniform[j] for j in range(len(p_1))]
                k = np.argmin(p_1)
                min_vals.append(p_1[k])
                bounds.append(min(real_r*uniform[k],(1-real_r)*uniform[k]))
            
                amount_more.append(np.std(value_percent))

                if real_r in [0,0.5,0.9,0.95,1]:
                    incomes.append(p_1)
                    labels.append(str(real_r))
                    
                
            plt.figure(figsize=(6,4))
            plot_income_distro(incomes,labels)
            plt.title("Income distribution for various r",fontsize=16)
#            plt.xlabel("R",fontsize=16)
            plt.ylabel("r",fontsize=16)
            plt.tight_layout()
            plt.show()


plot_8()
plot_10()
plot_12()
plot_13()

