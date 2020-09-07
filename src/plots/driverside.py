import matplotlib.pyplot as plt
from util import *

def plot_lorenz(data,num_drivers):
    payment = payment_by_driver(data)
    X = np.array(sorted([i for i in payment.values()]))
    X_lorenz = X.cumsum()/X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)
    X_lorenz[0], X_lorenz[-1]
    plt.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz, label=current_label)    
    plt.plot([0,1], [0,1],color='g')

def plot_income_distro(incomes,names):
    plt.title("Income distribution for policies",fontsize=20)
    plt.boxplot(incomes,showfliers=True,vert=False)
    plt.ylabel("Policy",fontsize=18)
    plt.xlabel("Payment ($)",fontsize=18)
    plt.yticks(list(range(1,len(names)+1)),names,fontsize=14)
    plt.xticks(fontsize=18)

def plot_gini_by_policy(data_list):
    lamb_values = set()
    value_num = [7,8]
    for i in data_list:
        if i['settings']['value_num'] in value_num and i['settings']['lambda']>=0:
            lamb_values.add(round(i['settings']['lambda'],2))

    colors = ['r','b','k','y','c','m']
    labels = []
    num_times = 0
    for lamb in lamb_values:
        for i in range(2):
            current_label = "Lambda: "+str(lamb)
            if i == 1:
                current_label+=" with profitz"
            labels.append(current_label)
            data = [0 for k in range(4)]
            for j in data_list:
                if j['settings']['value_num'] in value_num and round(j['settings']['lambda'],2) == lamb:
                    if i == 0 and "nn_inputs" not in j['settings'] or (i == 1 and "nn_inputs" in j['settings'] and 'profit_z' in j['settings']['nn_inputs']):
                        num_times+=1
                        data[j['settings']['training_days']] = gini(j)
            if i == 1:
                plt.plot(list(range(0,4))[1:],data[1:],c=colors[len(labels)-1])

            else:
                plt.plot(list(range(0,4)),data,c=colors[len(labels)-1])

    plt.legend(labels)
    plt.show()

    return num_times

def box_whisker_income(data_list,ymin=-1,ymax=-1):
    data = [i for i in data_list if(i['settings']['value_num'] in [1,2,10]
                               and (i['settings']['value_num'] in [2] or i['settings']['training_days']>=2)
                               and ('lambda' not in i['settings'] or i['settings']['lambda']>=0))
                                and (i['settings']['value_num'] in [1,2] or 'nn_inputs' in i['settings'])
            and ('pickup_delay' not in i['settings'])]
    names = [get_nice_name(i) for i in data]
    incomes = [list(payment_by_driver(i).values()) for i in data]
    plot_income_distro(incomes,names)

    if ymin!=-1 and ymax!=-1:
        plt.xlim((ymin,ymax))
