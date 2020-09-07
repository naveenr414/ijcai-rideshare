from util import *
from pareto import *
from driverside import *
import matplotlib.pyplot as plt
from math import log 

robustness = get_robustness()

def plot_1():
    for i in [10,50,100,200]:
        for d in [1]:
            # Plot Pareto Colored
            colors = {1:'b',2:'r',10:'g',14:'k',15:'y'}
            labels = {1:'Request',2:'Baseline',10:'Income Variance',14:'Acceptance Variance',15:'Profit'}
            data = [j for j in robustness if j['settings']['num_agents'] == i and j['settings']['down_sample'] == d]


            color_list = [colors[j['settings']['value_num']] for j in data]
            label_list = [labels[j['settings']['value_num']] for j in data]
            plt.figure(figsize=(8,5))
            plot_num_min_request(data,label_list,color_list)
            plt.title("{} Drivers".format(i),fontsize=24)
            plt.show()

def plot_2():
    for i in [10,50,100,200]:
        for k in [0.25,0.5,1]:
            # Plot Pareto Colored
            data = [j for j in robustness if j['settings']['num_agents'] == i and j['settings']['down_sample'] == k]
            colors = []
            labels = []
            for j in data:
                if 'nn_inputs' in j['settings'] and j['settings']['value_num'] == 10:
                    colors.append('r')
                    labels.append("profitz income variance")
                elif 'nn_inputs' in j['settings'] and j['settings']['value_num'] == 14:
                    colors.append('k')
                    labels.append("profitz request variance")
                elif j['settings']['value_num'] == 10:
                    colors.append('y')
                    labels.append("regular income variance")     
                else:
                    colors.append('b')
                    labels.append("regular request variance")

            plot_driver(data,label=False,colors=colors,legend=labels)
            plt.title("Driver side {} {}".format(i,k))
            plt.show()
            plt.clf()

            plot_rider(data,label=False,colors=colors,legend=labels)
            plt.title("Rider side {} {}".format(i,k))
            plt.show()
            plt.clf()

def plot_3():
    for i in [10,50,100,200]:
        for d in [1]:
            # Plot Pareto Colored
            colors = {1:'b',2:'r',10:'g',14:'k',15:'y'}
            labels = {1:'Request',2:'Baseline',10:'Income Variance',14:'Acceptance Variance',15:'Profit'}
            data = [j for j in robustness if j['settings']['num_agents'] == i and j['settings']['down_sample'] == d]


            color_list = [colors[j['settings']['value_num']] for j in data]
            label_list = [labels[j['settings']['value_num']] for j in data]

            plot_driver(data,label=False,colors=color_list,legend=label_list)
            plt.title("Num drivers {}".format(i))
            plt.show()
            plt.clf()

def plot_4():
    for i in [200]:
        for d in [1]:    
            labels = {1:'Request',2:'Baseline',10:'Inc Var.',14:'Req Var.',15:'Profit'}
            allowed_lambda = [0,0.33,0.67,1,10**8,10**9,10**10]
            data = [i for i in robustness if i['settings']['num_agents'] == 200 and i['settings']['down_sample'] == 1 and ('lambda' not in i['settings'] or round(i['settings']['lambda'],2) in allowed_lambda)]
            def sortby(i):
                if 'lambda' not in i['settings']:
                    return i['settings']['value_num']
                return i['settings']['value_num']+i['settings']['lambda']/10**11
            data = sorted(data,key=sortby,reverse=True)
            label_list = []
            for j in data:
                if 'lambda' in j['settings']:
                    if j['settings']['lambda']>100:
                        label_list.append(labels[j['settings']['value_num']]+"  {}".format('10^'+str(round(log(j['settings']['lambda'],10)))))
                    else:
                        label_list.append(labels[j['settings']['value_num']]+"  {}".format(round(j['settings']['lambda'],2)))
                else:
                    label_list.append(labels[j['settings']['value_num']])
            incomes = [list(payment_by_driver(i).values()) for i in data]
            plt.figure(figsize=(15,5))
            plot_income_distro(incomes,label_list)
            plt.title("Income by Policy at 200 drivers",fontsize=24)
            plt.show()

def info_1():
    for j in [10,50,100,200]:
        data = [i for i in robustness if i['settings']['num_agents'] == j and i['settings']['down_sample'] == 1]
        profits = [(total_profit(i),i['settings']['value_num']) for i in data]
        print(min(profits),max(profits))
        print(min(profits)[0]/j,max(profits)[0]/j)
        print(sorted(profits)[-5:])
        print()

def info_2():
    for j in [10,50,100,200]:
        data = [i for i in robustness if i['settings']['num_agents'] == j  and i['settings']['down_sample'] == 1]
        lambdas = []
        for i in data:
            if 'lambda' in i['settings']:
                lambdas.append(i['settings']['lambda'])
            else:
                lambdas.append(-1)
        profits = [(total_profit(i),income_25(i),i['settings']['value_num'],lambdas[k]) for k,i in enumerate(data)]
        print(min(profits),max(profits))
        print(min(profits,key=lambda x: x[1]),max(profits,key=lambda x: x[1]))
        print(min(profits)[0]/j,max(profits)[0]/j)
        print(sorted(profits)[-5:])
        print()

def info_25():
    for j in [10,50,100,200]:
        data = [i for i in robustness if i['settings']['num_agents'] == j  and i['settings']['down_sample'] == 1]
        lambdas = []
        for i in data:
            if 'lambda' in i['settings']:
                lambdas.append(i['settings']['lambda'])
            else:
                lambdas.append(-1)
        profits = [(total_profit(i),std_income(i),i['settings']['value_num'],lambdas[k]) for k,i in enumerate(data)]
        print(min(profits),max(profits))
        print(min(profits,key=lambda x: x[1]),max(profits,key=lambda x: x[1]))
        print(min(profits)[0]/j,max(profits)[0]/j)
        print(sorted(profits)[-5:])
        print()

def info_3():
    for j in [10,50,100,200]:
        data = [i for i in robustness if i['settings']['num_agents'] == j and i['settings']['down_sample'] == 1]
        lambdas = []
        for i in data:
            if 'lambda' in i['settings']:
                lambdas.append(i['settings']['lambda'])
            else:
                lambdas.append(-1)
        profits = [(requests_completed(i),i['settings']['value_num'],lambdas[k]) for k,i in enumerate(data)]
        print([i for i in profits if i[1] == 1])
        print(min(profits),max(profits))
        print(min(profits)[0]/j,max(profits)[0]/j)
        print(sorted(profits)[-5:])
        print()

def info_4():
    data = [i for i in robustness if i['settings']['num_agents'] == 200 and i['settings']['down_sample'] == 1 and i['settings']['value_num'] in [10]]
    profits = [(total_profit(i),income_25(i),std_income(i),i['settings']['value_num'],i['settings']['lambda']) for k,i in enumerate(data)]
    print(min(profits),max(profits))
    print(min(profits,key=lambda x: x[1]),max(profits,key=lambda x: x[1]))
    print(min(profits,key=lambda x: x[2]),max(profits,key=lambda x: x[2]))
    print(sorted(profits))

def info_5():
    for j in [10,50,100,200]:
        data = [i for i in robustness if i['settings']['num_agents'] == j and i['settings']['down_sample'] == 1 and i['settings']['value_num'] in [14]]
        print(len(data))
        profits = [(requests_completed(i),min_region(i),rider_fairness(i),i['settings']['value_num'],i['settings']['lambda']) for k,i in enumerate(data)]
        print(min(profits),max(profits))
        print(min(profits,key=lambda x: x[1]),max(profits,key=lambda x: x[1]))
        print(min(profits,key=lambda x: x[2]),max(profits,key=lambda x: x[2]))
        print(sorted(profits))

def info_6():
    for j in [10,50,100,200]:
        data = [i for i in robustness if i['settings']['num_agents'] == j and i['settings']['down_sample'] == 1]
        lambdas = []
        for i in data:
            if 'lambda' in i['settings']:
                lambdas.append(i['settings']['lambda'])
            else:
                lambdas.append(-1)

        profits = [(requests_completed(i),min_region(i),rider_fairness(i),i['settings']['value_num'],lambdas[k]) for k,i in enumerate(data)]
        print(min(profits),max(profits))
        print(min(profits,key=lambda x: x[1]),max(profits,key=lambda x: x[1]))
        print(min(profits,key=lambda x: x[2]),max(profits,key=lambda x: x[2]))
        print()

def info_7():
    for j in [0.25,0.5,1]:
        data = [i for i in robustness if i['settings']['num_agents'] == 100 and i['settings']['down_sample'] == j and i['settings']['value_num'] == 1]
        profits = [(std_income(i),income_25(i),min_region(i),rider_fairness(i),i['settings']['value_num']) for k,i in enumerate(data)]
        print(profits)



plot_4()

