def get_coords(data):
    return [total_profit(data),1-gini(data),1/average_dropoff_delay(data),requests_completed(data),1/rider_fairness(data)]

def plot_two_axis(axis_one,axis_two,labels,use_labels=True):
    for i in range(len(axis_one)):
        x = axis_one[i]
        y = axis_two[i]
        plt.scatter(x,y)
        if(use_labels):
            plt.text(x * (1 + 0.01), y * (1 + 0.01) , labels[i], fontsize=12)


def get_two_axis(all_pickles,axis_one,axis_two):
    a = []
    b = []
    for i in all_pickles:
        c = get_coords(i)
        a.append(c[axis_one])
        b.append(c[axis_two])

    return a,b

def get_pareto(axis_one,axis_two,labels=[]):
    pareto = []
    for i in range(len(axis_one)):
        pareto.append(True)
        for j in range(len(axis_one)):
            if(i!=j and axis_one[j]>=axis_one[i] and axis_two[j]>=axis_two[i]):
                pareto[i] = False

    axis_one = [a for i,a in enumerate(axis_one) if pareto[i]]
    axis_two = [a for i,a in enumerate(axis_two) if pareto[i]]
    labels = [a for i,a in enumerate(labels) if pareto[i]]

    return axis_one,axis_two,labels


def plot_driver(all_pickles):
    profit, inverse_inequality = get_two_axis(all_pickles,0,1)
    labels = [get_name(i) for i in all_pickles]

    plt.scatter(profit,inverse_inequality)
    plt.xlabel("Total Profit")
    plt.ylabel("1-Gini Coefficient")
    plt.title("Driver side all")

    for i in range(len(all_pickles)):
        plt.annotate(labels[i],(profit[i],inverse_inequality[i]))

def plot_rider(all_pickles):
    profit, inverse_inequality = get_two_axis(all_pickles,3,4)
    labels = [get_name(i) for i in all_pickles]

    plt.scatter(profit,inverse_inequality)
    plt.title("Rider side all")
    plt.xlabel("Requests Completed")
    plt.ylabel("1/std of location distribution")

    for i in range(len(all_pickles)):
        plt.annotate(labels[i],(profit[i],inverse_inequality[i]))

def plot_driver_pareto(all_pickles):
    profit, inverse_inequality = get_two_axis(all_pickles,0,1)
    labels = [get_name(i) for i in all_pickles]

    profit,inverse_inequality,labels = get_pareto(profit,inverse_inequality,labels)
    plt.scatter(profit,inverse_inequality)
    plt.title("Pareto driver side")
    plt.xlabel("Total Profit")
    plt.ylabel("1-Gini Coefficient")

    for i in range(len(profit)):
        plt.annotate(labels[i],(profit[i],inverse_inequality[i]))

def plot_rider_pareto(all_pickles):
    profit, inverse_inequality = get_two_axis(all_pickles,3,4)
    labels = [get_name(i) for i in all_pickles]

    profit,inverse_inequality,labels = get_pareto(profit,inverse_inequality,labels)
    plt.scatter(profit,inverse_inequality)
    plt.title("Pareto rider side")
    plt.xlabel("Requests Completed")
    plt.ylabel("1/std of location distribution")

    for i in range(len(profit)):
        plt.annotate(labels[i],(profit[i],inverse_inequality[i]))



def plot_driver_pareto_within_valuenum(all_pickles):
    profit = []
    inverse_inequality = []
    labels = [get_name(i) for i in all_pickles]
    for i in range(len(all_pickles)):
        data = all_pickles[i]
        profit.append(total_profit(data))
        inverse_inequality.append(1-gini(data))

    pareto = []
    for i in range(len(profit)):
        pareto.append(True)
        for j in range(len(profit)):
            if(i!=j and profit[j]>=profit[i] and inverse_inequality[j]>=inverse_inequality[i] and all_pickles[i]["settings"]["value_num"] == all_pickles[j]["settings"]["value_num"]):
                pareto[i] = False

    profit = [a for i,a in enumerate(profit) if pareto[i]]
    inverse_inequality = [a for i,a in enumerate(inverse_inequality) if pareto[i]]
    labels = [a for i,a in enumerate(labels) if pareto[i]]

    plt.title("Pareto driver within value number")
    plt.scatter(profit,inverse_inequality)
    plt.xlabel("Total Profit")
    plt.ylabel("1-Gini Coefficient")

    for i in range(len(profit)):
        plt.annotate(labels[i],(profit[i],inverse_inequality[i]))

def plot_rider_pareto_within_valuenum(all_pickles):
    profit = []
    inverse_inequality = []
    labels = [get_name(i) for i in all_pickles]
    for i in range(len(all_pickles)):
        data = all_pickles[i]
        profit.append(requests_completed(data))
        inverse_inequality.append(1/rider_fairness(data))

    pareto = []
    for i in range(len(profit)):
        pareto.append(True)
        for j in range(len(profit)):
            if(i!=j and profit[j]>=profit[i] and inverse_inequality[j]>=inverse_inequality[i] and all_pickles[i]["settings"]["value_num"] == all_pickles[j]["settings"]["value_num"]):
                pareto[i] = False

    profit = [a for i,a in enumerate(profit) if pareto[i]]
    inverse_inequality = [a for i,a in enumerate(inverse_inequality) if pareto[i]]
    labels = [a for i,a in enumerate(labels) if pareto[i]]

    plt.title("Pareto rider within value number")
    plt.scatter(profit,inverse_inequality)
    plt.xlabel("Requests Completed")
    plt.ylabel("1/std of location distribution")

    for i in range(len(profit)):
        plt.annotate(labels[i],(profit[i],inverse_inequality[i]))
