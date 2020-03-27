import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import glob
from collections import Counter

def running_mean(x, N):
    averages = np.convolve(x, np.ones((N,))/N, mode='valid')
    # pad with 0s
    averages = np.append(averages,np.zeros(len(x)-len(averages)))
    return averages

def plot_daily(y):
    x = np.arange(0,24,1/60)
    plt.scatter(x,y,s=20)

def plot_running_mean(y):
    x = np.arange(0,24,1/60)
    sns.regplot(x,running_mean(y,30),lowess=True)        

def plot_requests_over_time(data):
    plot_running_mean(data['epoch_requests_seen'])

def plot_aggregate_requests_over_time(data):
    plot_daily(np.cumsum(data['epoch_requests_seen']))

def plot_requests_accepted_over_time(data):
    plot_running_mean(data['epoch_requests_accepted'])
    
def plot_aggregate_requests_accepted_over_time(data):
    plot_daily(np.cumsum(data['epoch_requests_accepted']))

def plot_requests_completed_over_time(data):
    plot_running_mean(data['epoch_requests_completed'])

def plot_aggregate_requests_completed_over_time(data):
    plot_daily(np.cumsum(data['epoch_requests_completed']))

def plot_average_wait_time(data):
    x = np.arange(0,24,1/60)
    y = np.array(data['epoch_dropoff_delay'])/np.array(data['epoch_requests_completed'])

    hourly_data = [y[i:i+60] for i in range(0,len(y),60)]
    hourly_data = [i[~np.isnan(i)] for i in hourly_data] 
    plt.boxplot(hourly_data)

def plot_percent_time_driving(data):
    x = np.arange(0,24,1)
    y = data["epoch_driver_0_empty"]
    minutes_driving = [60-sum(y[i:i+60]) for i in range(0,len(y),60)]
    plt.scatter(x,minutes_driving)

def payment_by_driver(data,num_drivers,n=-1):
    if n == -1:
        n = len(data)
    
    driver_pays = {}
    for i in range(num_drivers):
        driver_pays[i] = 0
    for i in data['epoch_each_agent_profit'][:n]:
        for j,k in i:
            driver_pays[j]+=k

    return driver_pays

def gini(data,num_drivers,n=-1):
    
    payment = payment_by_driver(data,num_drivers,n)
    mu = np.sum([i for i in payment.values()])/num_drivers
    
    a = 0
    for i in range(num_drivers):
        for j in range(num_drivers):
            a+=abs(payment[i]-payment[j])


    s =  a/(2*num_drivers**2 * mu)
    return s

def plot_gini_over_time(data,num_drivers):
    y_values = [gini(data,num_drivers,n=i) for i in range(len(data['epoch_each_agent_profit']))]
    plot_daily(y_values)

def plot_most_common_requests(data,n=10):
    all_requests = []
    for i in data['epoch_locations_all']:
        all_requests+=i

    freq = Counter(all_requests)
    sorted_list = sorted(list(set(all_requests)),key=lambda x: -freq[x])
    most_common = sorted_list[:n]
    frequencies = [freq[x] for x in most_common]
    plt.bar(list(range(n)),frequencies,tick_label=[str(i) for i in most_common])

def plot_most_common_acceptance(data,n=10):
    all_requests = []
    for i in data['epoch_locations_accepted']:
        all_requests+=i

    freq = Counter(all_requests)
    sorted_list = sorted(list(set(all_requests)),key=lambda x: -freq[x])
    most_common = sorted_list[:n]
    frequencies = [freq[x] for x in most_common]
    plt.bar(list(range(n)),frequencies,tick_label=[str(i) for i in most_common])

def plot_lorenz(data,num_drivers):
    payment = payment_by_driver(data,num_drivers)
    X = np.array(sorted([i for i in payment.values()]))
    
    X_lorenz = X.cumsum()/X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)
    X_lorenz[0], X_lorenz[-1]
    plt.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz, color='green')    
    plt.plot([0,1], [0,1], color='k')

def get_data(file_name=''):
    if not file_name:
        file_name = glob.glob("../logs/epoch_data/*")[0]
    day_0 = pickle.load(open(file_name,"rb"))

    return day_0

value_function_1 = get_data("../logs/epoch_data/day_11_epoch_data_agents10_value1_training0_testing1.pkl")
value_function_1_better = get_data("../logs/epoch_data/day_11_epoch_data_agents10_value1_training1_testing1.pkl")
value_function_1_evenbetter = get_data("../logs/epoch_data/day_11_epoch_data_agents10_value1_training2_testing1.pkl")
value_function_4 = get_data("../logs/epoch_data/day_11_epoch_data_agents10_value4_training0_testing1.pkl")
value_function_5 = get_data("../logs/epoch_data/day_11_epoch_data_agents10_value5_training0_testing1.pkl")
