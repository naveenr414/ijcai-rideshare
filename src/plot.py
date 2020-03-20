import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import glob 

def running_mean(x, N):
    averages = np.convolve(x, np.ones((N,))/N, mode='valid')
    # pad with 0s
    averages = np.append(averages,np.zeros(len(x)-len(averages)))
    return averages

def plot_daily(y):
    x = np.arange(0,24,1/60)
    plt.scatter(x,y,s=2)

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

def get_data():
    all_days = glob.glob("../logs/epoch_data/*")
    day_0 = pickle.load(open(all_days[0],"rb"))

    return day_0

day_0 = get_data()
plot_requests_over_time(day_0)
plot_requests_accepted_over_time(day_0)
plt.show()
