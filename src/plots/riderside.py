import matplotlib.pyplot as plt
import numpy as np
from util import *

def plot_histogram_of_locations(data,accepted=False):
    all_requests = []

    if accepted:
        for i in data['epoch_locations_accepted']:
            all_requests+=i
    else:
        for i in data['epoch_locations_all']:
            all_requests+=i

    freq = Counter(all_requests)

    for i in range(4461):
        if i not in freq:
            freq[i] = 0
        
    plt.bar(list(freq.keys()), freq.values(), color='g')


def income_requests(data_list):
    income = []
    requests = []
    names = [get_nice_name(i) for i in data_list]
    for data in data_list:
        success = get_region_percentages(data)
        income.append(np.sum(payment_by_driver(data).values()))
        requests.append(total_services(data))

    plt.xlabel("Income")
    plt.ylabel("Total Requests")

    plot_two_axis(income,requests,names,use_labels=False)

def min_avg_rider_percentage(data_list):
    min_success = []
    mean_success = []
    names = [get_nice_name(i) for i in data_list]
    for data in data_list:
        success = get_region_percentages(data)
        min_success.append(min(success))
        mean_success.append(np.mean(success))

    plot_two_axis(min_success,mean_success,names,use_labels=False)
        
    return min_success,mean_success,names

def min_total_rider_percentage(data_list):
    min_success = []
    total_rides = []
    names = [get_nice_name(i) for i in data_list]

    for data in data_list:
        success = get_region_percentages(data)
        print(np.min(success),success)
        min_success.append(np.min(success))
        total_rides.append(total_services(data))

    plt.title("Min Percentage vs. Total Requests for different policies")
    plt.xlabel("Worst region, acceptance rate")
    plt.ylabel("Total requests serviced")

    plot_two_axis(min_success,total_rides,names,use_labels=False)
    return min_success,total_rides,names

def diff_avg_rider_percentage(data_list):
    diff_success = []
    mean_success = []
    names = [get_nice_name(i) for i in data_list]
    for data in data_list:
        success = get_region_percentages(data)
        diff_success.append(max(success)-min(success))
        mean_success.append(np.mean(success))

    diff_success,mean_success,names = get_pareto(diff_success,mean_success,names)
    plot_two_axis(diff_success,mean_success,names,use_labels=True)
        
    return diff_success,mean_success,names
