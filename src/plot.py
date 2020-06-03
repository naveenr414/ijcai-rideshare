import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import glob
from collections import Counter
import pandas as pd
import glob
from scipy.special import kl_div
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']) 

current_label = ""

def running_mean(x, N):
    averages = np.convolve(x, np.ones((N,))/N, mode='valid')
    # pad with 0s
    diff = len(x)-len(averages)
    start = diff//2
    end = diff-start

    l = []
    for i in range(start):
        l.append(sum(x[:i+1])/(i+1))
    l+=list(averages)
    for i in range(end,0,-1):
        l.append(sum(x[-i:])/len(x[-i:]))
    
    return np.array(l)

def plot_daily(y):
    x = np.arange(0,24,1/60)
    plt.plot(x,y,label=current_label)

def plot_running_mean(y):
    x = np.arange(0,24,1/60)
    plt.plot(x,running_mean(y,30),label=current_label)

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

# As a box plot
def plot_average_wait_time_box(data):
    x = np.arange(0,24,1/60)
    y = np.array(data['epoch_dropoff_delay'])/np.array(data['epoch_requests_completed'])

    hourly_data = [y[i:i+60] for i in range(0,len(y),60)]
    hourly_data = [i[~np.isnan(i)] for i in hourly_data] 
    plt.boxplot(hourly_data)

def plot_service_rates(data_list,labels):
    service_rates = [np.sum(i['epoch_requests_accepted'])/np.sum(i['epoch_requests_seen']) for i in data_list]
    plt.bar(list(range(len(labels))),service_rates,tick_label=labels)

def plot_KL_divergences(data_list):
    plt.bar(list(range(len(data_list))),[get_KL_divergence(data_list[i]) for i in data_list])
    plt.xticks(list(range(len(data_list))),list(data_list.keys()))

def plot_service_rates_dict(data):
    labels = sorted([i for i in data])
    data_list = [data[i] for i in labels]
    plot_service_rates(data_list,labels)

def plot_average_wait_time(data):
    y = np.array(data['epoch_dropoff_delay'])/np.array(data['epoch_requests_completed'])
    plot_running_mean(y)

def plot_percent_time_driving(data):
    x = np.arange(0,24,1)
    y = data["epoch_driver_0_empty"]
    minutes_driving = [60-sum(y[i:i+60]) for i in range(0,len(y),60)]
    plt.scatter(x,minutes_driving)

def payment_by_driver(data,num_drivers,n=-1):
    if n == -1:
        n = len(data['epoch_each_agent_profit'])
    
    driver_pays = {}
    for i in range(num_drivers):
        driver_pays[i] = 0
    for i in data['epoch_each_agent_profit'][:n]:
        for j,k in i:
            driver_pays[j]+=k

    return driver_pays

def entropy(data,num_drivers):
    payment_driver = np.array(list((payment_by_driver(data,num_drivers).values())))
    payment_driver+=10**-6
    ybar = np.mean(payment_driver)

    print(payment_driver)
    print(np.log(payment_driver/ybar))
    
    return -1/num_drivers * (np.sum(np.log(payment_driver/ybar)))

def plot_profit_over_time(data):
    y_values = []
    for i in data['epoch_each_agent_profit']:
        y_values.append(sum([x[1] for x in i]))
    plot_running_mean(y_values)

def plot_gini_over_time(data,num_drivers):
    driver_pays = {}
    for i in range(num_drivers):
        driver_pays[i] = 0
    gini_list = []
    total_pay = 0
    for q, i in enumerate(data['epoch_each_agent_profit']):
        for j,l in i:
            driver_pays[j]+=l
            total_pay+=l
        mu = total_pay/num_drivers
        a = 0

        driver_pays_list = list(driver_pays.values())
        driver_pays_list = sorted(driver_pays_list)
        
        for j in range(num_drivers):
            a+=j*driver_pays_list[j] - (num_drivers-1-j)*driver_pays_list[j]
            
        gini_list.append(2*a/(2*num_drivers**2*mu))
    
    plot_daily(np.array(gini_list))

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
    plt.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz, label=current_label)    
    plt.plot([0,1], [0,1],color='g')

def histogram_of_locations(data,accepted=False):
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

    return np.array([float(i) for i in freq.values()])

def plot_cosine_similarity_over_time(data):
    similarity_list = []
    accepted = {}
    all_loc = {}

    for i in range(4461):
        all_loc[i] = 0
        accepted[i] = 0
    
    for i in range(len(data['epoch_locations_accepted'])):
        for j in data['epoch_locations_accepted'][i]:
            accepted[j]+=1
        for j in data['epoch_locations_all'][i]:
            all_loc[j]+=1

        similarity_list.append(1-cosine(np.array([float(k) for k in accepted.values()]),np.array([float(k) for k in all_loc.values()])))

    plot_running_mean(similarity_list)

def plot_request_locations(data,accepted=False):
    all_requests = []

    if accepted:
        for i in data['epoch_locations_accepted']:
            all_requests+=i
    else:
        for i in data['epoch_locations_all']:
            all_requests+=i

    freq = Counter(all_requests)
    rated_geo = pd.read_csv("../data/ny/zone_latlong.csv",header=None,names=['zone','longitude','latitude'])
    new_col = [freq[i] if i in freq else 0 for i in range(len(rated_geo))]
    rated_geo['score'] = new_col

    min_lon = rated_geo['longitude'].min()
    max_lon = rated_geo['longitude'].max()
    min_lat = rated_geo['latitude'].min()
    max_lat = rated_geo['latitude'].max()
    max_score = rated_geo['score'].max()
    min_score = rated_geo['score'].min()

    mean = np.mean(rated_geo['latitude'])
    sd = np.sqrt(np.sum(rated_geo['score']*(rated_geo['latitude']-mean)**2)/np.sum(rated_geo['score']))
    
    bound = ((min_lon, max_lon, min_lat, max_lat))
    map_bound = ((-74.05, -73.948, 40.682, 40.79))
    basemap = plt.imread('../data/ny/ny.jpg')    

    plt.xlim(map_bound[0],map_bound[1])
    plt.ylim(map_bound[2],map_bound[3])
    plt.scatter('longitude', 'latitude', data = rated_geo, s=rated_geo['score']**.5)
    #hexbin = plt.hexbin('longitude', 'latitude', data = rated_geo, C=rated_geo['score'], gridsize = 200)
    #plt.colorbar(hexbin, orientation='vertical').set_label('Inspection Count')
    plt.xlabel('Longitude');
    plt.ylabel('Latitude');
    plt.title('Geospatial Density of Scores of Rated Restaurants');
    # Setting aspect ratio and plotting the hexbins on top of the base map layer
    plt.imshow(basemap, zorder=0, extent = map_bound, aspect= 'equal');

def plot_request_rejections(data):
    all_requests = []
    accepted = []

    for i in data['epoch_locations_accepted']:
        accepted+=i

        
    for i in data['epoch_locations_all']:
        all_requests+=i

    freq_all = Counter(all_requests)
    freq_accepted = Counter(accepted)

    print(sum(freq_all.values()),sum(freq_accepted.values()))

    freq = {}
    for i in freq_all:
        if freq_all[i]>5:
            if i not in freq_accepted:
                freq_accepted[i] = 0
            acceptance_percentage = freq_accepted[i]/freq_all[i]
            if acceptance_percentage<0.03:
                freq[i] = freq_all[i]
        
    rated_geo = pd.read_csv("../data/ny/zone_latlong.csv",header=None,names=['zone','longitude','latitude'])
    new_col = [freq[i] if i in freq else 0 for i in range(len(rated_geo))]
    rated_geo['score'] = new_col

    min_lon = rated_geo['longitude'].min()
    max_lon = rated_geo['longitude'].max()
    min_lat = rated_geo['latitude'].min()
    max_lat = rated_geo['latitude'].max()
    max_score = rated_geo['score'].max()
    min_score = rated_geo['score'].min()

    mean = np.mean(rated_geo['latitude'])
    sd = np.sqrt(np.sum(rated_geo['score']*(rated_geo['latitude']-mean)**2)/np.sum(rated_geo['score']))
    print("Average lattitude {}".format(np.sum(rated_geo['score']*rated_geo['latitude'])/np.sum(rated_geo['score'])))
    print("SD latitude {}".format(sd))
    
    bound = ((min_lon, max_lon, min_lat, max_lat))
    map_bound = ((-74.05, -73.948, 40.682, 40.79))
    # DO NOT MODIFY THIS BLOCK
    # Read in the base map and setting up subplot
    # DO NOT MODIFY THESE LINES
    basemap = plt.imread('../data/ny/ny.jpg')    

    plt.xlim(map_bound[0],map_bound[1])
    plt.ylim(map_bound[2],map_bound[3])
    # DO NOT MODIFY THESE LINES
    # Create the hexbin plot
    plt.scatter('longitude', 'latitude', data = rated_geo, s=rated_geo['score']**0.5)
    #hexbin = plt.hexbin('longitude', 'latitude', data = rated_geo, C=rated_geo['score'], gridsize = 200)
    #plt.colorbar(hexbin, orientation='vertical').set_label('Inspection Count')
    plt.xlabel('Longitude');
    plt.ylabel('Latitude');
    plt.title('Geospatial Density of Scores of Rated Restaurants');
    # Setting aspect ratio and plotting the hexbins on top of the base map layer
    # DO NOT MODIFY THIS LINE
    plt.imshow(basemap, zorder=0, extent = map_bound, aspect= 'equal');

def get_data(file_name=''):
    if not file_name:
        file_name = glob.glob("../logs/epoch_data/*")[0]
    day_0 = pickle.load(open(file_name,"rb"))

    return day_0

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a*b != 0, a * np.log(a / b), 0))

def get_KL_divergence(data):
    all_requests = histogram_of_locations(data)
    accepted = histogram_of_locations(data,accepted=True)
    return KL(all_requests,accepted)

def plot_entropy_profit_pareto(data_list,num_drivers):
    points = [(total_profit(data_list[i]),1/entropy(data_list[i],num_drivers)) for i in data_list]
    plt.scatter([i[0] for i in points],[i[1] for i in points])

    for i,name in enumerate(data_list):
        plt.annotate(name,points[i])

# Driver Utility  
def total_profit(data):
    return np.sum([sum([j[1] for j in i]) for i in data['epoch_each_agent_profit']])

# Driver Fairness
def gini(data,n=-1):
    num_drivers = data['settings']['num_agents']
    payment = payment_by_driver(data,num_drivers,n)
    mu = np.sum([i for i in payment.values()])/num_drivers
    
    a = 0
    for i in range(num_drivers):
        for j in range(num_drivers):
            a+=abs(payment[i]-payment[j])


    s =  a/(2*num_drivers**2 * mu)
    return s

# Rider utility 
def average_dropoff_delay(data):
    avg_time = np.sum(data['epoch_dropoff_delay'])/np.sum(data['epoch_requests_completed'])
    return avg_time

# Rider Utility #2
def requests_completed(data):
    return np.sum(data['epoch_requests_completed'])/np.sum(data['epoch_requests_seen'])

# Rider fairness 
def rider_fairness(data,clusters=10):
    zone_lat_long = open("../data/ny/zone_latlong.csv").read().split("\n")
    d = {}
    for i in zone_lat_long:
        if i!='':
            a,b,c = i.split(",")
            d[a] = (float(b),float(c))

    coords = [d[i] for i in d]
    regions = KMeans(n_clusters=clusters).fit(coords)
    labels = regions.labels_
    centers = regions.cluster_centers_

    loc_region = {}
    loc_requests = {}
    loc_acceptances = {}

    for i in range(len(labels)):
        loc_region[i] = labels[i]

    for i in set(labels):
        loc_requests[i] = 0
        loc_acceptances[i] = 0

    for i in data['epoch_locations_all']:
        for j in i:
            loc_requests[loc_region[j]]+=1

    for i in data['epoch_locations_accepted']:
        for j in i:
            loc_acceptances[loc_region[j]]+=1

    success = []
    for i in loc_requests:
        success.append(loc_acceptances[i]/loc_requests[i])

    return np.std(success)
    

loc_list = ["../logs/epoch_data/entropy","../logs/epoch_data/variance","../logs/epoch_data/baseline"]

all_files = []
for loc in loc_list:
    all_files+=glob.glob(loc+"/*.pkl")

all_pickles = [pickle.loads(open(i,"rb").read()) for i in all_files]

runs = {}

for data in all_pickles:
    name = str(data["settings"]["value_num"])
    if "lambda" in data["settings"]:
        name+="_"+str(data["settings"]["lambda"])
    if "training_days" in data["settings"]:
        name+="_"+str(data["settings"]["training_days"])
    if "nn_inputs" in data["settings"]:
        name+="_nn"

    print(data["settings"])

    runs[name] = (total_profit(data),gini(data),average_dropoff_delay(data),requests_completed(data),rider_fairness(data))  

variable_names = ["profit","gini","dropoff delay","requests completion","completion variance"]
direction = [1,-1,-1,1,-1]

print("Variable names {}".format(variable_names))

pareto_frontier = []
for name in runs:
    pareto = True
    for other_name in runs:
        is_bigger = False
        if other_name != name:
            for d in range(len(direction)):
                if runs[name][d]*direction[d]>=runs[other_name][d]*direction[d]:
                    is_bigger = True
            if not is_bigger:
                pareto = False
    if pareto:
        pareto_frontier.append(name)

for i in pareto_frontier:
    print(i,runs[i])
