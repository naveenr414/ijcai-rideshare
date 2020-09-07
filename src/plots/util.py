import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt 

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

def plot_bar(data,labels):
    plt.bar(list(range(len(data))),data)
    plt.xticks(list(range(len(data))),labels,fontsize=16)

def payment_by_driver(data,n=-1):
    if n == -1:
        n = len(data['epoch_each_agent_profit'])

    num_drivers = data['settings']['num_agents']
    driver_pays = {}
    for i in range(num_drivers):
        driver_pays[i] = 0
    for i in data['epoch_each_agent_profit'][:n]:
        for j,k in i:
            driver_pays[j]+=k

    return driver_pays

def income_25(data):
    a = payment_by_driver(data)
    a = list(a.values())
    a = sorted(a)
    return a[len(a)//4]

def entropy(data,num_drivers):
    payment_driver = np.array(list((payment_by_driver(data,num_drivers).values())))
    payment_driver+=10**-6
    ybar = np.mean(payment_driver)
    
    return -1/num_drivers * (np.sum(np.log(payment_driver/ybar)))

# Driver Utility  
def total_profit(data):
    return np.sum([sum([j[1] for j in i]) for i in data['epoch_each_agent_profit']])

# Driver Fairness
def gini(data,n=-1):
    num_drivers = data['settings']['num_agents']
    payment = payment_by_driver(data,n)
    mu = np.sum([i for i in payment.values()])/num_drivers
    
    a = 0
    for i in range(num_drivers):
        for j in range(num_drivers):
            a+=abs(payment[i]-payment[j])


    s =  a/(2*num_drivers**2 * mu)
    return s

# Another Driver Fairness
def std_income(data):
    num_drivers = data['settings']['num_agents']
    payment = payment_by_driver(data)
    return np.std(list(payment.values()))

# Rider utility 
def average_dropoff_delay(data):
    avg_time = np.sum(data['epoch_dropoff_delay'])/np.sum(data['epoch_requests_completed'])
    return avg_time

# Rider Utility #2
def requests_completed(data):
    return np.sum(data['epoch_requests_completed'])/np.sum(data['epoch_requests_seen'])

# Rider fairness 
def rider_fairness(data,clusters=10):
    success = get_region_percentages(data)

    return np.std(success)

def min_region(data):
    return min(get_region_percentages(data))

def get_region_percentages(data):
    loc_requests = {}
    loc_acceptances = {}

    for i in set(region_labels):
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
    return success

# Rider fairness 
def rider_min(data,clusters=10):
    success = get_region_percentages(data)

    return np.min(success)

def load_kmeans():
    global loc_region
    global coords
    global region_labels
    
    """Loading the KMeans regions"""
    zone_lat_long = open("../../data/ny/zone_latlong.csv").read().split("\n")
    d = {}
    for i in zone_lat_long:
        if i!='':
            a,b,c = i.split(",")
            d[a] = (float(b),float(c))

    coords = [d[i] for i in d]
    region_labels = pickle.loads(open("../../data/ny/new_labels.pkl","rb").read())

    loc_region = {}

    for i in range(len(region_labels)):
        loc_region[i] = region_labels[i]

    return loc_region

def get_pickle(file_name):
    return pickle.loads(open(file_name,"rb").read())

def get_name(data):
    name = str(data["settings"]["value_num"])
    if "lambda" in data["settings"]:
        name+="_"+str(round(data["settings"]["lambda"],1))
    if "training_days" in data["settings"]:
        name+="_"+str(data["settings"]["training_days"])
    if "nn_inputs" in data["settings"]:
        name+="_nn"
    if "pickup_delay" in data["settings"]:
        name+="_d{}".format(data["settings"]["pickup_delay"])
    return name

def get_nice_name(data):
    if(data['settings']['value_num'] == 1):
        return "NeurADP"
    elif (data['settings']['value_num'] in [7,8,9,10]):
        return "Driver Side"
    elif (data['settings']['value_num'] in [11,12,13,14]):
        return "Rider side"
    elif 'pickup_delay' in data['settings']:
        return "Pickup Constraint"
    elif 'add_constraints' in data['settings']:
        return "Income Constraint"
    elif data['settings']['value_num'] in [2]:
        return "Greedy Profit"


def get_data(get_baseline=True,get_rider_side=True,get_driver_side=True):
    baseline = ["../../logs/epoch_data/baseline"]
    rider_side = ["../../logs/epoch_data/waittime","../../logs/epoch_data/entropy_rider","../../logs/epoch_data/variance_rider"]
    driver_side = ["../../logs/epoch_data/variance","../../logs/epoch_data/income_hard","../../logs/epoch_data/entropy"]

    all_data = []
    if get_baseline:
        all_data+=baseline
    if get_rider_side:
        all_data+=rider_side
    if get_driver_side:
        all_data+=driver_side

    all_files = []
    for i in all_data:
        all_files+=glob.glob(i+"/*.pkl")
    all_pickles = [get_pickle(i) for i in all_files]
    return all_pickles

def get_shapley():
    all_files = glob.glob("../../logs/epoch_data/shapley_tests/*.pkl")
    all_pickles = [get_pickle(i) for i in all_files]
    return all_pickles

def get_robustness():
    all_files = glob.glob("../../logs/epoch_data/robustness/*.pkl")
    all_pickles = [get_pickle(i) for i in all_files]
    return all_pickles    

def get_profitz():
    all_files = glob.glob("../../logs/epoch_data/robustness_profitz/*.pkl")
    all_pickles = [get_pickle(i) for i in all_files]
    return all_pickles

def plot_over_newyork():
    all_requests = []
    loc_regions = pickle.loads(open("../../data/ny/new_labels.pkl","rb").read())

    rated_geo = pd.read_csv("../../data/ny/zone_latlong.csv",header=None,names=['zone','longitude','latitude'])
    min_lon = rated_geo['longitude'].min()
    max_lon = rated_geo['longitude'].max()
    min_lat = rated_geo['latitude'].min()
    max_lat = rated_geo['latitude'].max()

    mean = np.mean(rated_geo['latitude'])
    
    bound = ((min_lon, max_lon, min_lat, max_lat))
    map_bound = ((-74.05, -73.948, 40.682, 40.79))
    colors = ['b','g','r','c','m','y','k','w','#888888','tab:pink']
    basemap = plt.imread('../../data/ny/ny.jpg')    

    plt.xlim(map_bound[0],map_bound[1])
    plt.ylim(map_bound[2],map_bound[3])

    for j in range(10):
        points = []
        for i in range(len(rated_geo)):
            if loc_regions[rated_geo['zone'][i]] == j:
                points.append(i)
        plt.scatter('longitude', 'latitude', data = rated_geo[rated_geo.index.isin(points)],c=colors[j])
    plt.xlabel('Longitude');
    plt.ylabel('Latitude');
    plt.title('Neighborhoods of New York');
    plt.imshow(basemap, zorder=0, extent = map_bound, aspect= 'equal');
    plt.show()

def write_kmeans():

    zone_lat_long = open("../../data/ny/zone_latlong.csv").read().split("\n")
    d = {}
    coords = []
    for i in zone_lat_long:
        if i!='':
            a,b,c = i.split(",")
            d[a] = (float(b),float(c))
            coords.append((float(b),float(c)))

    regions = KMeans(n_clusters=10).fit(coords)
    labels = regions.labels_
    centers = regions.cluster_centers_

    pickle.dump(labels,open("../../data/ny/new_labels.pkl","wb"))
