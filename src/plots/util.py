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

def payment_by_driver(data,n):
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

def load_kmeans():
    global loc_region
    global coords
    
    """Loading the KMeans regions"""
    zone_lat_long = open("../data/ny/zone_latlong.csv").read().split("\n")
    d = {}
    for i in zone_lat_long:
        if i!='':
            a,b,c = i.split(",")
            d[a] = (float(b),float(c))

    coords = [d[i] for i in d]
    labels = pickle.loads(open("../data/ny/labels.pkl","rb").read())

    loc_region = {}

    for i in range(len(labels)):
        loc_region[i] = labels[i]

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
    baseline = ["../logs/epoch_data/baseline"]
    rider_side = ["../logs/epoch_data/waittime","../logs/epoch_data/entropy_rider","../logs/epoch_data/variance_rider"]
    driver_side = ["../logs/epoch_data/variance","../logs/epoch_data/income_hard","../logs/epoch_data/entropy"]

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
    return all_picles
