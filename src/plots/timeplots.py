def plot_daily(y):
    x = np.arange(0,24,1/60)
    plt.plot(x,y,label=current_label)

def plot_running_mean(y):
    x = np.arange(0,24,1/60)
    plt.plot(x,running_mean(y,30),label=current_label)

def plot_requests_over_time(data):
    plot_running_mean(data['epoch_requests_seen'])

def plot_requests_accepted_over_time(data):
    plot_running_mean(data['epoch_requests_accepted'])
    
def plot_requests_completed_over_time(data):
    plot_running_mean(data['epoch_requests_completed'])

def plot_average_wait_time(data):
    y = np.array(data['epoch_dropoff_delay'])/np.array(data['epoch_requests_completed'])
    plot_running_mean(y)

def plot_average_wait_time_box(data):
    x = np.arange(0,24,1/60)
    y = np.array(data['epoch_dropoff_delay'])/np.array(data['epoch_requests_completed'])

    hourly_data = [y[i:i+60] for i in range(0,len(y),60)]
    hourly_data = [i[~np.isnan(i)] for i in hourly_data] 
    plt.boxplot(hourly_data)

def plot_service_rates(data_list,labels):
    service_rates = [np.sum(i['epoch_requests_accepted'])/np.sum(i['epoch_requests_seen']) for i in data_list]
    plt.bar(list(range(len(labels))),service_rates,tick_label=labels)

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
