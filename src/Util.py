import numpy as np

def change_profit(envt,action):
    profit = 0

    for request in action.requests:
        dropoff = request.dropoff
        pickup = request.pickup
        travel_time = envt.get_travel_time(pickup,dropoff)
        action_profit = envt.profit_function(travel_time)
        profit+=action_profit

    return profit

def change_entropy(envt,action,driver_num):
    y = envt.driver_profits[driver_num]
    if y == 0:
        y = 10**-6
    ybar = np.mean(envt.driver_profits)
    N = len(envt.driver_profits)
    R = change_profit(envt,action)   
    return np.log((ybar+R/N)/ybar * ((y+R)/y)**(-1/N))

def change_entropy_rider(envt,action,driver_num):
    percent_success = []
    for i in range(len(envt.requests_region)):
        if envt.requests_region[i] == 0:
            percent_success.append(0)
        else:
            percent_success.append(envt.success_region[i]/envt.requests_region[i])
    new_requests = [0 for i in range(len(envt.requests_region))]

    for i in action.requests:
        new_requests[envt.labels[i.pickup]]+=1
    
    current_entropy = get_entropy_list(percent_success)
    percent_success_new = []
    for i in range(len(envt.requests_region)):
        if(envt.requests_region[i]+new_requests[i]) == 0:
            percent_success_new.append(0)
        else:
            percent_success_new.append((envt.success_region[i]+new_requests[i])/(envt.requests_region[i]+new_requests[i]))
    new_entropy = get_entropy_list(percent_success_new)

    change_entropy = new_entropy-current_entropy
    return change_entropy

def change_variance_rider(envt,action,driver_num):
    percent_success = []
    for i in range(len(envt.requests_region)):
        if envt.requests_region[i] == 0:
            percent_success.append(0)
        else:
            percent_success.append(envt.success_region[i]/envt.requests_region[i])

    new_requests = [0 for i in range(len(envt.requests_region))]

    for i in action.requests:
        new_requests[envt.labels[i.pickup]]+=1
    
    current_variance = np.var(percent_success)
    percent_success_new = []
    for i in range(len(envt.requests_region)):
        if(envt.requests_region[i]+new_requests[i]) == 0:
            percent_success_new.append(0)
        else:
            percent_success_new.append((envt.success_region[i]+new_requests[i])/(envt.requests_region[i]+new_requests[i]))
    new_variance = np.var(percent_success_new)

    change_variance = new_variance-current_variance
    return change_variance
    
def get_entropy(envt):
    driver_profits = np.array(envt.driver_profits)+10**-6
    N = len(envt.driver_profits)
    ybar = np.mean(envt.driver_profits)
    
    if ybar!=0:
        return -1/(N) * np.sum(np.log(driver_profits/ybar))
    else:
        return 0

def get_entropy_list(l):
    ybar = np.mean(l)
    N = len(l)
    
    if ybar!=0:
        return -1/(N) * np.sum(np.log(l/ybar))
    else:
        return 0
    

def change_variance(envt,action,driver_num):
    R = change_profit(envt,action)
    y = envt.driver_profits[driver_num]
    if y == 0:
        y = 10**-6
    ybar = np.mean(envt.driver_profits)
    n = len(envt.driver_profits)
    return R*((n-1)/n**2) * (2*(y - ybar) + ((n-1)/n)*R + R/n) + 2*(y-ybar)*R/n**2
