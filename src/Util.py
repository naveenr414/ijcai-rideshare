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
    if previous_reward == 0:
        previous_reward = 10**-6
    ybar = np.mean(envt.driver_profits)
    N = len(envt.driver_profits)
    R = get_profit_action(envt,action)   
    return np.log((ybar+R/N)/ybar * ((y+R)/y)**(-1/N))
    
def get_entropy(envt):
    driver_profits = np.array(envt.driver_profits)+10**-6
    N = len(envt.driver_profits)
    ybar = np.mean(envt.driver_profits)
    
    if ybar!=0:
        return -1/(N) * np.sum(np.log(driver_profits/ybar))
    else:
        return 0
