import Settings
Settings.read_from_arguments()

from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
import ValueFunction
from Experience import Experience
from Request import Request

from typing import List

import pdb
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import Pool
import argparse
import pickle
import datetime
import numpy as np
import time
import datetime

start = time.time()

# Get statistics by simulating the next epoch 
def get_statistics_next_epoch(agent,envt):
    ret_dictionary = {'total_delivery_delay':0,'requests_served':0}
    start_time = envt.current_time
    current_time = envt.current_time + agent.position.time_to_next_location
    current_location = agent.position.next_location
    current_capacity = agent.path.current_capacity

    for node_idx, node in enumerate(agent.path.request_order):
        next_location, deadline = agent.path.get_info(node)

        # Delay related checks
        travel_time = envt.get_travel_time(current_location, next_location)
        if (current_time + travel_time > deadline):
            return invalid_path_trace('Does not meet deadline at node {}'.format(node_idx))

        current_time += travel_time
        current_location = next_location

        if current_time-start_time>envt.EPOCH_LENGTH:
            break

        # Updating available delay
        if (node.expected_visit_time != current_time):
            invalid_path_trace("(Ignored) Visit time incorrect at node {}".format(node_idx))
            node.expected_visit_time = current_time

        if (node.is_dropoff):
            ret_dictionary['total_delivery_delay']+=deadline - node.expected_visit_time
            ret_dictionary['requests_served']+=1

        # Capacity related checks
        if (current_capacity > envt.MAX_CAPACITY):
            return invalid_path_trace('Exceeds MAX_CAPACITY at node {}'.format(node_idx))

        if (node.is_dropoff):
            next_capacity = current_capacity - 1
        else:
            next_capacity = current_capacity + 1
        if (node.current_capacity != next_capacity):
            invalid_path_trace("(Ignored) Capacity incorrect at node {}".format(node_idx))
            node.current_capacity = next_capacity
        current_capacity = node.current_capacity

    return ret_dictionary

def get_profit_distribution(scored_final_actions):
    profits = []
    agent_profits = []
    for agent, (action,_) in enumerate(scored_final_actions):
        # Calculate the profit 
        for request in action.requests:
            dropoff = request.dropoff
            pickup = request.pickup
            travel_time = envt.get_travel_time(pickup,dropoff)
            action_profit = envt.profit_function(travel_time)

            if action_profit!=0:
                profits.append(action_profit)
                agent_profits.append((agent,action_profit))
            
            
    return profits,agent_profits

def run_epoch(envt,
              oracle,
              central_agent,
              value_function,
              DAY,
              is_training,
              agents_predefined=None,
              TRAINING_FREQUENCY: int=1):

    # INITIALISATIONS
    Experience.envt = envt
    # Initialising agents
    if agents_predefined is not None:
        agents = deepcopy(agents_predefined)
    else:
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

    # ITERATING OVER TIMESTEPS
    print("DAY: {}".format(DAY))
    down_sample = 1
    if Settings.has_value("down_sample") and Settings.get_value("down_sample"):
        down_sample = Settings.get_value("down_sample")
    request_generator = envt.get_request_batch(DAY,downsample=down_sample)
    total_value_generated = 0
    num_total_requests = 0

    ret_dictionary = {'epoch_requests_completed':[],
                      'epoch_requests_accepted':[],
                      'epoch_dropoff_delay':[],
                      'epoch_requests_seen':[],
                      'epoch_requests_accepted_profit':[],
                      'epoch_each_agent_profit':[],
                      'epoch_locations_all':[],
                      'epoch_locations_accepted':[],}

    while True:
        # Get new requests
        try:
            current_requests = next(request_generator)

            if Settings.has_value("print_verbose") and Settings.get_value("print_verbose"):
                print("Current time: {} or {} on DAY {}".format(envt.current_time,datetime.timedelta(seconds=envt.current_time),DAY))
                print("Number of new requests: {}".format(len(current_requests)))
        except StopIteration:
            break

        ret_dictionary['epoch_locations_all'].append([i.pickup for i in current_requests])

        for i in current_requests:
            envt.requests_region[envt.labels[i.pickup]]+=1

        # Get feasible actions
        feasible_actions_all_agents = oracle.get_feasible_actions(agents, current_requests)

        # Score feasible actions
        experience = Experience(deepcopy(agents), feasible_actions_all_agents, envt.current_time, len(current_requests))
        scored_actions_all_agents = value_function.get_value([experience])

        # Choose actions for each agent
        scored_final_actions = central_agent.choose_actions(scored_actions_all_agents, is_training=is_training, epoch_num=envt.num_days_trained)

        # Assign final actions to agents
        for agent_idx, (action, _) in enumerate(scored_final_actions):
            agents[agent_idx].path = deepcopy(action.new_path)

            position = experience.agents[agent_idx].position.next_location
            time_driven = 0
            for request in action.requests:
                time_driven+=envt.get_travel_time(request.pickup,request.dropoff)

            time_to_request = sum([envt.get_travel_time(position,request.pickup) for request in action.requests])

            envt.driver_utilities[agent_idx]+=max((time_driven-time_to_request),0)

        # Calculate reward for selected actions
        rewards = []
        locations_served = []
        for action, _ in scored_final_actions:
            reward = len(action.requests)
            locations_served += [request.pickup for request in action.requests]
            for request in action.requests:
                envt.success_region[envt.labels[request.pickup]]+=1
            rewards.append(reward)
            total_value_generated += reward

        if Settings.has_value("print_verbose") and Settings.get_value("print_verbose"):
            print("Reward for epoch: {}".format(sum(rewards)))

        profits,agent_profits = get_profit_distribution(scored_final_actions)            
        for i,j in agent_profits:
            envt.driver_profits[i]+=j

        # Update
        if (is_training):
            # Update replay buffer
            value_function.remember(experience)

            # Update value function every TRAINING_FREQUENCY timesteps
            if ((int(envt.current_time) / int(envt.EPOCH_LENGTH)) % TRAINING_FREQUENCY == TRAINING_FREQUENCY - 1):
                value_function.update(central_agent)

        # Sanity check
        for agent in agents:
            assert envt.has_valid_path(agent)

        # Writing statistics to logs
        value_function.add_to_logs('rewards_day_{}'.format(envt.num_days_trained), sum(rewards), envt.current_time)
        avg_capacity = sum([agent.path.current_capacity for agent in agents]) / envt.NUM_AGENTS
        value_function.add_to_logs('avg_capacity_day_{}'.format(envt.num_days_trained), avg_capacity, envt.current_time)

        epoch_dictionary = {}
        for agent in agents:
            agent_dictionary = get_statistics_next_epoch(agent,envt)
            if epoch_dictionary == {}:
                epoch_dictionary = agent_dictionary
            else:
                for key in agent_dictionary:
                    epoch_dictionary[key]+=agent_dictionary[key]

        # Simulate the passing of time
        envt.simulate_motion(agents, current_requests)
        num_total_requests += len(current_requests)

        ret_dictionary['epoch_requests_completed'].append(epoch_dictionary['requests_served'])
        ret_dictionary['epoch_dropoff_delay'].append(epoch_dictionary['total_delivery_delay'])
        ret_dictionary['epoch_requests_accepted'].append(sum(rewards))
        ret_dictionary['epoch_requests_seen'].append(len(current_requests))
        ret_dictionary['epoch_requests_accepted_profit'].append(sum(profits))
        ret_dictionary['epoch_each_agent_profit'].append(agent_profits)
        ret_dictionary['epoch_locations_accepted'].append(locations_served)

        if Settings.has_value("print_verbose") and Settings.get_value("print_verbose"):
            print("Requests served {}".format(np.sum(ret_dictionary["epoch_requests_completed"])))
            print("Requests accepted {}".format(sum(rewards)))

    # Printing statistics for current epoch
    print('Number of requests accepted: {}'.format(total_value_generated))
    print('Number of requests seen: {}'.format(num_total_requests))


    ret_dictionary['total_requests_accepted'] = total_value_generated
    
    return ret_dictionary


if __name__ == '__main__':
    # pdb.set_trace()

    PICKUP_DELAY = 300
    CAPACITY = 4
    DECISION_INTERVAL = 60
    START_HOUR: int = 0
    END_HOUR: int = 24
    NUM_EPOCHS: int = 1
    CAPACITY = 4
    PICKUP_DELAY = 300
    DECISION_INTERVAL=60
    VALID_DAYS: List[int] = [2]
    VALID_FREQ: int = 4
    SAVE_FREQ: int = VALID_FREQ
    Request.MAX_PICKUP_DELAY = PICKUP_DELAY
    Request.MAX_DROPOFF_DELAY = 2 * PICKUP_DELAY
    NEURAL_VALUE_FUNCTIONS = [1,8]

    # Load in different settings
    training_days = Settings.get_value("training_days")
    testing_days = Settings.get_value("testing_days")
    num_agents = Settings.get_value("num_agents")
    write_to_file = Settings.get_value("write_file")
    value_num = Settings.get_value("value_num")

    if Settings.has_value("pickup_delay"):
        PICKUP_DELAY = Settings.get_value("pickup_delay")

    TRAINING_DAYS: List[int] = list(range(3, 3+training_days))
    TEST_DAYS: List[int] = list(range(11, 11+testing_days))

    # Initialising components
    # TODO: Save start hour not start epoch
    envt = NYEnvironment(num_agents, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600,
                         MAX_CAPACITY=CAPACITY, EPOCH_LENGTH=DECISION_INTERVAL)
    oracle = Oracle(envt)
    central_agent = CentralAgent(envt)
    central_agent.mode = "train"
    value_function = ValueFunction.num_to_value_function(envt,value_num)

    print("Input settings {}".format(Settings.settings_list))

    max_test_score = 0
    for epoch_id in range(NUM_EPOCHS):
        for day in TRAINING_DAYS:
            epoch_data = run_epoch(envt, oracle, central_agent, value_function, day, is_training=True)
            total_requests_served = epoch_data['total_requests_accepted']
            print("DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
            value_function.add_to_logs('requests_served', total_requests_served, envt.num_days_trained)

            # Check validation score every VALID_FREQ days
            if (envt.num_days_trained % VALID_FREQ == VALID_FREQ - 1):
                test_score = 0
                for day in VALID_DAYS:
                    total_requests_served = run_epoch(envt, oracle, central_agent, value_function, day, is_training=False)['total_requests_accepted']
                    print("(VALIDATION) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
                    test_score += total_requests_served
                value_function.add_to_logs('validation_score', test_score, envt.num_days_trained)

                # TODO: Save results better
                if (isinstance(value_function, NeuralNetworkBased)):
                    if (test_score > max_test_score or (envt.num_days_trained % SAVE_FREQ) == (SAVE_FREQ - 1)):
                        value_function.model.save('../models/{}_{}agent_{}capacity_{}delay_{}interval_{}_{}.h5'.format(type(value_function).__name__, numagents, args.capacity, args.pickupdelay, args.decisioninterval, envt.num_days_trained, test_score))
                        max_test_score = test_score if test_score > max_test_score else max_test_score

            envt.num_days_trained += 1
            if value_num in NEURAL_VALUE_FUNCTIONS:
                value_function.model.save('../models/{}_{}.h5'.format(num_agents,  envt.num_days_trained))

    # Reset the driver utilities
    envt.reset()
    central_agent.reset()

    for day in TEST_DAYS:
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training=False)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

        epoch_data = run_epoch(envt, oracle, central_agent, value_function, day, is_training=False, agents_predefined=agents)
        total_requests_served = epoch_data['total_requests_accepted']
        print("(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))

        # Write our pickled resutls 
        if write_to_file:
            epoch_data['settings'] = Settings.settings_list
            epoch_data['settings']['time'] = int(time.time()-start)
            epoch_data['truncated_shapley'] = central_agent.truncated_shapley_final
            epoch_data['random_shapley'] = central_agent.random_shapley_final
            epoch_data['one_permutation_shapley'] = central_agent.one_permutation_shapley_final
            file_name = str(datetime.datetime.now()).split(".")[0].replace(" ","_").replace(":","")
            pickle.dump(epoch_data,open("../logs/epoch_data/"+file_name+".pkl","wb"))
            
        value_function.add_to_logs('test_requests_served', total_requests_served, envt.num_days_trained)

        envt.num_days_trained += 1

print("Took {} seconds".format(int(time.time()-start)))
