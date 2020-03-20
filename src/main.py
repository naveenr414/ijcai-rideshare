from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
from ValueFunction import PathBasedNN, RewardPlusDelay, NeuralNetworkBased
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

def profit_function(travel_time):
    minutes_driven = travel_time/60
    return round(minutes_driven+5,2)

def get_profit_distribution(scored_final_actions):
    profits = []
    for action,_ in scored_final_actions:
        # Calculate the profit 
        for request in action.requests:
            dropoff = request.dropoff
            pickup = request.pickup
            travel_time = envt.get_travel_time(pickup,dropoff)
            profits.append(profit_function(travel_time))
    return profits

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
    request_generator = envt.get_request_batch(DAY)
    total_value_generated = 0
    num_total_requests = 0

    ret_dictionary = {'epoch_requests_completed':[],
                      'epoch_requests_accepted':[],
                      'epoch_dropoff_delay':[],
                      'epoch_requests_seen':[],
                      'epoch_driver_0_empty':[],
                      'epoch_requests_accepted_profit':[]}

    all_profits = []
    while True:
        # Get new requests
        try:
            current_requests = next(request_generator)
            print("Current time: {} or {} on DAY {}".format(envt.current_time,datetime.timedelta(seconds=envt.current_time),DAY))
            print("Number of new requests: {}".format(len(current_requests)))
        except StopIteration:
            break

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

        # Calculate reward for selected actions
        rewards = []
        for action, _ in scored_final_actions:
            reward = envt.get_reward(action)
            rewards.append(reward)
            total_value_generated += reward
        print("Reward for epoch: {}".format(sum(rewards)))

        profits = get_profit_distribution(scored_final_actions)
        all_profits+=profits
        print(all_profits)

        # Update
        if (is_training):
            # Update replay buffer
            value_function.remember(experience)

            # Update value function every TRAINING_FREQUENCY timesteps
            if ((int(envt.current_time) / int(envt.EPOCH_LENGTH)) % TRAINING_FREQUENCY == TRAINING_FREQUENCY - 1):
                value_function.update(central_agent)

                # Diagnostics
                """for action, score in scored_actions_all_agents[0]:
                    print("{}: {}, {}, {}".format(score, action.requests, action.new_path, action.new_path.total_delay))
                print()
                for idx, (action, score) in enumerate(scored_final_actions[:10]):
                    print("{}: {}, {}, {}".format(score, action.requests, action.new_path, action.new_path.total_delay))
                """

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
        ret_dictionary['epoch_driver_0_empty'].append(agents[0].path.is_empty())
        ret_dictionary['epoch_requests_accepted_profit'].append(sum(profits))

    # Printing statistics for current epoch
    print('Number of requests accepted: {}'.format(total_value_generated))
    print('Number of requests seen: {}'.format(num_total_requests))


    ret_dictionary['total_requests_accepted'] = total_value_generated
    
    return ret_dictionary


if __name__ == '__main__':
    # pdb.set_trace()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--capacity', type=int, default=4)
    parser.add_argument('-n', '--numagents', type=int, default=1000)
    parser.add_argument('-d', '--pickupdelay', type=int, default=300)
    parser.add_argument('-t', '--decisioninterval', type=int, default=60)
    parser.add_argument('-m', '--modellocation', type=str)
    args = parser.parse_args()

    Request.MAX_PICKUP_DELAY = args.pickupdelay
    Request.MAX_DROPOFF_DELAY = 2 * args.pickupdelay

    # Constants
    START_HOUR: int = 0
    END_HOUR: int = 24
    NUM_EPOCHS: int = 1
    TRAINING_DAYS: List[int] = list(range(3, 10)) 
    VALID_DAYS: List[int] = [2]
    TEST_DAYS: List[int] = list(range(11, 16)) 
    VALID_FREQ: int = 4
    SAVE_FREQ: int = VALID_FREQ
    LOG_DIR: str = '../logs/{}agent_{}capacity_{}delay_{}interval/'.format(args.numagents, args.capacity, args.pickupdelay, args.decisioninterval)

    # Initialising components
    # TODO: Save start hour not start epoch
    envt = NYEnvironment(args.numagents, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600, MAX_CAPACITY=args.capacity, EPOCH_LENGTH=args.decisioninterval)
    oracle = Oracle(envt)
    central_agent = CentralAgent(envt)
    # value_function = PathBasedNN(envt, log_dir=LOG_DIR, load_model_loc=args.modellocation)
    value_function = RewardPlusDelay(DELAY_COEFFICIENT=1e-7, log_dir=LOG_DIR)

    max_test_score = 0
    for epoch_id in range(NUM_EPOCHS):
        for day in TRAINING_DAYS:
            epoch_data = run_epoch(envt, oracle, central_agent, value_function, day, is_training=True)
            total_requests_served = epoch_data['total_requests_accepted']
            print("\nDAY: {}, Requests: {}\n\n".format(day, total_requests_served))
            value_function.add_to_logs('requests_served', total_requests_served, envt.num_days_trained)

            # Check validation score every VALID_FREQ days
            if (envt.num_days_trained % VALID_FREQ == VALID_FREQ - 1):
                test_score = 0
                for day in VALID_DAYS:
                    total_requests_served = run_epoch(envt, oracle, central_agent, value_function, day, is_training=False)['total_requests_accepted']
                    print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
                    test_score += total_requests_served
                value_function.add_to_logs('validation_score', test_score, envt.num_days_trained)

                # TODO: Save results better
                if (isinstance(value_function, NeuralNetworkBased)):
                    if (test_score > max_test_score or (envt.num_days_trained % SAVE_FREQ) == (SAVE_FREQ - 1)):
                        value_function.model.save('../models/{}_{}agent_{}capacity_{}delay_{}interval_{}_{}.h5'.format(type(value_function).__name__, args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, envt.num_days_trained, test_score))
                        max_test_score = test_score if test_score > max_test_score else max_test_score

            envt.num_days_trained += 1

    # CHECK TEST SCORE
    # value_function_baseline = RewardPlusDelay(DELAY_COEFFICIENT=1e-7, log_dir=LOG_DIR)

    for day in TEST_DAYS:
        # Initialising agents
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training=False)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

        epoch_data = run_epoch(envt, oracle, central_agent, value_function, day, is_training=False, agents_predefined=agents)
        total_requests_served = epoch_data['total_requests_accepted']
        print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
        pickle.dump(epoch_data,open("../logs/epoch_data/day_{}_epoch_data.pkl".format(day),"wb"))
        value_function.add_to_logs('test_requests_served', total_requests_served, envt.num_days_trained)

        # total_requests_served = run_epoch(envt, oracle, central_agent, value_function_baseline, day, is_training=False, agents_predefined=agents)
        # print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
        # value_function_baseline.add_to_logs('test_requests_served', total_requests_served, envt.num_days_trained)

        envt.num_days_trained += 1
