from Request import Request
from Action import Action
from LearningAgent import LearningAgent
from Environment import Environment
import Settings
import Util 

from typing import List, Dict, Tuple, Set, Any, Optional, Callable

from docplex.mp.model import Model  # type: ignore
from docplex.mp.linear import Var  # type: ignore
from random import gauss, shuffle, randint, random
import numpy as np
import time
from math import factorial
from copy import deepcopy
import random

one_permutation_shapley = []
truncated_shapley = []
random_shapley = []


class CentralAgent(object):
    """
    A CentralAgent arbitrates between different Agents.

    It takes the users 'preferences' for different actions as input
    and chooses the combination that maximises the sum of utilities
    for all the agents.

    It also trains the Learning Agents' shared value function by
    querying the rewards from the environment and the next state.
    """

    def __init__(self, envt: Environment, is_epsilon_greedy: bool=False):
        super(CentralAgent, self).__init__()
        self.envt = envt
        self._choose = self._epsilon_greedy if is_epsilon_greedy else self._additive_noise
        self.one_permutation_shapley_final = [0 for i in range(envt.NUM_AGENTS)]
        self.truncated_shapley_final = [0 for i in range(envt.NUM_AGENTS)]
        self.random_shapley_final = [0 for i in range(envt.NUM_AGENTS)]
        self.mode = "test"

    def choose_actions(self, agent_action_choices: List[List[Tuple[Action, float]]], is_training: bool=True, epoch_num: int=1) -> List[Tuple[Action, float]]:
        return self._choose(agent_action_choices, is_training, epoch_num)

    def reset(self):
        self.one_permutation_shapley_final = [0 for i in range(self.envt.NUM_AGENTS)]
        self.truncated_shapley_final = [0 for i in range(self.envt.NUM_AGENTS)]
        self.random_shapley_final = [0 for i in range(self.envt.NUM_AGENTS)]
        self.mode = "test"


    def _epsilon_greedy(self, agent_action_choices: List[List[Tuple[Action, float]]], is_training: bool=True, epoch_num: int=1) -> List[Tuple[Action, float]]:
        # Decide whether or not to take random action
        rand_num = random()
        random_probability = 0.1 + max(0, 0.9 - 0.01 * epoch_num)

        if not is_training or (rand_num > random_probability):
            final_actions = self._choose_actions_ILP(agent_action_choices)
        else:
            final_actions = self._choose_actions_random(agent_action_choices)

        return final_actions

    def _additive_noise(self, agent_action_choices: List[List[Tuple[Action, float]]], is_training: bool=True, epoch_num: int=1) -> List[Tuple[Action, float]]:
        # Define noise function for exploration
        def get_noise(variable: Var) -> float:
            stdev = 1 + (4000 if 'x0,' in variable.get_name() else 1000) / ((epoch_num + 1) * self.envt.NUM_AGENTS)
            return abs(gauss(0, stdev)) if is_training else 0

        final_actions = self._choose_actions_ILP(agent_action_choices, get_noise=get_noise)

        return final_actions

    def get_score_ILP(self, agent_action_choices: List[List[Tuple[Action, float]]], which_to_choose: str, get_noise: Callable[[Var], float]=lambda x: 0) -> List[Tuple[Action, float]]:
        # Model as ILP
        model = Model()


        # For converting Action -> action_id and back
        action_to_id: Dict[Action, int] = {}
        id_to_action: Dict[int, Action] = {}
        action_profit: Dict[id, float] = {}
        current_action_id = 0

        # For constraint 2
        requests: Set[Request] = set()

        start_time = time.time()
        total_agents = 0

        # Create decision variables and their coefficients in the objective
        # There is a decision variable for each (Action, Agent).
        # The coefficient is the value associated with the decision variable
        decision_variables: Dict[int, Dict[int, Tuple[Any, float]]] = {}
        for agent_idx, scored_actions in enumerate(agent_action_choices):
            if(len(scored_actions)>1):
                total_agents+=1
                
            
            for action, value in scored_actions:
                # Convert action -> id if it hasn't already been done
                if action not in action_to_id:
                    action_to_id[action] = current_action_id
                    id_to_action[current_action_id] = action
                    if which_to_choose[total_agents-1] == '0':
                        action_profit[current_action_id] = 0
                    else:
                        action_profit[current_action_id] = Util.change_profit(self.envt,action)
                    current_action_id += 1

                    action_id = current_action_id - 1
                    decision_variables[action_id] = {}
                else:
                    action_id = action_to_id[action]

                # Update set of requests in actions
                for request in action.requests:
                    if request not in requests:
                        requests.add(request)

                # Create variable for (action_id, agent_id)
                variable = model.binary_var(name='x{},{}'.format(action_id, agent_idx))

                # Save to decision_variable data structure
                if which_to_choose[total_agents-1] == '0':
                    decision_variables[action_id][agent_idx] = (variable, 0)
                else:
                    decision_variables[action_id][agent_idx] = (variable, value)
    

        # Create Constraint 1: Only one action per Agent
        for agent_idx in range(len(agent_action_choices)):
            agent_specific_variables: List[Any] = []
            for action_dict in decision_variables.values():
                if agent_idx in action_dict:
                    agent_specific_variables.append(action_dict[agent_idx])
            model.add_constraint(model.sum(variable for variable, _ in agent_specific_variables) == 1)

        # Create Constraint 2: Only one action per Request
        for request in requests:
            relevent_action_dicts: List[Dict[int, Tuple[Any, float]]] = []
            for action_id in decision_variables:
                if (request in id_to_action[action_id].requests):
                    relevent_action_dicts.append(decision_variables[action_id])
            model.add_constraint(model.sum(variable for action_dict in relevent_action_dicts for variable, _ in action_dict.values()) <= 1)


        # Create Constraint 3: The difference in max and min salary < 100 + 0.2*max
        if Settings.has_value("add_constraints"):
            sorted_profits = sorted(self.envt.driver_profits)

            lower_bound = sorted_profits[len(sorted_profits)//10]
            upper_bound = sorted_profits[-1]
            
            for agent_idx in range(len(agent_action_choices)):
                previous_profit = self.envt.driver_profits[agent_idx]
                agent_specific_variables: List[Any] = []
                new_profits = []

                for action_id in decision_variables:
                    action_dict = decision_variables[action_id]
                    if agent_idx in action_dict:
                        new_profit = action_profit[action_id]
                        new_profits.append(new_profit)
                        agent_specific_variables.append(action_dict[agent_idx])
            
                if Settings.get_value("add_constraints") == "max":
                    model.add_constraint(model.sum(agent_specific_variables[i][0]*(new_profits[i] + previous_profit)  for i in range(len(agent_specific_variables)))>=(upper_bound*0.5-200))
                elif Settings.get_value("add_constraints") == "min":
                    model.add_constraint(model.sum(agent_specific_variables[i][0]*(new_profits[i] + previous_profit)  for i in range(len(agent_specific_variables)))<=(lower_bound*Settings.get_value("lambda")+200))


        # Create Objective
        score = model.sum((value + get_noise(variable)) * variable for action_dict in decision_variables.values() for (variable, value) in action_dict.values())
        model.maximize(score)

        # Solve ILP
        solution = model.solve()
        assert solution  # making sure that the model doesn't fail

        # Get vehicle specific actions from ILP solution
        assigned_actions: Dict[int, int] = {}
        for action_id, action_dict in decision_variables.items():
            for agent_idx, (variable, _) in action_dict.items():
                if (solution.get_value(variable) == 1):
                    assigned_actions[agent_idx] = action_id

        final_actions: List[Tuple[Action, float]] = []
        total_profit = 0
        for agent_idx in range(len(agent_action_choices)):
            assigned_action_id = assigned_actions[agent_idx]
            total_profit+=action_profit[assigned_action_id]
            assigned_action = id_to_action[assigned_action_id]
            scored_final_action = None
            for action, score in agent_action_choices[agent_idx]:
                if (action == assigned_action):
                    scored_final_action = (action, score)
                    break

            assert scored_final_action is not None
            final_actions.append(scored_final_action)    
        
        return total_profit

    def shapley(self,all_values,num,total_nums):
        tot = 0
        for i in range(2**(total_nums-1)):
            bin_others = bin(i)[2:]
            if(len(bin_others)<total_nums-1):
                bin_others = '0'*(total_nums-1-len(bin_others))+bin_others
            S = bin_others.count('1')
            coeff = factorial(S)*factorial(total_nums-S-1)/factorial(total_nums)
            num_with = bin_others[:num]+'1'+bin_others[num:]
            num_without = bin_others[:num]+'0'+bin_others[num:]
            tot+=(all_values[num_with]-all_values[num_without])*1/(2**(total_nums-1))
        return tot

    def truncated_shapley(self,true_shapley,total_nums,agent_action_choices,agent_nums,get_noise):
        agent_action_choices = deepcopy(agent_action_choices)
        predicted_shapley = [0 for i in range(total_nums)]
        total_shapley = [0 for i in range(total_nums)]
        num_runs = [0 for i in range(total_nums)]
        previous_predicted_shapley = deepcopy(predicted_shapley)
        errors = []
        differences = []
        i = 0
        for q in range(2):
            nums = [0 for k in range(total_nums)]
            permuted = [k for k in range(total_nums)]
            random.shuffle(permuted)
            previous_value = 0

            for j in permuted:
                new = deepcopy(nums)
                new[j] = 1
                new_value = self.get_score_ILP(agent_action_choices,''.join([str(k) for k in new]),get_noise)
                total_shapley[j]+= (new_value-previous_value)
                num_runs[j]+=1
                predicted_shapley[j] = total_shapley[j]/num_runs[j]
                previous_value = new_value
                nums = new
                errors.append(np.linalg.norm(np.array(predicted_shapley)-np.array(true_shapley)))
                differences.append(np.linalg.norm(np.array(previous_predicted_shapley)-np.array(predicted_shapley)))
                previous_predicted_shapley = deepcopy(predicted_shapley)
            i+=1
        for i in range(len(agent_nums)):
            self.truncated_shapley_final[agent_nums[i]]+=predicted_shapley[i]
        return predicted_shapley
    
    def random_shapley(self,true_shapley,total_nums,agent_action_choices,agent_nums,get_noise):
        agent_action_choices = deepcopy(agent_action_choices)
        predicted_shapley = [0 for i in range(total_nums)]
        total_shapley = [0 for i in range(total_nums)]
        num_runs = [0 for i in range(total_nums)]
        previous_predicted_shapley = deepcopy(predicted_shapley)
        errors = []
        differences = []
        i = 0

        total_value = self.get_score_ILP(agent_action_choices,'1'*total_nums,get_noise)
        for q in range(2):
            for current_num in range(total_nums):
                nums = [random.randint(0,1) for k in range(total_nums)]
                bigger_run = deepcopy(nums)
                smaller_run = deepcopy(nums)
                bigger_run[current_num] = 1
                smaller_run[current_num] = 0

                bigger_value = self.get_score_ILP(agent_action_choices,''.join([str(k) for k in bigger_run]),get_noise)
                smaller_value = self.get_score_ILP(agent_action_choices,''.join([str(k) for k in smaller_run]),get_noise)
                diff = bigger_value-smaller_value
                num_runs[current_num]+=1
                total_shapley[current_num]+=diff
                predicted_shapley[current_num] = total_shapley[current_num]/(i+1)
                for k in range(2):
                    errors.append(np.linalg.norm(np.array(predicted_shapley)-np.array(true_shapley)))
                    differences.append(np.linalg.norm(np.array(previous_predicted_shapley)-np.array(predicted_shapley)))

                previous_predicted_shapley = deepcopy(predicted_shapley)
            i+=1

        normalization_constant = total_value/np.sum(predicted_shapley)
        if np.sum(predicted_shapley) != 0:
            predicted_shapley = [i*normalization_constant for i in predicted_shapley]

            for i in range(len(agent_nums)):
                self.random_shapley_final[agent_nums[i]]+=predicted_shapley[i]

        return predicted_shapley
        

    def one_permutation(self,true_shapley,total_nums,agent_action_choices,agent_nums,get_noise):
        agent_action_choices = deepcopy(agent_action_choices)
        predicted_shapley = [0 for i in range(total_nums)]
        total_shapley = [0 for i in range(total_nums)]
        num_runs = [0 for i in range(total_nums)]
        previous_predicted_shapley = deepcopy(predicted_shapley)
        errors = []
        differences = []
        i = 0
        total_value = self.get_score_ILP(agent_action_choices,'1'*total_nums,get_noise)
        for q in range(2):
            nums = [random.randint(0,1) for k in range(total_nums)]
            const_value = self.get_score_ILP(agent_action_choices,''.join([str(k) for k in nums]),get_noise)
            for current_num in range(total_nums):
                other_run = deepcopy(nums)
                other_run[current_num] = 1-nums[current_num]

                other_value = self.get_score_ILP(agent_action_choices,''.join([str(k) for k in other_run]),get_noise)
                if nums[current_num] == 1:
                    diff = const_value-other_value
                else:
                    diff = other_value-const_value
                
                num_runs[current_num]+=1
                total_shapley[current_num]+=diff
                predicted_shapley[current_num] = total_shapley[current_num]/(i+1)
                errors.append(np.linalg.norm(np.array(predicted_shapley)-np.array(true_shapley)))
                differences.append(np.linalg.norm(np.array(previous_predicted_shapley)-np.array(predicted_shapley)))

                previous_predicted_shapley = deepcopy(predicted_shapley)
            i+=1
        normalization_constant = total_value/np.sum(predicted_shapley)
        if np.sum(predicted_shapley) != 0:
            predicted_shapley = [i*normalization_constant for i in predicted_shapley]

            for i in range(len(agent_nums)):
                self.one_permutation_shapley_final[agent_nums[i]]+=predicted_shapley[i]

        return predicted_shapley
        
    def _choose_actions_ILP(self, agent_action_choices: List[List[Tuple[Action, float]]], get_noise: Callable[[Var], float]=lambda x: 0) -> List[Tuple[Action, float]]:
        # Model as ILP
        model = Model()


        # For converting Action -> action_id and back
        action_to_id: Dict[Action, int] = {}
        id_to_action: Dict[int, Action] = {}
        action_profit: Dict[id, float] = {}
        current_action_id = 0

        # For constraint 2
        requests: Set[Request] = set()

        start_time = time.time()
        total_agents = 0

        # Create decision variables and their coefficients in the objective
        # There is a decision variable for each (Action, Agent).
        # The coefficient is the value associated with the decision variable
        decision_variables: Dict[int, Dict[int, Tuple[Any, float]]] = {}
        agent_nums = []
        for agent_idx, scored_actions in enumerate(agent_action_choices):
            if(len(scored_actions)>1):
                total_agents+=1
                agent_nums.append(agent_idx)
            
            for action, value in scored_actions:
                # Convert action -> id if it hasn't already been done
                if action not in action_to_id:
                    action_to_id[action] = current_action_id
                    id_to_action[current_action_id] = action
                    action_profit[current_action_id] = Util.change_profit(self.envt,action)
                    current_action_id += 1

                    action_id = current_action_id - 1
                    decision_variables[action_id] = {}
                else:
                    action_id = action_to_id[action]

                # Update set of requests in actions
                for request in action.requests:
                    if request not in requests:
                        requests.add(request)

                # Create variable for (action_id, agent_id)
                variable = model.binary_var(name='x{},{}'.format(action_id, agent_idx))

                # Save to decision_variable data structure
                decision_variables[action_id][agent_idx] = (variable, value)
        compare_to_brute_force = False
        calculate_shapley=False
        if compare_to_brute_force:
    
            # Calculate Shapley
            if total_agents<=12:
                # Let's brute force
                all_vals = {}
                for i in range(2**total_agents):
                    current_binary = ('0'*total_agents + bin(i)[2:])[-total_agents:]
                    all_vals[current_binary] = self.get_score_ILP(agent_action_choices,current_binary,get_noise)

                true_shapley = []

                for i in range(total_agents):
                    true_shapley.append(self.shapley(all_vals,i,total_agents))
                """
                true_shapley = [0 for i in range(total_agents)]
                """
                print("Truncated")
                self.truncated_shapley(true_shapley,total_agents,agent_action_choices,agent_nums,get_noise)   
                print("Random")
                self.random_shapley(true_shapley,total_agents,agent_action_choices,agent_nums,get_noise)
                print("One permutation")
                self.one_permutation(true_shapley,total_agents,agent_action_choices,agent_nums,get_noise)
                print("True Shapley {}".format(true_shapley))
        elif total_agents>0 and self.mode == "test" and calculate_shapley:
            print("HERE!!!")
            self.truncated_shapley([0 for i in range(total_agents)],total_agents,agent_action_choices,agent_nums,get_noise)   
            self.random_shapley([0 for i in range(total_agents)],total_agents,agent_action_choices,agent_nums,get_noise)
            self.one_permutation([0 for i in range(total_agents)],total_agents,agent_action_choices,agent_nums,get_noise)    
    
        # Create Constraint 1: Only one action per Agent
        for agent_idx in range(len(agent_action_choices)):
            agent_specific_variables: List[Any] = []
            for action_dict in decision_variables.values():
                if agent_idx in action_dict:
                    agent_specific_variables.append(action_dict[agent_idx])
            model.add_constraint(model.sum(variable for variable, _ in agent_specific_variables) == 1)

        # Create Constraint 2: Only one action per Request
        for request in requests:
            relevent_action_dicts: List[Dict[int, Tuple[Any, float]]] = []
            for action_id in decision_variables:
                if (request in id_to_action[action_id].requests):
                    relevent_action_dicts.append(decision_variables[action_id])
            model.add_constraint(model.sum(variable for action_dict in relevent_action_dicts for variable, _ in action_dict.values()) <= 1)


        # Create Constraint 3: The difference in max and min salary < 100 + 0.2*max
        if Settings.has_value("add_constraints"):
            sorted_profits = sorted(self.envt.driver_profits)

            lower_bound = sorted_profits[len(sorted_profits)//10]
            upper_bound = sorted_profits[-1]
            
            for agent_idx in range(len(agent_action_choices)):
                previous_profit = self.envt.driver_profits[agent_idx]
                agent_specific_variables: List[Any] = []
                new_profits = []

                for action_id in decision_variables:
                    action_dict = decision_variables[action_id]
                    if agent_idx in action_dict:
                        new_profit = action_profit[action_id]
                        new_profits.append(new_profit)
                        agent_specific_variables.append(action_dict[agent_idx])
            
                if Settings.get_value("add_constraints") == "max":
                    model.add_constraint(model.sum(agent_specific_variables[i][0]*(new_profits[i] + previous_profit)  for i in range(len(agent_specific_variables)))>=(upper_bound*0.5-200))
                elif Settings.get_value("add_constraints") == "min":
                    model.add_constraint(model.sum(agent_specific_variables[i][0]*(new_profits[i] + previous_profit)  for i in range(len(agent_specific_variables)))<=(lower_bound*Settings.get_value("lambda")+200))


        # Create Objective
        score = model.sum((value + get_noise(variable)) * variable for action_dict in decision_variables.values() for (variable, value) in action_dict.values())
        model.maximize(score)

        # Solve ILP
        solution = model.solve()
        assert solution  # making sure that the model doesn't fail

        # Get vehicle specific actions from ILP solution
        assigned_actions: Dict[int, int] = {}
        for action_id, action_dict in decision_variables.items():
            for agent_idx, (variable, _) in action_dict.items():
                if (solution.get_value(variable) == 1):
                    assigned_actions[agent_idx] = action_id

        final_actions: List[Tuple[Action, float]] = []
        for agent_idx in range(len(agent_action_choices)):
            assigned_action_id = assigned_actions[agent_idx]
            assigned_action = id_to_action[assigned_action_id]
            scored_final_action = None
            for action, score in agent_action_choices[agent_idx]:
                if (action == assigned_action):
                    scored_final_action = (action, score)
                    break

            assert scored_final_action is not None
            final_actions.append(scored_final_action)

        total_score = sum(i[1] for i in final_actions)        

        
        return final_actions

    def _choose_actions_random(self, agent_action_choices: List[List[Tuple[Action, float]]]) -> List[Tuple[Action, float]]:
        final_actions: List[Optional[Tuple[Action, float]]] = [None] * len(agent_action_choices)
        consumed_requests: Set[Request] = set()

        # Create a random ordering
        order = list(range(len(agent_action_choices)))
        shuffle(order)

        # Pick agents in a random order
        for agent_idx in order:
            # Create a list of feasible actions
            allowable_actions_idxs: List[int] = []
            for action_idx, (action, _) in enumerate(agent_action_choices[agent_idx]):
                is_not_consumed = [(request in consumed_requests) for request in action.requests]
                if sum(is_not_consumed) == 0:
                    allowable_actions_idxs.append(action_idx)

            # Pick a random feasible action
            final_action_idx = randint(0, len(allowable_actions_idxs) - 1)
            final_action, score = agent_action_choices[agent_idx][final_action_idx]
            final_actions[agent_idx] = (final_action, score)

            # Update inefasible action information
            for request in final_action.requests:
                consumed_requests.add(request)

        for action in final_actions:  # type: ignore
            assert action is not None

        return final_actions  # type: ignore
