from typing import Mapping, Set, Sequence, Tuple, TypeVar, List, Generic, Optional
from MPv2 import MP
from MRP import MRP
from MDP import MDP
import numpy as np

#declare the generic State and Action types
S = TypeVar('S')
A = TypeVar('A')

SATSff = Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]]
SASTff = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]
SASf = Mapping[S, Mapping[A, Mapping[S, float]]]
SAf = Mapping[S, Mapping[A,float]] #policy
Sf = Mapping[S,float] #value function

class DP:

    #member variables
    world: Generic
    tol: float = 1e-5

    def __init__(
        self,
        mdp: Generic,
        tol: float
    ) -> None:
        self.world = mdp
        self.tol = tol

    def single_policy_evaluation_step(
        self,
        policy: SAf,
        vfk: Sf #state value function
    ) -> Sf:
        """ Perform policy evaluation to calculate the value function v_\pi """

        #first, declare a value function object (all zeros here)
        new_vf = {s: 0.0 for s in self.world.state_list}

        #given a state
        for s in self.world.state_list:
                        
            #sum over actions
            for a in list(self.world.P_map[s].keys()):

                #check action exists in policy at state s
                if a not in policy[s]: 
                    continue 

                #for each action caculate get the transition probability and reward for
                # moving from s to s'
                
                #for each s' get the transition prob
                for sp, prob in self.world.P_map[s][a].items():
                    
                    #get the reward
                    reward = self.world.R_map[s][a][sp]

                    #update vf
                    new_vf[s] += policy[s][a] * prob * (reward + self.world.gamma * vfk[sp])
 
        return new_vf
    
    def max_diff_dict(
        self,
        d1: Sf,
        d2: Sf
    ) -> float:
        """ compares common elements of two dictionaries to find the max absolute difference """
        common_keys = set(list(d1.keys())).intersection(list(d2.keys()))
        max_diff = 0
        for k in common_keys:
            diff = abs( d1[k] - d2[k])
            if diff > max_diff:
                max_diff = diff
        return max_diff

    def policy_evaluation(
        self,
        policy: SAf,
        #vf_guess: Optional[Sf] = {s: 0.0 for s in self.world.state_list},
        max_iter: Optional[int] = 1000
    ) -> Sf:
        """ perform many policy evaluation steps and terminate when tolorance is crossed"""

        count = 0
        max_diff = 1000
        vf_guess = {s: 0.0 for s in self.world.state_list}
        old_vf = vf_guess
        #loop
        while ( count <= max_iter and max_diff >= self.tol):
            
            #update value function
            new_vf = self.single_policy_evaluation_step(policy, old_vf)

            #calcualte difference 
            max_diff = self.max_diff_dict(new_vf,old_vf)

            #move it over
            old_vf = new_vf

            #update counter
            count += 1
            #print("iter {}, max diff {}:\n  value function: {} ".format(count,max_diff,new_vf) )
        
        print("     {} policy eval iterations".format(count))
        return new_vf


    # def get_state_action_vf_from_state_vf(
    #     self,
    #     vf: Sf
    # ) -> SAf:
    #     cacluate the state action value function 

    def greedy_policy_improvement(
        self,
        vf: Sf
    ) -> SAf:
        """ greedily improve the policy by choosing the best action for each state """
        #for each state, calculate the value of each actions

        #define the tracking SAf
        tracker = dict()

        #given a state calcualte the state action value function 
        for s in self.world.state_list:
            
            #set up internal dict
            tracker[s] = dict()

            #sum over actions
            for a in list(self.world.P_map[s].keys()):
                
                #initalize action key in internal dict
                tracker[s][a] = 0

                #for each s' get the transition prob and caluc expected value 
                for sp, prob in self.world.P_map[s][a].items():
                    
                    #get the reward
                    reward = self.world.R_map[s][a][sp]

                    #get the value
                    val_sp = vf[sp]

                    #update tracker
                    tracker[s][a] += prob * (reward + self.world.gamma * val_sp)
        
        #now choose the action with the highest value 
        # TODO: split probs if equal

        #declare new policy SAf
        new_pol = {s: dict() for s in self.world.state_list}

        #loop through states and choose best action
        for s in self.world.state_list:

            #choose best action
            best_action = None

            #loop through actions
            for a, val in tracker[s].items():
                
                #set best action if not set
                if best_action == None:
                    best_action = (a, val)
                else: #replace with new best action
                    if val > best_action[1]:
                        best_action = (a,val)
            
            #update policy dictionary to take best action (deterministic now)
            new_pol[s][a] = 1.0
        
        return new_pol
    
    def policy_iteration(
        self,
        policy: SAf,
        max_iter: Optional[int] = 1000
    ) -> Tuple[Sf, SAf]: #return optimal value function and policy
        """ see book for implimentation """
        #set up 
        count = 0
        changed = True
        old_pol = policy
        #start iterating baby
        while (count <= max_iter and changed ):
            print("policy iteration: {}".format(count))

            #first perform policy evaluation
            evaluated = self.policy_evaluation(old_pol)

            print("     v_pi: {}".format(evaluated))

            #then improve policy 
            improved_pol = self.greedy_policy_improvement( evaluated )

            #then compare
            if self.policy_compare(old_pol, improved_pol):
                changed = False
            
            #move around and add to count 
            old_pol = improved_pol
            count+=1
        
        #return the optimal value function and policy
        return (evaluated, improved_pol)

    
    def policy_compare(
        self,
        A:  SAf,
        B:  SAf
    ) -> bool:
        """ return true if the policies are the same 
            uses that prob of all actions given state must sum to 1, only check one direction"""
        #look at all states in one
        for s in list(A.keys()):

            #look at action, probability pair in the old policy
            for a, av in A[s].items():
                    
                #get value in new policy 
                if( (a in B[s]) ):
                    bv = B[s][a]
                    #compare probabilties. If they are not the same return false
                    if (av != bv):
                        return False
                #if A has a probability of action a in state s but b does not, return false
                elif av != 0.0:
                    return False
    
        #if we get here without returning false, they must be the same
        return True

    def value_iteration(
        self,
        eval_steps: Optional[int] = 1,     #number of policy evaluation steps to include
        max_iter: Optional[int] = 1000
        #consider adding a starting guess for the value function
    ) -> Tuple[Sf, SAf]:
        """ Start with a 0 vector value function """
        
        #start with a zero value function
        vf_old = {s: 0.0 for s in self.world.state_list}
        vf_new = {s: 0.0 for s in self.world.state_list}

        #set up control variables
        max_diff = 1000
        count = 0
        #loop until convergence or max iter
        while( max_diff > self.tol and count < max_iter ):

            #loop through each state
            for s in self.world.state_list:
                
                #track maximum action value pair
                best_action = None

                #loop through actions in each state
                for a in list(self.world.P_map[s].keys())

                    #tract weighted reward across states
                    this_actions_value = 0.0

                    #for each s' get the transition prob
                    for sp, prob in self.world.P_map[s][a].items():

                        #get the reward
                        reward = self.world.R_map[s][a][sp]

                        #increment reward 
                        this_actions_value += prob * (reward + self.world.gamma * vf_old[sp])

                    #update best action
                    if (best_action == None):
                        best_action = (a, this_actions_value)
                
                #update the value function
                vf_new[s] = best_action[1]

                #add to count
                count += 1
            
            #compare value functions
            max_diff = self.max_diff_dict(vf_old, vf_new)

            #move around
            vf_old = vf_new
        
        #after converging
        return vf_new
        #TODO return policy from the action, look at be action check from the internal dict
            





                    




if __name__ == '__main__':
    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    mdp_refined_data = {
        1: {
            'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
            'b': {2: (0.3, -0.5), 3: (0.7, 2.6)},
            'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}
        },
        2: {
            'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
            'c': {1: (0.2, 4.8), 2: (0.4, 9.2), 3: (0.4, -8.2)}
        },
        3: {
            'a': {3: (1.0, 0.0)},
            'b': {3: (1.0, 0.0)}
        }
    }
    gamma_val = 0.9
    mdp1_obj = MDP(mdp_refined_data, gamma_val)
    dp_obj = DP(mdp1_obj, 1e-3)
    value_func = dp_obj.policy_evaluation(policy_data)
    optimal_vf, optimal_pol = dp_obj.policy_iteration(policy_data)
    print("Optimal value function: \n {}".format(optimal_vf))
    print("Optimal policy: \n {}".format(optimal_pol))
    ## CHECK OUTPUT AGAINST SAME MDP IN ASHWINS code

