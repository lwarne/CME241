#final.py
from typing import Sequence, Tuple, Mapping
import numpy as np

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    #I will be implimenting every visit monte carlo policy evaluations
    #I will use two maps (dictionaries). one track state value, other tracks 
    state_to_value = dict()
    state_to_visit = dict()

    #loop through all the data, return is provided
    for tup in state_return_samples:
        
        #pull out values
        s = tup[0]
        r = tup[1]

        #update if present
        if s in state_to_visit:
            #add to visit count
            state_to_visit[s] += 1 
            #update value 
            state_to_value[s] = state_to_value[s] + 1/state_to_visit[s] * (r - state_to_value[s])

        #add to dictionaries if not present
        else:
            state_to_visit[s] = 1 #number of visits is now 1
            state_to_value[s] = r #estimated value is observe return
    
    #return ValueFunc
    return state_to_value


def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    #here I will calculate the transition probability and average reward
    
    #set up reward (incremental) and prob (to be divided later)
    state_state_count = dict()
    state_reward = dict()
    #set up counting dictionary
    state_count = dict()

    #loop though all state reward next state observations
    for tup in srs_samples:
        #extract info from sample
        s = tup[0]
        r = tup[1]
        sp = tup[2]
        
        #if present, update
        if s in state_count:
            state_count[s] += 1
            state_reward[s] += 1/state_count[s] * ( r - state_reward[s] )
            #if next state has been visited before, update
            if sp in state_state_count[s]:
                state_state_count[s][sp] += 1
            else: #otherwise add it
                state_state_count[s][sp] = 1
        #otherwise add it
        else:
            state_count[s] = 1
            state_state_count[s] = { sp : 1}
            state_reward[s] = r
    
    #after using all observations, divide by count to get probability
    prob_dict = { 
        s: { sp : (count / state_count[s]) for sp, count in inner_dict.items()} 
        for s, inner_dict in state_state_count.items()   
        }

    # print(prob_dict)

    return (prob_dict, state_reward)
    #ProbFunc = Mapping[S, Mapping[S, float]]
    #RewardFunc = Mapping[S, float]

def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    #we have a pair of incomming dictionaries, prob is not guaranteed to have all
        #the inside values filled. 

    #get observered states
    states_list = list(reward_func)
    # print("states list: {}".format(states_list))
    n = len(states_list) + 1 #plus one for terminal state

    #create reward vector
    reward_vec = np.array([ r for s,r in reward_func.items() ])
    reward_vec = np.append(reward_vec,0.0) #for terminal state
    assert( n == reward_vec.size)

    #create probability matrix
    prob = np.zeros((n,n))

    #loop through prob function
    for s, inner_dict in prob_func.items():
        check_prob = 0.0
        row_index = states_list.index(s)
        for sp, p in inner_dict.items():
            # print("sp: {}".format(sp))
            check_prob += p
            #get index
            if sp in states_list:
                col_index = states_list.index(sp)
            else:
                col_index = n - 1
            prob[row_index,col_index] = p
        assert( np.isclose(1,check_prob) )
    
    #now use inverse to get value function
    value = np.linalg.inv( np.identity(n) - prob ) @ reward_vec
    # print(value)

    #convert to dictionary again
    value_dict = { states_list[i]: value[i] for i in range(n-1) }
    value_dict["T"] = 0

    return value_dict


def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """

    #core is V(S) <- V(S) + alpha * ( R + gamma *V(s') - V(s) )
    #do this for 30,000 repetitions

    #set up value function dictionary
    state_value = dict()
    #add terminal state
    state_value["T"] = 0 #guaranteed by MDP

    #get number of samples
    num_obs = len(srs_samples)
    
    #calcualte alpha
    alpha = learning_rate * np.power( num_updates / learning_rate_decay + 1 , -0.5)
    
    #loop through all observations, multiple times for experience replay
    for i in range(num_updates):
        
        #get relevant tuple
        tup = srs_samples[ i % num_obs ]

        #pull out values
        s = tup[0]
        r = tup[1]
        sp = tup[2]

        #if observed, update
        if s in state_value:
            #add next state if needed
            if sp not in state_value:
                state_value[sp] = 0
            #update state value 
            state_value[s] += alpha * (r + state_value[sp] - state_value[s] ) 
        else: #add it
            state_value[s] = 0
    
    #return back value dictionary value
    return state_value     


def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    #need to work with matrices again. 
    #note sutton-barto book which states "x(terminal) = 0"

    #get states, set up date value
    state_value = {tup[0]:0 for tup in srs_samples }
    states_list = list(state_value)
    n = len(states_list)
    # print(n)

    #set up formula components
    M = np.zeros((n,n))
    v = np.zeros((n,1))

    #loop through all observations, to find the number of state s
    for tup in srs_samples:
        #extract info from sample
        s = tup[0]
        r = tup[1]
        sp = tup[2]

        ##outer product
        #first vector
        xs = np.zeros(n)
        xs[states_list.index(s)] = 1
        xs = xs.reshape((n,1))
        #second vector
        xsp = np.zeros(n)
        if sp in states_list:
            xsp[states_list.index(sp)] = 1
        xsp = xsp.reshape((n,1))
        # outer produce
        temp = np.matmul(xs , (xs.T-xsp.T)) 
        #add to M
        M += temp

        # print(s,sp)
        # print("xs: \n",xs)
        # print("xsp: \n",xsp)
        # print("temp: \n",temp)
        # print("temp shape",temp.shape)
        # print("M: \n",M)

        ## reward vector sum
        v += xs * r
        # print("v: \n",v)
    
    #now computer the feature vector, in tabular also the state value function
    value = np.linalg.inv( M ) @ v

    #convert to dictionary again
    value_dict = { states_list[i][0]: float(value[i]) for i in range(n) }
    #value_dict["T"] = 0
    
    return value_dict



if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MDP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))
