#rltest.py
from MertonProblemMDP import Merton
from Sarsalam import RL_Merton
import numpy as np

t = 5
gamma = .01
mu = .1
rf = .03
sigma_sq = .05
states = 125

#initalize 
M = Merton(
    T = t, 
    rho = .95,
    rf = rf,
    mu = mu,
    sigma_sq = sigma_sq, #bad design!
    sigma = .1, #not used
    gamma = gamma,
    min_num_states = states,
    wealth_steps = 10,
    num_actions = 11,
    action_range = (0.0,1.0),
    W_0 = 50.0
)

RL = RL_Merton(
    M,
    M.A_set,
    M.S_cols,
    M.T,
    a = 0.2,
    g = 0.9,
    l = 0.9,
    e = 0.1
)

# #test interface
# t = 3
# W = M.S_cols[25]
# a = 0.0
# for i in range(10):
#     print(i)
#     print("    in state t:{} W:{}, action:{}".format(t,W,a))
#     tp,Wp,R = M.rl_interface(3,M.S_cols[25],0.0)
#     print("    t:{} W:{}".format(tp,Wp))

# #test interface, 0.0 action does not change wealth below wealth[11] = 31.778547053885116= M.S_cols[11]
# for i in range(states):
#     print(i)
#     w0 = M.S_cols[i]
#     print("    in state t:{} W:{}, action:{}".format(t,w0,a))
#     tp,Wp,R = M.rl_interface(3,M.S_cols[i],0.0)
#     print("    t:{} W:{}".format(tp,Wp))
#     # assert(not np.isclose(w0,Wp) )

# #test single episode
#RL.single_episode()

# #test many episodes
#RL.train(episodes = 100000)

# #test import
# RL.import_Q("Rlv1.csv")

# #test policy
# pol = RL.optimal_pol()
# print(pol)

#loop and run
for i in range(2):
    #import previous
    RL.import_Q("Rlv1.csv")
    #train additional 100K
    RL.train(episodes = 100000)
    #print policy
    pol = RL.optimal_pol()
    print(pol)

# #import and print policy
# RL.import_Q("Rlv1.csv")
# pol = RL.optimal_pol()
# print(pol)