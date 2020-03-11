#MertonCheck.py
from MertonProblemMDP import Merton

#mu rf gamma sigma^2
mu = [.05, .1]
rf = [0.03,0.04]
gamma = [.01, .03]
sigma_sq = [.05,.1]

#static variables
t = 5
rho = .95
states = 125
actions = 6
W = 50

#test mu
#problem
# M = Merton(
#         T = t, 
#         rho = .95,
#         rf = rf[0],
#         mu = mu[0],
#         sigma_sq = sigma_sq[0], #bad design!
#         sigma = .1, #not used
#         gamma = gamma[0],
#         min_num_states = states,
#         wealth_steps = 10,
#         num_actions = actions,
#         action_range = (0.0,1.0),
#         W_0 = W
#     )

# pol0 = M.extract_optimal_policy()

# print(pol0)

M = Merton(
        T = t, 
        rho = .95,
        rf = rf[0],
        mu = mu[1],
        sigma_sq = sigma_sq[0], #bad design!
        sigma = .1, #not used
        gamma = gamma[0],
        min_num_states = states,
        wealth_steps = 10,
        num_actions = actions,
        action_range = (0.0,1.0),
        W_0 = W
    )

pol1 = M.extract_optimal_policy()

print(pol1)

# print("mu = {} policy = {}".format( mu[0], pol0[ int(t/2) , int(states/2) ] ))    
print("mu = {} policy = {}".format( mu[1], pol1[ int(t/2) , int(states/2) ] ))




