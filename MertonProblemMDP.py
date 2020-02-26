import numpy as np
from typing import Mapping, Set, Sequence, Tuple, TypeVar, List
from scipy.stats import norm
from itertools import product

##Goal is to build an MDP that allows for dynamic programming

#declare the generic State and Action types
S = TypeVar('S')
A = TypeVar('A')

class Merton:

    #member variables:
    expiry: float  # = T
    rho: float  # = discount rate
    rf: float  # = risk-free rate
    mu: float  # = risky rate means
    sigma_sq: float  # = risky rate variacne
    sigma: float     #
    #epsilon: float  # = bequest parameter
    gamma: float  # = CRRA parameter
    t_step: float

    #state and action info
    action_range: Tuple[float, float] # [0,1] limit shorts
    num_states: int #number of states
    num_actions: int   #number of actions
    A_set: np.ndarray   #action vector [action is the % in risky asset]

    #store states in a matrix T x num_wealth states
    gap: float #wealth state gap
    max_wealth: int
    min_wealth: int
    num_wealth_states: int #number of wealth states

    S_value: np.ndarray #state space matrix
    S_rows: np.ndarray #row indices (time)
    S_cols: np.ndarray #col indices (wealth)

    #precomputed transition probabilties (no time) to save on compute time
    trans_probs: np.ndarray


    #set up a probability distribution


    def __init__(
        self,
        T: float,
        rho: float,                         # = discount rate
        rf: float,                           # = risk-free rate
        mu: float,                          # = risky rate means 
        sigma_sq: float,                    # = risky rate var
        sigma:float,                        # = risky sd
        gamma: float,                       # = CRRA parameter
        min_num_states: int,                # = discreteness of state space
        wealth_steps: float,                # = space between wealth states
        num_actions: int,                   # = discretenes of action space
        action_range: Tuple[float, float],  # = range of actions
        W_0: float                          # = starting wealth
    ) -> None:
        """ Constructor
            1) initalize the variables
            2) set up the granularity of the state and action space
        """
        #set up the member variables
        self.T = T
        self.rho = rho
        self.rf = rf
        self.mu = mu
        self.sigma_sq = sigma_sq
        self.sigma = np.sqrt(sigma_sq)
        self.gamma = gamma
        self.action_range = action_range
        #self.num_states = num_states
        self.num_actions = action_range
      
        #discretize the action space
        self.A_set = np.linspace(
                            action_range[0],
                            action_range[1],
                            num_actions
                        )
        
        #####    discretize the state space    #######
        #TODO CHANGE TO LOWER BOUND THE WEALTH ALSO
        #decide how to discretize wealth
        #cap the max wealth, function of variance and time
        #varT = self.sigma * self.T
        #sd_end = np.sqrt(varT)
        ##TRY 1
        sd_end = np.sqrt(sigma_sq) * np.sqrt(self.T)
        # print("sd_end {}".format(sd_end))
        # max_wealth =  W_0 + int(4*sd_end * W_0) #KEY VARIABLE, RANGE OF WEALTH STATES
        # min_wealth = max(0,W_0 - (max_wealth - W_0))
        ##TRY2
        max_wealth = W_0 * ((1+mu)**T) + 2*sd_end
        min_wealth = max(0,W_0 - (max_wealth - W_0))
        W_states_count = int( np.max( np.array([min_num_states, max_wealth / wealth_steps ]) ) )

        #set up state matrix
        self.S_rows = np.arange(0,T+1)
        self.S_cols = np.linspace(min_wealth,max_wealth,W_states_count)
        print("min wealth {} max wealth {}".format(min_wealth,max_wealth))
        print("S_cols 1 {}".format(self.S_cols))
        self.S_value = np.zeros((self.S_rows.size,  self.S_cols.size ))
        self.gap = self.S_cols[1] - self.S_cols[0]
        self.max_wealth = int(self.S_cols[-1:])
        self.min_wealth = int(self.S_cols[0])
        self.num_wealth_states = int(self.S_cols.size)

        #print( [(W,a) for W,a in product(self.S_cols , self.A_set) ])
        #precompute the transition probabilties
        self.trans_probs = np.array( [
            np.array([
            self.transition_prob(0,W,a)
            for W in self.S_cols
            ])
            for a in self.A_set 
        ])

        # np.set_printoptions(precision=1,suppress=True)
        # print(self.trans_probs[0,0:10,0:10])
        # print(self.trans_probs[0,1,:])
        # print(self.trans_probs[0,1,:].shape)
        #print(self.trans_probs)
        #print(self.trans_probs.shape)
        #assert(self.trans_probs.shape == (self.num_actions, self.num_wealth_states))

        print("time steps: {} Max wealth: {} wealth steps: {}".format(self.T, max_wealth, W_states_count))

    def pre_computed_trans(
        self,
        W : float,
        a : A,
    ) -> np.ndarray:
        """ easy access to the precomputed transiton probabilities """
        #identify the index of action and wealth
        a_ind = np.argmin(np.abs(self.A_set - a))
        w_ind = np.argmin(np.abs(self.S_cols - W))
        return self.trans_probs[a_ind,w_ind,:]

    def transition_prob(
        self,
        t : float,
        W : float,
        a : A,
        flag: str = "row"
    ) -> np.ndarray :
        """ calcualte the probability of transition to any next state given current state and acton (allocation)
            retruns vector of size self.W_states_count if flag = "row" (does not account for t)
            return matrix of size self.T + 1 x self.W_states_count if flag = "matrix"
        """

        # No transitions at t = T, absorbing states
        if t == self.T:
            wealth_probs = np.zeros(self.num_wealth_states)
            #print(np.where(self.S_cols == W))
            wealth_probs[ np.where(self.S_cols == W)] = 1.0
            assert(  np.isclose(np.sum(wealth_probs) ,1) )

            if flag == "row":
                return wealth_probs
            elif flag == "matrix":
                prob = np.zeros( (self.S_rows.size,  self.S_cols.size) )
                prob[t,:] = wealth_probs
                return prob
            else:
                print("flag not recognized in transition probs, returned row")
                return wealth_probs

        #if nothing in risky, then transition is deterministic
        if a == 0:
            wealth_probs = np.zeros(self.num_wealth_states)
            W1 = (1+self.rf) * W
            #diff = np.abs(self.S_cols - W1)
            #index = np.argmin(diff)
            wealth_probs[ np.argmin(np.abs(self.S_cols - W1)) ] = 1.0
            return wealth_probs

        #allocation to risky asset
        #probability wealth will move to the next state
        wealth_probs = np.array([
            norm.cdf(( nextW + .5*self.gap - (1+self.rf)*(1-a)*W - a*W ) /(a*W), self.mu, self.sigma  ) \
                - norm.cdf(( nextW - .5*self.gap - (1+self.rf)*(1-a)*W - a*W ) /(a*W), self.mu, self.sigma  )
            for nextW in np.linspace(self.min_wealth,self.max_wealth,self.num_wealth_states) 
        ])
        #adjust first and last.
        wealth_probs[0] += norm.cdf(( self.min_wealth - .5*self.gap - (1+self.rf)*(1-a)*W - a*W ) /(a*W), self.mu, self.sigma  ) 
        wealth_probs[-1:] += 1 -  np.sum(wealth_probs)
        # print("wealth probs: {}".format(wealth_probs))
        # print("sum wealth probs: {}".format(np.sum(wealth_probs)))

        assert(  np.isclose(np.sum(wealth_probs) ,1) )

        #return details 
        if flag == "row":
            return wealth_probs
        elif flag == "matrix":
            prob = np.zeros( (self.S_rows.size,  self.S_cols.size) )
            prob[t,:] = wealth_probs
            return prob
        else:
            print("flag not recognized in transition probs, returned row")
            return wealth_probs
            
    def reward_at_t(
        self, 
        t : float
    ) -> np.ndarray:
        """ only rewards at the last time step
            return array of size self.num_wealth_states
        """
        #apply utility
        if t != self.T :
            return np.zeros(self.num_wealth_states)
        else:
            # print("cols {}",format(self.S_cols))
            # print("exp(-gamma*w) {}".format(np.exp(- self.gamma * self.S_cols)))
            # print("1 - exp(-gamma*w) {}".format(1 -np.exp(- self.gamma * self.S_cols)))
            return (1 - np.exp(- self.gamma * self.S_cols)) / self.gamma  


    #next, value iteraton:
    #can I work with the matrix? Can i feed it into the value iterator

    def value_iteration(
        self,
        verbose : bool = True
    ) -> np.ndarray:
        """ use value iteration to obtain the optimal policy
        """
        #start with a guess for the value function
        vf_old = np.zeros( (self.S_rows.size,  self.S_cols.size) )
        vf_new = np.zeros( (self.S_rows.size,  self.S_cols.size) )

        #set up control variables
        tol = 1e-3
        max_iter = 100
        max_diff = 1000
        count = 0

        #loop until convergence or max iter
        while( max_diff > tol and count < max_iter ):

            #save the previous values
            vf_old = vf_new

            #update values: 
            #think in unrolled vector terms 
            vf_new = np.array([
                #inner loop
                #choose max across actions
                np.max(
                    np.array([
                        #assume reward, prob are vectors
                        np.dot( 
                            #self.transition_prob( self.S_rows[r], self.S_cols[c], a ), #transition probability W only
                            self.pre_computed_trans(self.S_cols[c],a ),
                            self.reward_at_t(r) + self.rho * vf_old[self.S_rows[min(self.T,r+1)],:] #get the value of the next time step 
                          )
                        #for each actions
                        for a in self.A_set
                    ])
                )
                #for each state
                for r,c in product(np.arange(self.S_rows.size),np.arange(self.S_cols.size) )
            ])
            
            #print(vf_new)
            #reshape back to matrix 
            vf_new = vf_new.reshape(self.S_rows.size,self.S_cols.size)#,order='F')

            #find max absolute difference
            max_diff = np.max( np.abs( vf_new - vf_old ))

            #increment count
            count += 1

            if verbose == True:
                print("iter {} \n    max diff: {} \n    ".format(count, max_diff))
        
        #return
        return vf_new


if __name__ == '__main__':
    t = 10
    #initalize 
    M = Merton(
        T = t, 
        rho = .5,
        rf = .005,
        mu = .02,
        sigma_sq = .005, #bad design!
        sigma = .1, #not used
        gamma = .01,
        min_num_states = 50,
        wealth_steps = 10,
        num_actions = 6,
        action_range = (0.0,1.0),
        W_0 = 100.0
    )

    print("max wealth: {} \n steps: {} \n gap: {} ".format( 
            M.max_wealth, M.num_wealth_states, M.gap ))
    print("action set {}".format(M.A_set))
    print( "initalized correctly")
    
    #run transition probs
    prob = M.transition_prob(1, M.S_cols[int(M.num_wealth_states/2)],1)
    prob = M.transition_prob(t, M.S_cols[int(M.num_wealth_states/2)] ,.4)
    #print(prob)
    #test rewards
    #reward = M.reward_at_t(1)
    #print(reward)
    reward = M.reward_at_t(t)
    print("reward:",reward)

    #reward check
    print("S_cols: \n  {}".format(M.S_cols))
    print( (1 - np.exp(- .01 * M.S_cols)) / .01 )

    np.set_printoptions(precision=1,suppress=True)
    #test value iteration
    val = M.value_iteration()
    
    #print(val.shape)
    print(val)

    
    ### Next steps
    # 1) get my shit together and figure out the log normal distribution to bound the wealth
    # 2) extract optimal policy
    # 3) vary paramaters and observe direction changes in optimal policy align with analytical solution
    # 4) START THE FUCK RL ALGO 







        


