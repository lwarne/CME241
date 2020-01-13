from typing import List, Callable
import numpy as np

#Start with a simple markov process class

class MP():

    #member variables
    S: int              #number of states
    P: np.ndarray       #transition matrix
-
    #constructor
    def __init__(
        self,
        state_count: int,         #number of states is finite, natural number
        trans_matrix: np.ndarray  #transition matrix is state_count x state_count
    ) -> None :
        #set up internal variables
        self.S = state_count    #store the number of states
        self.P = trans_matrix   #keep the transition matrix given
        
        #check that the matrix is square
        assert( self.P.shape[0] = self.S)
        assert( self.P.shape[1] = self.S)


class MRP(MP):

    #member variables
    MP: Generic         #markov process [S,P]
    R: np.ndarray       #reward matrix this is in the r(s,s') formulation
    gamma: float        #discount factor

    #constructor
    def __init__(
        self,
        state_count: int,               #number of states is finite, natural number
        trans_matrix: np.ndarray,       #transition matrix is state_count x state_count
        reward_function: np.ndarray,    #matrix of r(s,s')
        gamma: float                    #discount factor
    ) -> None:
        #set up the internal markov process
        MP.__init__(self, state_count, trans_matrix)
        #store reward and gamma
        self.R = reward_function
        self.gamma = gamma
        #check the size of the reward matches number of states
        assert(self.R.shape[0] == self.MP.S)
        if self.R.ndim == 2:
            assert(self.R.shape[1] == self.MP.S)
        #check gamma in bounds
        assert(gamma >=0 and gamma <=1)
    
    #convert r(s,s') to R(s)
    def get_Rs_state_rewards(
        self
    ) -> np.ndarray:
        """ impliments \sum_{s \in S} P_{ss'} R(s,s')"""
        #multuply P R, element wise
        #row sum result
        Rs = np.sum( np.multiply(self.MP.P, R) , axis = 1)
        #check dimensions
        assert(Rs.shape[0] == self.MP.S)
        return Rs

class MDP(MRP):

    #member variables
    MRP: Generic        #markov reward process [S,P,R,gamma]
    A: int              #number of possible actions 

    #constructor
    def __init__(
        self,
        state_count: int,               #number of states is finite, natural number
        trans_matrix: np.ndarray,       #transition matrix is state_count x state_count x action
        reward_function: np.ndarray,    #matrix of r(s,s')
        gamma: float,                    #discount factor
        a: int,                         #number of actions
    ) -> None
        
        
    
    


