#Sarsalam.py
from MertonProblemMDP import Merton
import numpy as np
from typing import Mapping, Set, Sequence, Tuple, TypeVar, List

#declare the generic State and Action types
S = TypeVar('S')
A = TypeVar('A')

class RL_Merton:

    ##member variables
    #merton underlying (only use rl_interface)
    M: Merton

    #state and action spaces
    A_set: np.ndarray
    W_set: np.ndarray
    T: int

    #learning arrays
    Q: np.ndarray   
    E: np.ndarray 

    #algo constants
    alpha: float #learning rate
    gamma: float #discount rate
    lam: float   #lambda for SARSA lambda
    epsilon: float #epsilon for e-greedy policy 
    
    def __init__(
        self,
        M: Merton,
        action_set: np.ndarray,
        wealth_set: np.ndarray,
        end_time: int,
        a: float,
        g: float,
        l: float,
        e: float
    ) -> None:
        """ set up and initalize the SARSA algo
        """
        self.M = M
        ##initalize Q(s,a) @ 0
        #set up the number of rows (index), include zero time
        row_count = (end_time+1) * wealth_set.size
        #set up number of columns
        col_count = action_set.size
        #initalize to zeros
        self.Q = np.zeros((row_count,col_count))

        #set up the eligibility trace
        self.E = np.zeros((row_count,col_count))

        #store the action set
        self.A_set = action_set 

        #store the weath_set 
        self.W_set = wealth_set

        #store time
        self.T = end_time

        #store the algo variables
        self.alpha = a
        self.gamma = g
        self.lam = l
        self.epsilon = e
    
    def vec(
        self,
        t: int,
        W: float,
    ) -> int:
        """ converts t, W into a index (row index) for the Q(s,a) 2d array
        """
        w_ind = np.argmin(np.abs(self.W_set - W))
        return (t+1)*w_ind
    
    def single_episode(
        self,
    ) -> None:
        """ Perform a single episode with backwards Sarsa(lambda) algo
        """
        #reset eligibility
        self.E = np.zeros((self.Q.shape[0],self.Q.shape[1]))

        #have a starting state
        t = 0
        W = np.random.choice(self.W_set) #random choosing of the state

        #choose a random action as first action
        a_ind = np.random.choice(np.arange(self.A_set.size))
        a = self.A_set[a_ind]

        #start the loop
        while (t != self.T):
            # take action A, and get R,S'
            print(t,W,a)
            tp,Wp,R = self.M.rl_interface(t,W,a)

            ##choose A' by epsilon greedy
            #retreive Q(s', . )
            action_vals = self.Q[self.vec(tp,Wp),:]
            
            #find the index of max action
            i_max = np.argmax(action_vals)
            
            #set up egreedy probability
            egreedy = np.full(self.A_set.size, self.epsilon/self.A_set.size)
            egreedy[i_max] += 1- self.epsilon
            assert( np.isclose(np.sum(egreedy),1) )

            #randomly get the action
            ap_ind = np.random.choice(np.arange(self.A_set.size), p = egreedy) #TODO, note that this only chooses the first if palces are equal
            ap = self.A_set[ap_ind]

            #calculate the error
            td_error = R + self.gamma * self.Q[self.vec(tp,Wp), ap_ind] - self.Q[ self.vec(t,W), a_ind]

            #update the eligability (plus 1)
            self.E[ self.vec(t,W), a_ind ] += 1

            #update all states
            self.Q = self.Q + self.E * self.alpha * td_error

            #update eligability (discout by lambda)
            self.E = self.gamma * self.lam * self.E
            
            #flip for next loop
            a = ap
            a_ind = ap_ind
            t = tp
            W = Wp

            print("action: {} t: {} W:{}".format(a,t,W))
    
    #next steps, optimal policy extraction,



            

        





