#Sarsalam.py
from MertonProblemMDP import Merton
import numpy as np
from typing import Mapping, Set, Sequence, Tuple, TypeVar, List

#declare the generic State and Action types
S = TypeVar('S')
A = TypeVar('A')

np.set_printoptions(linewidth = 1000, precision = 2, edgeitems =125)

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
        return (t*self.W_set.size) + w_ind
    
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
            # print("t = {}".format(t))
            # take action A, and get R,S'
            #print("Input of interface t{},W{},a{}".format(t,W,a))
            tp,Wp,R = self.M.rl_interface(t,W,a)
            # print("  output of interface tp,wp {}".format(self.M.rl_interface(t,W,a)))

            ##choose A' by epsilon greedy
            #retreive Q(s', . )
            action_vals = self.Q[self.vec(tp,Wp),:]

            # print("  action values: \n    {}".format(action_vals))
            
            #find the index of max action
            i_max = np.argmax(action_vals)
            
            #set up egreedy probability
            egreedy = np.full(self.A_set.size, self.epsilon/self.A_set.size)
            egreedy[i_max] += 1- self.epsilon
            assert( np.isclose(np.sum(egreedy),1) )
            # print("  action probs egreedy: \n    {}".format(egreedy))

            #randomly get the action
            ap_ind = np.random.choice(np.arange(self.A_set.size), p = egreedy) #TODO, note that this only chooses the first if palces are equal
            ap = self.A_set[ap_ind]
            # print("  action index choice, action: \n    {},{}".format(ap_ind, ap))

            #calculate the error
            td_error = R + self.gamma * self.Q[self.vec(tp,Wp), ap_ind] - self.Q[ self.vec(t,W), a_ind]
            
            # print("  td_error {}".format(td_error))
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

            #print("action: {} t: {} W:{}".format(a,t,W))
            
            #the value of the terminal states in SARSA(lambda) is always 0
        
        #print changes to state action value after single episode
        # print("Q dim {}".format(self.Q.shape))
        # print("Q(S,A) \n {}".format(self.Q))
        # print("E(S,A) \n {}".format(self.E))

    def train(
        self,
        episodes :int = 1000
    ) -> None:
        """ run episodes to train the model """
        cum_diff = 0
        for i in range(episodes):
            Q_prev = self.Q
            self.single_episode()
            max_diff = np.max(np.abs(self.Q - Q_prev))
            cum_diff += max_diff
            if (i%1000 == 0):
                print("iter: {}, cum_diff {}".format(i,cum_diff))
                cum_diff = 0
            if max_diff <= 1e-3:
                break
        
        #print("ending Q(S,A) value function \n {}".format(self.Q[0:(750-125),:]))
        np.savetxt('Rlv1.csv', self.Q, delimiter=',',fmt="%.8f")


    def optimal_pol(
        self
    ) -> np.ndarray:
        """ extrack the optimal policy from the learned Q(S,A) state action value function"""
        #first choose the optimal action for each row
        action_index = np.argmax(self.Q, axis = 1)
        #next get the optimal action
        opt_action_vec = self.A_set[action_index]
        #reshape into t x W matrix format
        return opt_action_vec.reshape((self.T+1,self.W_set.size))

    
    def import_Q(
        self,
        file_path: str
    ) -> None:
        """ set selfQ to an imported file """
        #load
        I = np.loadtxt(file_path,delimiter=",",dtype="float")
        #check size
        assert( (self.Q.shape[0] == I.shape[0]) and (self.Q.shape[1] == I.shape[1])   )
        #assign
        self.Q = I
    
    def printQ(
        self
    ) -> None:
        """print Q with internal settings """
        print(self.Q)

    

            

        





