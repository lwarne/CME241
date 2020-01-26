from typing import Mapping, Set, Sequence, Tuple, TypeVar, List
from MPv2 import MP
from MRP import MRP
import numpy as np

#declare the generic State and Action types
S = TypeVar('S')
A = TypeVar('A')

SATSff = Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]]
SASTff = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]
SASf = Mapping[S, Mapping[A, Mapping[S, float]]]
SAf = Mapping[S, Mapping[A,float]]

def SASTff_to_SASf(
    info: SASTff     
) -> Tuple[ SASf, SASf ]:
    P = { s : {a : { sp : prob for sp, (prob,_) in inner2.items() }  \
            for a, inner2 in inner1.items()  } for s, inner1 in info.items() }
    R = { s : {a : { sp : r for sp, (_,r) in inner2.items() } \
            for a, inner2 in inner1.items()  } for s, inner1 in info.items() }
    return (P,R)

def s2i(
    l : str
) -> int:
    """ ord('a') = 97 """
    return ord(l)-97 

class MDP():

    #member variables 
    R_map : SASf  #reward map
    P_map : SASf    #transition map
    R_matrix: np.ndarray #reward matrix
    P_matrix: np.ndarray #transiton matrix 
    gamma: float
    state_list : List[S]    #list of states
    action_set : Set[A]     #list of actions
    
    def __init__(
        self,
        data: SASTff,
        gamma: float
    ) -> None:
        #split and assign
        P_R = SASTff_to_SASf(data)
        self.R_map = P_R[1]
        self.P_map = P_R[0]

        #extract states
        self.state_list = self.get_states(self.P_map)
        #extract actions
        self.action_set = self.get_actions(self.P_map)
        #convert to matrix (s,sp,a)
        self.R_matrix = self.SASf_to_3d( self.R_map )
        self.P_matrix = self.SASf_to_3d( self.P_map )
        #save gamma
        self.gamma = gamma
            
    def SASf_to_3d(
        self,
        map: SASf
    ) -> np.ndarray:
        snum = len(self.state_list)
        anum = len(self.action_set)
        print(snum, anum)
        M = np.zeros((snum,snum,anum))
        for s in list(map.keys()):
            for a in list(map[s].keys()):
                for sp in list(map[s][a].keys()):
                    M[s-1, sp-1, s2i(a)] = map[s][a][sp]
        return M

    def get_states(
        self,
        data: SASf
    ) -> List[S]:
        return list(data.keys())

    def get_actions(
        self,
        data: SASf
    ) -> Set[A]:
        aset = set()
        for s in list(data.keys()):
            for a in list(data[s].keys()):
                aset.add(a)
        return aset

    def generate_MRP(
        self,
        policy: SAf #deterministic policy 
    ) -> MRP:
        """ Convert R(s,s') MDP + policy to an MRP """
        #create transition probabilities
        #\sum_a pi(a|s) * P^a_{s,s'}
        data = dict()

        #given a state
        for s in self.state_list: 
            
            #create interal dictionary
            data[s] = { sp : [0,0] for sp in self.state_list }
            #print("data: {}".format(data))

            #sum over actions
            for a in self.action_set: 

                #check action exists in transition dict
                if a not in self.P_map[s]:
                    continue #skip action if not 

                #loop through innermost dict k = sp, v =prob
                for k,prob in self.P_map[s][a].items(): 

                    #check action exists in policy
                    if a not in policy[s]: 
                        continue #skip if action missing 
                    #print("s: {} a: {}, sp:{}, prob:{} ".format(s,a,k,prob))
                    
                    #add to incremental transition probabilties
                    #print("data[s][k]: {}".format(data[s][k]))
                    data[s][k][0] += policy[s][a] * prob
                    
                    #add to incremental reward
                    data[s][k][1] += policy[s][a] * self.R_map[s][a][k] * prob
            
            #divide by transition prob, or delete entry
            #track entries that are zero, or divide by transition prob
            delete_list = list()
            for sp, v in data[s].items():
                if data[s][sp][0] == 0:
                    #print("s: {},sp: {}, v: {}".format(s,sp,v))
                    delete_list.append( (s,sp) )
                else:
                    data[s][sp][1] /= data[s][sp][0]
            #delete zero entries
            for t in delete_list:
                del data[t[0]][t[1]]
        
        #transform data to have inner tuples
        mrp_SSTff = { s: { sp : (l[0],l[1]) for sp,l in lvl1.items() } \
            for s,lvl1 in data.items() }
        #feed into MRP constructor
        mrp = MRP(mrp_SSTff, self.gamma)

        return mrp

if __name__ == '__main__':

    print("This is MDPRefined")
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
    mdp = MDP(mdp_refined_data,.9)
    print( "R map {}".format(mdp.R_map) )
    print( "R matrix {}".format(mdp.R_matrix) )
    print( "P map {}".format(mdp.P_map) )
    print( "P matrix {}".format(mdp.P_matrix) )
    print( "P matrix action a {}".format(mdp.P_matrix[:,:,0]) )
    print( "state_list {}".format(mdp.state_list) )
    print( "action set {}".format(mdp.action_set) )

    print("----------------")
    print("This is the Policy")
    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    print("----------------")
    print("This is the MRP")
    mrp_obj = mdp.generate_MRP(policy_data)
    print("trans matrix")
    print(mrp_obj.P_matrix)
    print("trans map")
    print(mrp_obj.P_map)
    print("state list")
    print(mrp_obj.state_list)
    print("Rss matrix")
    print(mrp_obj.Rss_matrix)
    print("Rs")
    print(mrp_obj.Rs)

    """{1: {'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)}, 'b': {2: (0.3, -0.5), 3: (0.7, 2.6)}, 'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}}}
    """

