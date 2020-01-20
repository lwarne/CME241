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

def SASTff_to_SASf(
    info: SASTff     
) -> Tuple( SASf, SASf ):
    P = { s : {a : { sp, prob for sp, (prob,_) in inner2.items() } for a, inner2 in inner1.items()  } s, inner1 in info.items() }
    R = { s : {a : { sp, r for sp, (_,r) in inner2.items() } for a, inner2 in inner1.items()  } s, inner1 in info.items() }
    return (P,R)

def s2i(
    l : str
) -> int:
    return ord(l)-97

def SASf_to_3d(
    map: SASf
) -> np.ndarray:
    s = len(list(map.keys()))
    M = np.zeros(s,s,s)
    for s in list(map.keys()):
        for a in list(map[s].keys()):
            for sp in list(map[s][a].keys()):
                M(s-1, s2i, sp-1) = map[s][a][sp]




class MDP():

    #member variables 
    R_map : SASf  #reward map
    P_map : SASf    #transition map
    R_matrix: np.ndarray #reward matrix
    P_matrix: np.ndarray #transiton matrix 
    gamma: float
    state_list : List[S]
    action_set : Set[A]

    
    def __init__(
        self,
        data: SASTff,
        gamma: float
    ) -> None:
        #split and assign
        P_R = SASTff_to_SASf
        self.R_map = P_R[0]
        self.P_map = P_R[1]

        #extract states
        state_list = get_states(self.P_map)
        
        #convert to matrix (s,sp,a)
         


    def get_states(
        self,
        data: SASf
    ) -> List[S]:
        return list(data.keys())

    def get_actions(
        self
        data: SASf
    ) -> Set[A]:
        aset = set()
        for s in list(data.keys()):
            for a in list(data[s].keys())
                aset.add(a)


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
