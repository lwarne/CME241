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

def SASTff_to_3d(
    self,
       
)

class MDP():

    #member variables 
    R_map : SASf  #reward map
    P_map : SASf    #transition map
    R_matrix: np.ndarray #reward matrix
    P_matrix: np.ndarray #transiton matrix 
    gamma: float

    
    def __init__(
        self,
        data: SASTff,
        gamma: float
    ) -> None:
        



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
