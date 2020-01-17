from typing import Mapping, TypeVar, List
import numpy as np


#declare the generic State and Action types
S = TypeVar('S')
A = TypeVar('A')

#declare the transition matrix type
SSf = Mapping[S, Mapping[S, float]]

class MP():

    #member variables
    state_list: List[S] = None               #list of states
    P_map: SSf = None               #transition mapping
    P_matrix: np.ndarray = None     #transition matrix

    #constructor
    def __init__(
        self,
        trans_dict: SSf     #transition dictionary
    ) -> None :
        #set up internal variables
        self.P_map = trans_dict
        self.state_list = self.get_states(trans_dict)
        self.P_matrix = self.convert_to_matrix(trans_dict)
    
    def get_states(
        self,
        trans_dict: SSf     #transition dictionary
    ) -> List[S]:
        return list(trans_dict.keys())
    
    def convert_to_matrix(
        self,
        trans_dict: SSf     #transition dictionary
    ) -> np.ndarray:
        #create matrix 
        P = np.zeros( (len(self.state_list),len(self.state_list)) )
        #loop through states
        for s in self.state_list :
            for sp in list(trans_dict[s].keys()):
                P[s-1,sp-1] = trans_dict[s][sp]
        return P


if __name__ == '__main__':
    transitions = {
        1: {1: 0.1, 2: 0.6, 3: 0.1, 4: 0.2},
        2: {1: 0.25, 2: 0.22, 3: 0.24, 4: 0.29},
        3: {1: 0.7, 2: 0.3},
        4: {1: 0.3, 2: 0.5, 3: 0.2}
    }
  
    mp_obj = MP(transitions)
    print(mp_obj.P_map)
    print(mp_obj.state_list)
    print(mp_obj.P_matrix)

