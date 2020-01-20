from typing import Mapping, Set, Sequence, Tuple, TypeVar, List
from MPv2 import MP
import numpy as np

#declare the generic State and Action types
S = TypeVar('S')
A = TypeVar('A')

STSff = Mapping[S, Tuple[Mapping[S, float], float]]
SSTff = Mapping[S, Mapping[S, Tuple[float, float]]]
Sf =  Mapping[S, float]
#declare the transition matrix type
SSf = Mapping[S, Mapping[S, float]]



def SSTff_to_SSf_Sf(
    info: SSTff
) -> Tuple[ SSf, SSf ] :
    P = {s : {sp : prob for sp, (prob,_) in innerdict.items() } for s, innerdict in info.items()  }
    Rssp = {s : {sp : r for sp, (_,r) in innerdict.items() } for s, innerdict in info.items()  }
    return (P, Rssp )


def STSff_to_SSf(
    info: STSff
) -> SSf :
    return {k : P for k, (P, _) in info.items()}

def STSff_to_Sf(
    info: STSff
) -> Sf :
    return {k : r for k, (P, r) in info.items()}

class MRP(MP):

    #member variables
    Rss_map : SSf   #rewards SSf
    Rs : List[float] #reward vector
    Rss_matrix : np.ndarray #reward matrix
    #inheriteds
    # state_list: List[S] = None      #list of states
    # P_map: SSf = None               #transition mapping
    # P_matrix: np.ndarray = None     #transition matrix
    

    def __init__(
        self,
        state_P_R : STSff,
        gamma : float
    ) -> None:
        #initalize the underlying MP
        P_R = SSTff_to_SSf_Sf(state_P_R)
        super().__init__( P_R[0] )
        #initalize the reward vector
        self.Rss_map = P_R[1]
        self.Rss_matrix = self.convert_to_matrix(self.Rss_map)
        self.gamma = gamma
        self.Rs = self.get_Rs()
    
    def get_Rs(
        self
    ) -> List[float]:
        #matrix multiply
        M = np.multiply(self.P_matrix, self.Rss_matrix)
        #sum across s'
        Rs = np.sum(M, axis=1)
        return list(Rs)


if __name__ == '__main__':
    #SSTff
    data = {
        1: {1: (0.3, 9.2), 2: (0.6, 3.4), 3: (0.1, -0.3)},
        2: {1: (0.4, 0.0), 2: (0.2, 8.9), 3: (0.4, 3.5)},
        3: {3: (1.0, 0.0)}
    }
    # #STSff
    # data = {
    #     1: ({1: 0.6, 2: 0.3, 3: 0.1}, 7.0),
    #     2: ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0),
    #     3: ({3: 1.0}, 0.0)
    # }
    mrp_obj = MRP(data, 1.0)
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


