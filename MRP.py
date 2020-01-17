from typing import Mapping, Set, Sequence, Tuple
from MPv2.py import MP
import numpy as np

STSff = Mapping[S, Tuple[Mapping[S, float], float]]
SSTff = Mapping[S, Mapping[S, Tuple[float, float]]]
Sf =  Mapping[S, float]]

def SSTff_to_SSf_Sf(
    info: SSTff
) -> Tuple[ Mapping[S, Mapping[S,f]], Mapping[S,f] ] :
    P = {s : {sp : prob for sp, (prob,_) in innerdict.items() } for s, innerdict in info.items()  }
    Rssp = {s : {sp : r for sp, (_,r) in innerdict.items() } for s, innerdict in info.items()  }
    return (P, Rssp  )


def STSff_to_SSf(
    info: STSff
) -> SSf :
    return {k : P for k, (P, _) for info.items()}

def STSff_to_Sf(
    info: STSff
) -> Sf :
    return {k : r for k, (P, r) for info.items()}

class MRP_Rs(MP):

    #member variables
    

    def __init__(
        self,
        state_P_R : STSff
    ) -> None:
        #initalize the underlying MP
        super().__init__( STSff_to_SSf(state_P_R) )
        #initalize the 

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
    print(mrp_obj.trans_matrix)
    print(mrp_obj.rewards_vec)
    terminal = mrp_obj.get_terminal_states()
    print(terminal)
    value_func_vec = mrp_obj.get_value_func_vec()
    print(value_func_vec)


