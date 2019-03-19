import numpy as np

def encode_actions(min_arr,max_arr,step_arr):
    arr = []
    for i in range(len(min_arr)):
        arr.append(np.arange(min_arr[i],max_arr[i]+step_arr[i],step_arr[i]))
    res = np.array(np.meshgrid(*arr)).T.reshape(-1,len(min_arr))
    return res

