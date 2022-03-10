#%%
import numpy as np

def split_data_by_element_value(numpy_data):
   
   print(numpy_data[numpy_data[:,0] > 3])
   
data =np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [3, 2, 1]
    ]
)
split_data_by_element_value(data)
    
# %%
