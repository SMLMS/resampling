# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import numpy as np

class Bootstrap:
    def __init__(self):
        self._data = []
        self._seed = 42
        
    @property
    def data(self):
        return np.copy(self._data)
    
    @data.setter
    def data(self, values: np.ndarray):
        self._data = np.copy(values)
        
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, value: int):
        self._seed = value
        
    def resample(self, number: int):
        np.random.seed(seed = self.seed)
        idx = np.arange(0,np.shape(self._data)[0]).tolist()
        idy = np.random.randint(low = 0,
                                high = np.shape(self._data)[1],
                                size = (np.shape(self._data)[0], number))
        rsData = np.zeros([np.shape(self._data)[0],number])
        for i in range(number):
            rsData[:,i] = self._data[idx, idy[:,i].tolist()]
        return rsData
        
        
    def __str__(self):
        string = str("\nBootstrap\nseed: %s\n" %(self.seed))
        return string
    
    def __del__(self):
        message  = str("removed instance of Bootstrap from heap.")
        print(message)

def main():
# =============================================================================
#     define test data set
# =============================================================================
    n = 5
    values  = np.zeros([5,9])
    for i in range(n):
        values[i,:] = np.arange(1*10**i, 10*10**i, 1*10**i)
    print(values)
# =============================================================================
#     resample test data set
# =============================================================================  
    bs = Bootstrap()
    bs.data = values
    bs.seed = 42
    print(bs.resample(number = 3))
# =============================================================================
#     oob = [x for x in data if x not in boot]
#     print('OOB Sample: %s' % oob)
# =============================================================================

if __name__ == '__main__':
    main()