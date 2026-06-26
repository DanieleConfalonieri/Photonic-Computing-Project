import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class Node(ABC):

    @property
    @abstractmethod
    def loss(self):
        return self._lossValue
    
    @loss.setter
    @abstractmethod
    def loss(self, lossTupla):
        pass

    

