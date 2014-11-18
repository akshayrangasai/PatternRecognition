import os
import numpy as np
class dataset:
    def __init__(self, loc):
        self.directory = loc
    def loadData(self):
        pass
    def aggregate(self):
        pass
    def autoload(self):
        self.loadData()
        self.aggregate()
