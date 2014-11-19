import os
import numpy as np
class dataset:
    def __init__(self, loc):
        self.directory = loc
        self.datacontainer = []
    def loadData(self):
        for root, dirs, files in os.walk(self.directory):
            for f in files:
                datapoint = np.genfromtxt(os.path.join(root,f))
                self.datacontainer.append(datapoint)

    def aggregate(self):
        pass
    def autoload(self):
        self.loadData()
        self.aggregate()
