

import jax
import numpy as np

from .batch import Batch

class Dataset:
    def __init__(self, data, columns=None):
        if columns is None:
            raise ValueError("Columns must be provided for the dataset.")
        
        try:
            import pandas
            if isinstance(data, pandas.DataFrame):
                data = data.values
                if columns is None:
                    raise ValueError("Columns should not be provided for a pandas DataFrame.")
                columns = data.columns
        except ImportError:
            pass

        if isinstance(data, np.ndarray):
            data = jax.numpy.array(data)

        data = data.reshape(-1, len(columns))

        if not isinstance(data, jax.numpy.ndarray):
            raise TypeError("Data must be a numpy array or a pandas DataFrame.")

        self.data = data
        self.columns = columns               
        self.leftPadding = 0
        self.rightPadding = 0
        self.batches = []

    def setData(self, data):
        self.data = data

    def setLeftPadding(self, leftPadding):
        self.leftPadding = leftPadding

    def setRightPadding(self, rightPadding):
        self.rightPadding = rightPadding

    def getEffectiveNObs(self):
        return self.data.shape[0] - self.leftPadding - self.rightPadding
    
    def prepareBatches(self, batchSize):
        if batchSize == -1:
            self.batches = [Batch(self.data, self.leftPadding, self.rightPadding)]
        else:
            start = self.leftPadding
            end = self.data.shape[0] - self.rightPadding

            self.batches = []
            for i in range(start, end, batchSize):
                batchStart = i - self.leftPadding
                batchEnd = min(i + batchSize, end) + self.rightPadding
                batchData = self.data[batchStart:batchEnd]
                batch = Batch(batchData, self.leftPadding, self.rightPadding)
                self.batches.append(batch)
                
    def getNumberOfBatches(self):
        return len(self.batches)

    def getBatch(self, index):
        return self.batches[index]
