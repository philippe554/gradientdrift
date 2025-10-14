

import jax
import numpy as np

from .batch import Batch

class Dataset:
    def __init__(self, data, columns = None):
        self.shape = {}
        loaded = False

        try:
            import pandas
            if isinstance(data, pandas.DataFrame):
                if columns is not None:
                    raise ValueError("Columns should not be provided for a pandas DataFrame.")
                df = data
                
                for col in df.columns:
                    colType = df[col].dtype
                    if not np.issubdtype(colType, np.number):
                        column = df[col].astype("category")
                        df[col] = column.cat.codes
                        self.shape[col] = column.cat.categories.tolist()
                    else:
                        self.shape[col] = (1,)

                data = df.values
                loaded = True

        except ImportError:
            pass

        if not loaded:
            if columns is None:
                raise ValueError("Columns must be provided for the dataset.")
            
            for col in columns:
                self.shape[col] = (1,)

        if isinstance(data, np.ndarray):
            data = jax.numpy.array(data)
        data = data.reshape(-1, len(self.shape))         

        if not isinstance(data, jax.numpy.ndarray):
            raise TypeError(f"Data must be a numpy array or a pandas DataFrame, found {type(data)}")

        self.data = data
        self.leftPadding = 0
        self.rightPadding = 0
        self.batches = []

    def getDataShape(self):
        return self.shape

    def getDataColumns(self):
        return self.shape.keys()

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
            paddedBatchSize = batchSize + self.leftPadding + self.rightPadding

            self.batches = []
            for i in range(start, end, batchSize):
                batchStart = i - self.leftPadding
                batchEnd = min(i + batchSize, end) + self.rightPadding
                batchData = self.data[batchStart:batchEnd]
                if batchData.shape[0] != paddedBatchSize:
                    continue  # Skip incomplete batches for now
                batch = Batch(batchData, self.leftPadding, self.rightPadding)
                self.batches.append(batch)

            print(f"Prepared {len(self.batches)} batches with size {batchSize} (padded to {paddedBatchSize})")
                
    def getNumberOfBatches(self):
        return len(self.batches)

    def getBatch(self, index):
        return self.batches[index]
