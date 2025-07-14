


class Batch:
    def __init__(self, data, leftPadding, rightPadding):
        self.data = data
        self.leftPadding = leftPadding
        self.rightPadding = rightPadding

    def setData(self, data):
        self.data = data

    def getEffectiveNObs(self):
        return self.data.shape[0] - self.leftPadding - self.rightPadding