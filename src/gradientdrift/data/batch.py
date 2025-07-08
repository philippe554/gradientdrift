


class Batch:
    def __init__(self, data):
        self.data = data
        self.leftPadding = 0
        self.rightPadding = 0

    def setData(self, data):
        self.data = data

    def getEffectiveNObs(self):
        return self.data.shape[0] - self.leftPadding - self.rightPadding