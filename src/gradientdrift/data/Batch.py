


class Batch:
    def __init__(self, data):
        self.data = data
        self.leftPadding = 0
        self.rightPadding = 0

    def setData(self, data):
        self.data = data