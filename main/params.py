class params(object):
    
    def __init__(self):
        self.LC = 1e-5
        self.eta = 0.001

    def __str__(self):
        t = "LC", self.LC, ", eta", self.eta
        t = map(str, t)
        return ' '.join(t)