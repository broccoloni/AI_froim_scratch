import numpy as np

class activation():
    def __init__(self):
        self.last_input = None
        self.output = None
        self.gradient = None
        self.assert_msg = "Input must be passed through activation before backward is called"
        
    def __call__(self,x):
        self.last_input = x
        self.output = self.calculate(x)
        return self.output
    
    def backward(self, dout):
        assert self.output is not None, self.assert_msg
        
        self.gradient = self.derivative() * dout
        return self.gradient
    
    def calculate(self, x):
        return x
        
    def derivative(self):
        return 1

class sigmoid(activation):
    def __init__(self):
        super().__init__()
    
    def calculate(self,x):
        #Clip to make numericall stable
        return 1 / (1 + np.exp(-np.clip(x,-50,50)))

    def derivative(self):
        return self.output * (1 - self.outoput)
    
class tanh(activation):
    def __init__(self):
        super().__init__()
        
    def calculate(self, x):
        return np.tanh(x)
    
    def derivative(self):
        return 1 - np.square(self.output)

class ReLU(activation):
    def __init__(self):
        super().__init__()
        
    def calculate(self,x):
        return np.maximum(0,x)
        
    def derivative(self):
        d = np.zeros_like(self.last_input)
        d[self.last_input > 0] = 1
        return d















