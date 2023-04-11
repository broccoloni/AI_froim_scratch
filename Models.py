import numpy as np
from Layers import *

class Model():
    def __init__(self):
        self.loss_fn = None
        self.layers = []
        self.output = []
        self.loss = None
        self.numlayers = 0
        
        assert_loss_fn = "A loss function must be added to the model before calling backward"
        assert_output = "An input must be passed through forward before the loss can be calculated"
        assert_layer = "A layer must be added to the model before calling forward"
        assert_loss = "The loss must be calculated before backward can be called"
        assert_recurrent = "Recurrent layer must have second input"
        
    def __call__(self, x, state = None):
        assert self.layers, assert_layer
        for i,layer in enumerate(self.layers):
            #Linear layers
            if layer.layer_type == 'l':
                x = layer(x)
            elif layer.layer_type == 'r':
                assert state, assert_recurrent
                x, state = layer(x, state)
        self.output = x
        return self.output
        
    def calculate_loss(self, ytrue):
        assert self.output is not None
        assert self.loss_fn is not None
        self.loss = self.loss_fn(self.output, ytrue)
        return self.loss
        
    def backward(self, dout = None):
        if dout is None:
            assert self.loss_fn is not None, assert_loss_fn
            assert self.loss is not None, assert_loss
            dout = self.loss_fn.gradient
            
        for i,layer in enumerate(reversed(self.layers)):
            dout = layer.backward(dout)
        
    def addLayer(self, layer, activation = None):
        self.layers.append(layer)
        self.numlayers += 1
        
        if activation is not None:
            self.layers[-1].set_activation(activation)        
            
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
        
    def torch_weighted(self, torchmodel):
        # NOTE: Requires torch model layers to be defined 
        # in the order that they are applied
        
        # Get weights from torch model 
        all_weights = [x.detach().numpy() for x in torchmodel.parameters()]
        
        # Put weights into this model
        cur_weight = 0
        for layer in self.layers:
            if isinstance(layer, Model):
                for l in layer.layers:
                    l.weights = all_weights[cur_weight:cur_weight + l.numweights]
                    cur_weight += l.numweights
            if isinstance(layer, Layer):
                layer.weights = all_weights[cur_weight:cur_weight + layer.numweights]
                cur_weight += layer.numweights


class LSTM(Model):
    def __init__(self, input_size, hidden_size, num_layers, batch_first = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.layer_type = 'r'
        
        self.layers = [LSTMCell(input_size, 
                                 hidden_size)]
        
        for i in range(1,num_layers):
            self.layers.append(LSTMCell(hidden_size, 
                                         hidden_size))
        
    def __call__(self, x, hc):
        h, c = hc
        
        has_batch = len(x.shape)==3
        
        if self.batch_first and has_batch:
            for t in range(x.shape[1]):
                for i,layer in enumerate(self.layers):
                    if i == 0:
                            h[:,i,:], c[:,i,:] = layer(x[:,t,:], (h[:,i,:],c[:,i,:]))
                    else:
                            h[:,i,:], c[:,i,:] = layer(h[:,i-1,:], (h[:,i,:],c[:,i,:]))
                self.output.append(h[:,-1,:].copy())
        else:
            for t in range(x.shape[0]):
                for i,layer in enumerate(self.layers):
                    if i == 0:
                        h[i], c[i] = layer(x[t], (h[i],c[i])) 
                    else:
                        h[i], c[i] = layer(h[i-1], (h[i], c[i]))
                self.output.append(h[-1].copy())
        self.output = np.stack(self.output,axis = 0)
        return self.output, (h,c)
    
    def backward(self, dh):
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
            
class GRU(Model):
    def __init__(self, input_size, hidden_size, num_layers, batch_first = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.layer_type = 'r'
        
        self.layers = [GRUCell(input_size, 
                                 hidden_size)]
        
        for i in range(1,num_layers):
            self.layers.append(GRUCell(hidden_size, 
                                         hidden_size))
                               
    def __call__(self, x, h):
        output = []
        
        has_batch = (len(x.shape)==3)
        
        if self.batch_first and has_batch:
            for t in range(x.shape[1]):
                for i,layer in enumerate(self.layers):
                    if i == 0:
                            h[:,i,:] = layer(x[:,t,:], h[:,i,:])
                    else:
                            h[:,i,:] = layer(h[:,i-1,:], h[:,i,:])
                output.append(h[:,-1,:].copy())
        else:
            for t in range(x.shape[0]):
                for i,layer in enumerate(self.layers):
                    if i == 0:
                        h[i] = layer(x[t], h[i]) 
                    else:
                        h[i] = layer(h[i-1], h[i])
                output.append(h[-1].copy())

        output = np.stack(output, axis = 0)
        return output, h
    
    def backward(self, dh):
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        
class RNN(Model):
    def __init__(self, input_size, hidden_size, num_layers, batch_first = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.layer_type = 'r'
        
        self.layers = [RNNCell(input_size, 
                               hidden_size)]
        
        for i in range(1,num_layers):
            self.layers.append(RNNCell(hidden_size, 
                                       hidden_size))
                               
    def __call__(self, x, h):
        output = []
        
        has_batch = len(x.shape)==3
        
        if self.batch_first and has_batch:
            for t in range(x.shape[1]):
                for i,layer in enumerate(self.layers):
                    if i == 0:
                            h[:,i,:] = layer(x[:,t,:], h[:,i,:])
                    else:
                            h[:,i,:] = layer(h[:,i-1,:], h[:,i,:])
                output.append(h[:,-1,:].copy())
        else:
            for t in range(x.shape[0]):
                for i,layer in enumerate(self.layers):
                    if i == 0:
                        h[i] = layer(x[t], h[i]) 
                    else:
                        h[i] = layer(h[i-1], h[i])
                output.append(h[-1].copy())

        output = np.stack(output, axis = 0)
        return output, h
    
    def backward(self, dh):
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        
        
        