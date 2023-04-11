import numpy as np
import Activations as a

############################## Layer Outline ##############################

class Layer():
    def __init__(self):
        self.activation = None
        
        self.learning_rate = 0.001
    
        self.last_input = None
        self.output = None
        
        self.weights = None
        self.gradients = None
        self.numweights = 0
        self.layer_type = 'l' #linear, or 'r' - recurrent
        
    def __call__(self,x):
        self.last_input = x
        self.output = self.forward(x)
        if self.activation is not None:
            self.output = self.activation(self.output)
        return self.output
    
    def backward(self, dout):
        if self.activation is not None:
            dout = self.activation.backward(dout)
            
        dinput = self.calculate_gradients(dout)
        self.update_weights()
        return dinput
            
    def forward(self, x):
        return x
        
    def calculate_gradients(self, dout):
        self.gradients = dout
        return self.gradients

    def set_activation(self, activation):
        self.activation = activation
        
    def clip_gradients(self):
        gradnorms = [np.linalg.norm(grad) for grad in self.gradients]
        
        self.gradients = [self.gradients[i]/gradnorms[i] if gradnorms[i] > max_norm 
                          else self.gradients[i] for i in range(len(self.gradients))]
        
    def update_weights(self):
        if self.weights is not None:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * self.gradients[i]
        
    def xavier_init(self, shape):
        fan_in, fan_out = np.prod(shape[1:]), np.prod(shape[2:])
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, std, size=shape)
        
    def sigmoid(self, x):
        #Clip to make numericall stable
        return 1 / (1 + np.exp(-np.clip(x,-50,50)))
    
############################## Typical Layers ##############################
    
class Linear(Layer):
    def __init__(self, insize, outsize):
        super().__init__()
        
        self.insize = insize
        self.outsize = outsize
        
        weights = self.xavier_init((outsize, insize))
        bias = np.zeros((outsize))
        
        self.weights = [weights, bias]
        self.numweights = len(self.weights)
        
    def forward(self, x):
        weights, bias = self.weights
        return np.dot(x, weights.T) + bias
    
    def calculate_gradients(self, dout):        
        grad_weight = np.dot(dout.T, self.last_input)
        grad_bias = np.sum(dout, axis = 0)
        self.gradients = [grad_weight, grad_bias]
        
        dx = np.dot(dout, self.weights[0])
        return dx   
    
class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 2
        
        self.stride = stride
        if isinstance(stride, int):
            self.stride = [stride] * 2
            
        self.padding = padding
        if isinstance(padding,int):
            self.padding = [padding]*2
        
        weight = self.xavier_init((out_channels, 
                                   in_channels, 
                                   self.kernel_size[0], 
                                   self.kernel_size[1]))
        bias = np.zeros((out_channels))
        self.weights = [weight, bias]
        self.numweights = len(self.weights)
        
    def forward(self, x):
        weights, bias = self.weights
        N, C, H, W = x.shape
                
        # Calculate the output shape
        out_h = int((H + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        out_w = int((W + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)
        
        # Add padding to the input if necessary
        x_pad = np.pad(x, ((0,0),(0,0),
                           (self.padding[0], self.padding[0]),
                           (self.padding[1], self.padding[1])), 
                           'constant', constant_values = 0)
        output = np.zeros((N, self.out_channels, out_h, out_w))
           
        for i in range(out_h):
            for j in range(out_w):
                for k in range(self.out_channels):
                    wstart = i*self.stride[0]
                    wend = i*self.stride[0]+self.kernel_size[0]
                    hstart = j*self.stride[1]
                    hend = j*self.stride[1]+self.kernel_size[1]
                    window = x_pad[:, :, wstart:wend, hstart:hend]
                    output[:, k, i, j] = np.sum(window * weights[k],axis = (1,2,3))
        return output + bias.reshape(1,-1,1,1)
            
    def calculate_gradients(self, dout):
        weights, bias = self.weights
        N, _, out_h, out_w = dout.shape
        
        # Compute the gradient of the loss with respect to the weights and biases
        grad_weight = np.zeros_like(self.weights[0])

        grad_bias = np.sum(dout, axis=(0, 2, 3))

        x_pad = np.pad(self.last_input, ((0,0),(0,0),
                                         (self.padding[0], self.padding[0]),
                                         (self.padding[1], self.padding[1])),
                                         'constant', constant_values = 0)
        dx_pad = np.zeros_like(x_pad)

        for i in range(out_h):
            for j in range(out_w):
                for k in range(self.out_channels):
                    wstart = i*self.stride[0]
                    wend = i*self.stride[0]+self.kernel_size[0]
                    hstart = j*self.stride[1]
                    hend = j*self.stride[1]+self.kernel_size[1]
                    window = x_pad[:, :, wstart:wend, hstart:hend]

                    # Compute the gradient of the loss with respect to the weights
                    grad_weight[k] += np.sum(window * (dout[:, k, i, j])[:, None, None, None], axis=0)

                    # Compute the gradient of the loss with respect to the input
                    dx_pad[:, :, wstart:wend, hstart:hend] += weights[k] * (dout[:, k, i, j])[:, None, None, None]

        self.gradients = [grad_weight, grad_bias]
        
        # Remove padding from the gradient of the input
        dx = dx_pad[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]

        return dx
    
class Embedding(Layer):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        #initialize weights with normal distribution instead of xavier
        weights = np.random.rand(num_embeddings, embedding_dim) 
        self.weights = [weights]
        self.numweights = len(self.weights)
        
    def forward(self, x):
        return self.weights[0][x]
    
    def calculate_gradients(self, dout):
        grad_weights = np.zeros_like(self.weights[0])
        
        #gradient w.r.t. weights
        np.add.at(grad_weights, self.last_input, dout)
        self.gradients = [grad_weights]
        
        #gradient w.r.t. inputs
        dx = np.zeros(len(self.last_input))
        for i,ind in enumerate(self.last_input):
            dx[i] = np.dot(dout[i], self.weights[0][ind])
        
        return dx
    
    def from_pretrained(self, weights):
        num_embeddings, embedding_dim = weights.shape
        self.__init__(num_embeddings, embedding_dim)
        self.weights = [weights]
        return self
    
############################# RECURRENT LAYERS ##############################
    
class LSTMCell(Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_type = 'r'
        
        weight_ih = self.xavier_init((4*hidden_size, input_size))
        weight_hh = self.xavier_init((4*hidden_size,hidden_size))
        bias_ih = np.zeros((4*hidden_size))
        bias_hh = np.zeros((4*hidden_size))
        self.weights = [weight_ih, weight_hh, bias_ih, bias_hh]
        self.numweights = len(self.weights)

        self.input_gates = []
        self.forget_gates = []
        self.cell_gates = []
        self.output_gates = []
            
        self.c_t = []
        self.h_t = []
        self.inputs = []
        self.timesteps = 0
          
    def __call__(self, x, hc):        
        h, c = hc
        if self.timesteps == 0:
            self.h_t.append(h)
            self.c_t.append(c)
        new_h, new_c = self.forward(x, h, c)
        
        #save cell and hidden states of each timestep
        self.c_t.append(new_c)
        self.h_t.append(new_h)
        self.inputs.append(x)
        self.timesteps += 1
        return (new_h, new_c)
            
    def forward(self, x, h, c):
        batch_size = x.shape[0] if len(x.shape) == 2 else 1 
        weight_ih, weight_hh, bias_ih, bias_hh = self.weights
        
        # Linear transformations        
        gates = x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh
        
        # Split into 4 separate tensors
        i, f, g, o = np.split(gates, 4, axis=1)
        
        # Apply gate activation functions
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = np.tanh(g)
        o = self.sigmoid(o)
        
        #store gates for quicker backward method
        self.input_gates.append(i)
        self.forget_gates.append(f)
        self.cell_gates.append(g)
        self.output_gates.append(o)
        
        # Calculate new cell state and hidden state
        new_c = f * c + i * g
        new_h = o * np.tanh(new_c)
        
        return (new_h, new_c)
    
    def calculate_gradients(self, dh):        
        #initialize gradients to zero so we can sum over timesteps
        self.gradients = [np.zeros_like(weight) for weight in self.weights]
                
        dc = dh * self.output_gates[-1] * (1 - np.square(np.tanh(self.c_t[-1])))
        
        while self.timesteps > 0:
            dx, dh, dc = self.backward_timestep(dh, dc)
        
        return dx
        
    def backward_timestep(self, dh, dc):
        weight_ih, weight_hh, bias_ih, bias_hh = self.weights
        grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh = self.gradients
        
        #collection of stored variables we need for this timestep
        c = self.c_t[-1]
        c_tanh = np.tanh(c)
        cprev = self.c_t[-2]
        hprev = self.h_t[-2]
        x = self.inputs[-1]            
 
        #Gates
        i = self.input_gates[-1]
        f = self.forget_gates[-1]
        g = self.cell_gates[-1]
        o = self.output_gates[-1]
        
        #Derivative of gates
        di = i * (1 - i)
        df = f * (1 - f)
        dg = 1 - np.square(g)
        do = o * (1 - o)
        
        #gradient w.r.t. gates
        grad_i = dc * g * di
        grad_f = dc * cprev * df
        grad_g = dc * i * dg
        grad_o = dh * c_tanh * do
            
        #gradient w.r.t. weights
        dgates = np.concatenate([grad_i,grad_f,grad_g,grad_o], axis = 1)
        
        grad_weight_ih += dgates.T @ x
        grad_weight_hh += dgates.T @ hprev
        grad_bias = np.sum(dgates, axis = 0)
        grad_bias_ih += grad_bias
        grad_bias_hh += grad_bias
        
        #gradient w.r.t. previous hidden state and cell state        
        dhprev = dgates @ weight_hh
        dcprev = None
        if self.timesteps > 1:
            #gradient w.r.t. previous cell state
            oprev = self.output_gates[-2]
            dcprev = f * dc + dhprev * oprev * (1 - np.square(np.tanh(cprev)))
            
        #gradient w.r.t. input
        dx = dgates @ weight_ih
            
        #remove the last timestep from the saved states
        self.c_t.pop()
        self.h_t.pop()
        self.inputs.pop()
        self.input_gates.pop()
        self.forget_gates.pop()
        self.cell_gates.pop()
        self.output_gates.pop()
        self.timesteps -= 1
        
        self.gradients = [grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh]
        return dx, dhprev, dcprev
    
class GRUCell(Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_type = 'r'
        
        weight_ih = self.xavier_init((3*hidden_size, input_size))
        weight_hh = self.xavier_init((3*hidden_size, hidden_size))
        bias_ih = np.zeros((3*hidden_size))
        bias_hh = np.zeros((3*hidden_size))
        self.weights = [weight_ih, weight_hh, bias_ih, bias_hh]
        self.numweights = len(self.weights)

        self.r_gates = []
        self.z_gates = []
        self.n_gates = []
        self.n_hidden = []
            
        self.h_t = []
        self.inputs = []
        self.timesteps = 0
        
    def __call__(self, x, h):
        if self.timesteps == 0:
            self.h_t.append(h)
        new_h = self.forward(x, h)
        
        #save cell and hidden states of each timestep
        self.h_t.append(new_h)
        self.inputs.append(x)
        self.timesteps += 1
        return new_h
        
    def forward(self, x, h):
        weight_ih, weight_hh, bias_ih, bias_hh = self.weights
        batch_size = x.shape[0] if len(x.shape) == 2 else 1 
        
        # Linear transformations        
        input_gates = x @ weight_ih.T + bias_ih
        hidden_gates = h @ weight_hh.T + bias_hh
        
        # Split into 3 separate tensors
        r_in, z_in, n_in = np.split(input_gates, 3, axis=1)
        r_hid, z_hid, n_hid = np.split(hidden_gates, 3, axis = 1)
        
        #create gates
        r = self.sigmoid(r_in + r_hid)
        z = self.sigmoid(z_in + z_hid)
        n = np.tanh(n_in + r * n_hid)
        
        #store gates for quicker backward method
        self.r_gates.append(r)
        self.z_gates.append(z)
        self.n_gates.append(n)
        self.n_hidden.append(n_hid) #makes backprop easier
        
        # Calculate new hidden state
        return (1 - z) * n + z * h
    
    def calculate_gradients(self, dh):        
        #initialize gradients to zero so we can sum over timesteps
        self.gradients = [np.zeros_like(weight) for weight in self.weights]
        
        while self.timesteps > 0:
            dx, dh = self.backward_timestep(dh)
        
        return dx
        
    def backward_timestep(self, dh):
        weight_ih, weight_hh, bias_ih, bias_hh = self.weights
        grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh = self.gradients
        
        #collection of stored variables we need for this timestep
        h = self.h_t[-1]
        hprev = self.h_t[-2]
        x = self.inputs[-1]            
        n_hid = self.n_hidden[-1]
        
        #Gates
        r = self.r_gates[-1]
        z = self.z_gates[-1]
        n = self.n_gates[-1]
        
        #derivative of gates
        dr = r * (1 - r)
        dz = z * (1 - z)
        dn = 1 - np.square(n)
        
        #gradient w.r.t. gates
        grad_z = dh * (hprev - n) * dz
        grad_n = dh * (1 - z) * dn
        grad_r = grad_n * n_hid * dr
        grad_n_hid = grad_n * r
        
        dinput_gates = np.concatenate([grad_r,grad_z,grad_n], axis = 1)
        dhidden_gates = np.concatenate([grad_r,grad_z,grad_n_hid], axis = 1)
                
        a = 0
        print("h")
        print(h)
        print(np.sum(h, axis = a))
        print()
        print("dh")
        print(dh)
        print(np.sum(dh, axis = a))
        print()
        print("r")
        print(r)
        print(np.sum(r, axis = a))
        print()
        print("dr")
        print(dr)
        print(np.sum(dr, axis = a))
        print()
        print("n")
        print(n)
        print(np.sum(n, axis = a))
        print()
        print("dn")
        print(dn)
        print(np.sum(dn, axis = a))
        print()
        print("z")
        print(z)
        print(np.sum(z, axis = a))
        print()
        print("dz")
        print(dz)
        print(np.sum(dz, axis = a))
        print()
        print("hprev")
        print(hprev)
        print(np.sum(hprev,axis = a))
        print()
        print("grad_z")
        print(grad_z)
        print(np.sum(grad_z, axis = a))
        print()
        print("grad_z / B")
        print(grad_z)
        print(np.sum(grad_z, axis = a)/ grad_z.shape[0])

        
        
        
        #gradient w.r.t. input weights
        grad_weight_ih += dinput_gates.T @ x
        grad_bias_ih += np.sum(dinput_gates, axis = 0)
 
        #gradient w.r.t. hidden weights
        grad_weight_hh += dhidden_gates.T @ hprev
        grad_bias_hh += np.sum(dhidden_gates, axis = 0)
                
        #gradient w.r.t. previous hidden state
        dhprev = dhidden_gates @ weight_hh + z * dh
        
        #gradient w.r.t. input
        dx = dinput_gates @ weight_ih

        #remove the last timestep from the saved states
        self.h_t.pop()
        self.inputs.pop()
        self.r_gates.pop()
        self.z_gates.pop()
        self.n_gates.pop()
        self.n_hidden.pop()
        self.timesteps -= 1
        
        return dx, dhprev
    

class RNNCell(Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_type = 'r'
        
        weight_ih = self.xavier_init((hidden_size,input_size))
        weight_hh = self.xavier_init((hidden_size,hidden_size))
        bias_ih = np.zeros((hidden_size))
        bias_hh = np.zeros((hidden_size))
        
        self.weights = [weight_ih, weight_hh, bias_ih, bias_hh]
        self.numweights = len(self.weights)
        
        self.h_t = []
        self.inputs = []
        self.timesteps = 0
        
        self.activation = a.tanh()
        
    def __call__(self, x, h):
        if self.timesteps == 0:
            self.h_t.append(h)
        new_h = self.forward(x, h)
                    
        #save cell and hidden states of each timestep
        self.h_t.append(new_h)
        self.inputs.append(x)
        self.timesteps += 1
        return new_h
        
    def forward(self, x, h):
        weight_ih, weight_hh, bias_ih, bias_hh = self.weights
        batch_size = x.shape[0] if len(x.shape) == 2 else 1 
        
        # Linear transformations        
        h = x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh
        
        return self.activation(h)
    
    def calculate_gradients(self, dh):
        #initialize gradients to zero so we can sum over timesteps
        self.gradients = [np.zeros_like(weight) for weight in self.weights]
        
        while self.timesteps > 0:
            dx, dh = self.backward_timestep(dh)
        
        return dx
        
    def backward_timestep(self, dh):
        weight_ih, weight_hh, bias_ih, bias_hh = self.weights
        grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh = self.gradients
        
        hprev = self.h_t[-2]
        x = self.inputs[-1] 
        
        #gradient w.r.t. weight_ih
        grad_weight_ih += dh.T @ x
        
        #gradient w.r.t. weight_hh
        grad_weight_hh += dh.T @ hprev
        
        #gradient w.r.t. bias
        grad_bias = np.sum(dh, axis = 0)
        grad_bias_ih += grad_bias
        grad_bias_hh += grad_bias
        
        #gradient w.r.t. previous hidden state
        dhprev = dh @ weight_hh.T
        
        #gradient w.r.t. input
        dx = dh @ weight_ih
        
        #remove the last timestep from the saved states
        self.h_t.pop()
        self.inputs.pop()
        self.timesteps -= 1
        
        return dx, dhprev
        
    
############################## TRANSFORMATION LAYERS #############################

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.last_input.shape)
    
class FromRNN(Layer):
    #Drops second output from previous layer
    #E.g. used in LSTM to Linear layer
    def __init__(self, take_last = True):
        super().__init__()
        self.layer_type = 'r'
        
        #Take last represents taking the last output from the rnn
        #E.g. the output from the last layer on the last timestep
        self.take_last = take_last
        
    def __call__(self, x, hc):
        #Can do something with h / hc if you want
        if self.take_last:
            new_x = x[-1]
        
        if self.activation is not None:
            new_x = self.activation(new_x)
        return new_x, None
    
    def backward(self, dout):
        if self.activation is not None:
            return self.activation.backward(dout)
        return dout
    
class ActivationLayer(Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        
    def __call__(self, x):
        return self.activation(x)
    
    def backward(self, dout):
        return self.activation.backward(dout)
        







    