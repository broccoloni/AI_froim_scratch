import numpy as np

class LossFn():
    def __init__(self):
        self.ypred = None
        self.ytrue = None
        self.loss = None
        self.gradient = None
        
    def __call__(self, ypred, ytrue):
        self.ypred = ypred
        self.ytrue = ytrue
        self.loss = self.calculate()
        self.gradient = self.derivative()
        return self.loss
    
    def calculate(self):
        return 0
        
    def derivative(self):
        return np.zeros_like(self.ypred)
    
class MSELoss(LossFn):
    def __init__(self):
        super().__init__()

    def calculate(self):
        return np.mean((self.ypred - self.ytrue)**2)
        
    def derivative(self):
        return -2*(self.ytrue - self.ypred)/self.ytrue.size
        
class NLLLoss(LossFn):
    #Assumes that y_pred is already log probabilities
    #Assumes y_target is list of labels
    def __init__(self):
        super().__init__()
        
    def calculate(self):
        batch_size = self.ypred.shape[0] if len(self.ypred.shape) == 2 else 1
        return -np.sum(self.ypred[range(batch_size),self.ytrue])/batch_size
        
    def derivative(self):
        batch_size = self.ypred.shape[0] if len(self.ypred.shape) == 2 else 1
        onehot_labels = np.eye(self.ypred.shape[-1])[self.ytrue].squeeze()
        return (np.zeros(self.ypred.shape)-onehot_labels)/batch_size
        
class CrossEntropyLoss(LossFn):
    #Last layer has not had softmax applied yet
    #Assumes y_target is list of labels
    def __init__(self):
        super().__init__()
        self.probs = None
        
    def calculate(self):
        batch_size = self.ypred.shape[0] if len(self.ypred.shape) == 2 else 1

        self.probs = self.softmax(self.ypred, axis = 1)
        logprobs = -np.log(self.probs[range(batch_size),self.ytrue])
        return np.sum(logprobs)/batch_size
    
    def derivative(self):
        batch_size = self.ypred.shape[0] if len(self.ypred.shape) == 2 else 1
        d = self.probs
        d[range(batch_size),self.ytrue] -= 1
        return d/batch_size
                 
    def softmax(self, x, axis = None):
        #Numerically stable version
        e = np.exp(x - np.max(x, axis = axis, keepdims = True))
        return e/np.sum(e, axis = axis, keepdims = True)
             
        
#Merge this into numpy implementation
class CTCLoss(nn.Module):
    def __init__(self,rnn, decode_method = 'bp'):
        super(CTC,self).__init__()
        self.rnn = rnn
        self.decode_method = decode_method
        self.rnn_outputs = None # result of forward method
        self.probs = None       # result of forward method
        self.decoding = None    # result of decode method
        self.lprime = None      # result of process_decoding method
        self.gradient = None    # result of backward method

    def forward(self,x):
        """
        This function performs forward propagation. It processes a given input
        through the rnn input on object instantiation. Then, it calculates the
        softmax probabilities for each timestep, and stores them in self.probs.
        """
        self.rnn_outputs = torch.t(self.rnn.forward(x))
        self.probs = nn.functional.softmax(self.rnn_outputs.detach(),dim = 0)

    def backward(self):
        """
        This function performs backpropagation. It calculates the gradient using
        the method described in the paper, and propagates that backward through
        the rnn input in the instantiation of a CTC object.
        """
        
        if self.probs is None: 
            print("Error: forward method must be called before backward")
            return 
            
        self.calculate_gradient()
        self.rnn_outputs.backward(gradient = self.gradient)
        
    def clear_calculations(self):
        """
        This function is used to clear the calculations of other functions that 
        are stored as attributes. This can help to ensure that they current and relate 
        to the most recent input.
        """
        
        self.rnn_outputs = None
        self.probs = None
        self.decoding = None
        self.lprime = None
        self.gradient = None

    def calculate_gradient(self):
        """
        This function calculates the gradient according to equation 16
        in the CTC paper. It stores the result in self.gradient
        
        Note: self.probs must be calculated first. self.decode and self.lprime 
              should also exist to ensure the decoding is current.
        """
        
        if self.probs is None: 
            print("Error: forward method has to be called before calculate_gradient")
            return None
        
        if self.decoding is None:
            self.decode()
            
        if self.lprime is None:
            self.process_decoding()
        
        numlabels, _ = self.probs.shape
        
        a = self.forward_dp_calculation()
        b = self.backward_dp_calculation()

        ab = torch.multiply(a,b)

        Z = torch.sum(ab/self.probs[self.lprime,:],axis = 0)

        Q = torch.sum(ab,axis = 1)
        Q = torch.tensor([torch.sum(Q[self.lprime == i]) for i in range(numlabels)])
        Q = Q.reshape((numlabels,1))

        self.gradient =  self.probs - 1/(self.probs*Z)*Q

    def forward_dp_calculation(self):
        """
        This function is the forward dynamic programming 
        algorithm used to calculate the probability of a
        particular sequence of output labels

        OUTPUT:
            a - the dynamic programming calculations. These are 
                used in the calculation of the probability of a
                given label, and gradient for backpropagation
        """
        numlabels, numtimesteps = self.probs.shape
        seqlen = len(self.lprime)
        a = torch.zeros((seqlen,numtimesteps))
        
        #If only blanks are predicted, there is only one path
        #after applying scaling to avoid underflow, this results
        #in the probability of the blanks for each a[0,:]
        if seqlen == 1:
            a[0,:] = self.probs[-1,:]
            return a
        
        #initial conditions
        a[0,0] = self.probs[-1,0]
        a[1,0] = self.probs[self.lprime[1],0]

        for t in range(1,numtimesteps): 
            #update rules as described in the paper
            a[:,t] = a[:,t-1]+torch.cat((torch.tensor([0]),a[:-1,t-1]))
            for s in range(3,seqlen,2):
                if self.lprime[s] != self.lprime[s-2]:
                    a[s,t] += a[s-2,t-1]

            #Paths less than cutoff are set to zero, as they no
            #longer have enough time to reach the final label
            cutoff = seqlen - 2*(numtimesteps-t)
            if cutoff>0:
                a[:cutoff, t] *= 0

            #multiply by corresponding label probability and 
            #divide by previous sum to minimize underflow
            a[:,t] *= self.probs[self.lprime,t]/torch.sum(a[:,t-1])
        return a

    def backward_dp_calculation(self):
        """
        This function is used in the backward dynamic programming
        algorithm used to calculate the probability of a 
        particular sequence of output labels.

        OUTPUT:
            b - the dynamic programming calculations. These are 
                used in the calculation of the probability of a
                given label, and gradient for backpropagation
        """
        
        numlabels, numtimesteps = self.probs.shape
        seqlen = len(self.lprime)
        b = torch.zeros((seqlen,numtimesteps))

        #If only blanks are predicted, there is only one path
        #after applying scaling to avoid underflow, this results
        #in the probability of the blanks for each b[0,:]
        if seqlen == 1:
            b[0,:] = self.probs[-1,:]
            return b
        
        #initial conditions
        b[-1,-1] = self.probs[-1,-1]
        b[-2,-1] = self.probs[self.lprime[-2],-1]

        for t in range(numtimesteps-2,-1,-1):
            #update rules as described in the paper
            b[:,t] = b[:,t+1] + torch.cat((b[1:,t+1],torch.tensor([0])))
            for s in range(seqlen-4,0,-2):
                if self.lprime[s] != self.lprime[s+2]:
                    b[s,t] += b[s+2,t+1]


            #Paths greater than cutoff are set to zero, as they no
            #longer have enough time to reach the initial label
            cutoff = 2*(t+1)
            if cutoff<seqlen:
                b[cutoff:, t] *= 0

            #multiply by corresponding label probability and 
            #divide by previous sum to minimize underflow
            b[:,t] *= self.probs[self.lprime,t]/torch.sum(b[:,t+1])
        return b

    def decode(self):
        """ 
        This function assigns a label for the probabilistic output
        at each timestep. It then removes sequential duplicates and 
        blanks, and stores the result in self.decoding.

        Note: self.probs must be calculated first using the forward method
        
        Currently I've only implemented the best path, or greedy decoding.
        Other methods could be implemented such as prefix search, as proposed
        in the CTC paper.
        """
        
        if self.probs is None:
            print("Error: forward method must be called before decode")
            return
        
        numlabels, numtimesteps = self.probs.shape

        #Best Path decoding
        if self.decode_method == 'bp':
            self.decoding = torch.argmax(self.probs, axis = 0)
            
            
        #Prefix Search decoding
        elif self.decode_method == 'ps':
            # Choose boundary points where prob of observing a 
            # blanks label is above a certain threshold. Then,
            # calculate most probable labelling for each section 
            # using the forward backward algorithm, and 
            # concatenate each section to get the labels.
            pass
        
        #Invalid decoding method
        else:
            print("Error: invalid decoding method")
            return
        
        # Remove sequential label duplicates and blanks.
        # Pytorch doesn't have a nice delete method like numpy,
        # so specifying which elements to keep is the cleanest 
        # work around.
        toKeep = []
        if self.decoding[0] != numlabels-1:
            toKeep.append(0)
        for t in range(1,numtimesteps):
            cur = self.decoding[t]
            prev = self.decoding[t-1]
            if cur != prev and cur != numlabels-1:
                toKeep.append(t)

        self.decoding = self.decoding[toKeep]

    def process_decoding(self):
        """ 
        This function inserts blanks between the decoded label elements.
        The result is stores in self.lprime, as lprime is the notation used in 
        the CTC paper.

        Note: self.decoding must be calculated first using the decode method.
              self.probs should also exist to ensure the decoding is current.
        """
        
        if self.probs is None:
            print("Error: forward method must be called before process_decoding")
            return
        
        if self.decoding is None:
            print("Error: decode method must be called before process_decoding")
            return
        
        numlabels, numtimesteps = self.probs.shape

        #add blanks between labels without blanks
        self.lprime = torch.full((2*len(self.decoding)+1,),numlabels-1)
        self.lprime[1::2] = self.decoding
        
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   