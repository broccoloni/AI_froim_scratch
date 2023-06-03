# AI_from_scratch
I built a variety of AI tools in numpy that can be used similar to keras. 
The forward and backward calculations have been verified by comparison with Pytorch, as shown in Model_Test.ipynb

For example, one can instantiate a model and add layers as follows

```
import numpy as np
from Models import Model
from Layers import Linear
import Activations as a

model = Model() 
model.addLayer(Linear(10,20), activation = a.ReLU())
model.addLayer(Linear(20,3))
```

Currently this code supports the following layers
 - Linear
 - Conv2d
 - Embedding
 - RNNCell
 - RNN
 - LSTMCell
 - LSTM
 - Flatter (e.g. for Conv2d to Linear)
 - fromRNN (to utilise / dismiss outputs of previous timesteps)
 - ActivationLayer (to add an activation, not as part of another layer)

Additionally, the following loss functions are supported
 - sigmoid
 - tanh
 - ReLU

After the model architecture has been created, one can add a loss function. For example,

```
import Loss_Functions as lf

model.add_loss_fn(lf.MSELoss())
```

The loss functions supported are 
 - MSELoss
 - NLLLoss
 - CrossEntropy

Then, we can pass data forward through the network, calculate the loss, and call backward to update the gradients
For example,

```
output = model(x)
model.calculate_loss(y_true)
model.backward()
```

or if a loss function has not been defined for the model,

```
output = model(x)
model.backward(dout = output_gradients)
```

TODOs
 - Fix GRUCell gradients
 - Convert CTCLoss torch implementation I created to numpy
 - Add Transformer Layer
 - Add optimizers
