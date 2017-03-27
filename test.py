import lasagne
import MyLNLSTMLayer
import theano
import theano.tensor as T
import os
import numpy as np
os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,warn_float64=warn,compute_test_value = warn,optimizer=None,exception_verbosity='high',allow_gc=False"

inputs_var = T.tensor3('inputs_var')
inputs_var.tag.test_value = np.random.rand(20,5,539)
forget_gate = lasagne.layers.recurrent.Gate(b=lasagne.init.Constant(5.0))

l_input = lasagne.layers.InputLayer(shape=(None, 5, 539))
l_lstm = MyLNLSTMLayer.LNLSTMLayer(l_input, num_units=10,
                  forgetgate=forget_gate,peepholes=False)
ouput = lasagne.layers.get_output(l_lstm,inputs=inputs_var)

get_lstmoutput = theano.function([inputs_var],ouput)

inputs = np.random.rand(20,5,539)
outputs = get_lstmoutput(inputs)
