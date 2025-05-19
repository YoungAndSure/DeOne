import deone.core as Core
from deone.core import *
from deone.function import *
from deone.function_simple import *
from deone.config import *
import numpy as np
import weakref
import os
import logging

class Parameter(Variable) :
    pass

class Layer :
    def __init__(self) :
        # 保存所有Parammeter和Layer的名字，__dict__中保存了所有类型成员的参数名和值
        self._params = set()
    
    def __setattr__(self, name, value) :
        if isinstance(value, (Parameter, Layer)) :
            self._params.add(name)
        super().__setattr__(name, value)
    
    def __call__(self, *inputs) :
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple) :
            outputs = (outputs,)
        self.inputs = [weakref.ref(input) for input in inputs]
        # Funcion里整个是虚的，因为你给阉割了，输出的outputs都是单个值
        self.outputs = [weakref.ref(output) for output in outputs]
        # DAMN : 原来这么简单，我直接阉割了Function，让它只支持输出一个output，来实现这个功能
        return outputs if len(outputs) > 1 else outputs[0]
 
    def forward(self) :
        raise NotImplementedError()

    def params(self) :
        # 为什么不直接遍历__dict__，而是非要存储个params？
        # 因为class的所有属性都会存在__dict__中，包括上边的inputs/outputs
        # 而用户只需要返回 params
        for name in self._params :
            param = self.__dict__[name]
            if isinstance(param, Parameter) :
                yield param
            else :
                yield from param.params()

    def cleargrad(self) :
        for param in self.params() :
            param.cleargrad()

    def _flatten_params(self, params_dict, params_key="") :
        for name in self._params :
            obj = self.__dict__[name]
            key = params_key + '/' + name if params_key != "" else name
            if isinstance(obj, Layer) :
                obj._flatten_params(params_dict, key)
            else :
                params_dict[key] = obj

    def save_weights(self, path) :
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {}
        for k,v in params_dict.items() :
            if v is None :
                continue
            array_dict[k] = v.data
        try :
            np.savez_compressed(path, **array_dict)
            print("dump params to {} success".format(path))
        except(Exception, KeyboardInterrupt) as e :
            if os.path.exists(path) :
                os.remove(path)
            raise

    def load_weights(self, path) :
        if os.path.exists(path) :
            npz = np.load(path)
            params_dict = {}
            self._flatten_params(params_dict)
            for key,param in params_dict.items() :
                param.data = npz[key]
            print("load params from {} success".format(path))
        else :
            raise

class Linear(Layer) :
    def __init__(self, out_size, has_bias=True, dtype=np.float32, in_size=None) :
        super().__init__()
        self.I = in_size
        self.O = out_size
        self.W = Parameter(None, name='W')
        self.has_bias = has_bias
        self.dtype = dtype
        if self.has_bias :
            self.b = Parameter(np.zeros(self.O, dtype=self.dtype), name='b')
        else :
            self.b = None
        if self.I != None :
            self._init_W()

    def _init_W(self) :
        if Config.close_random :
            W_data = np.zeros((self.I, self.O)).astype(self.dtype) * np.sqrt(1 / self.I)
        else :
            # NOTE:惊天大BUG之二，也是查了很久，最后通过和官方版本打印对比找到。发现这里关掉随机后两边输出一致，确定问题就出在这一行。
            # 原版是 * np.sqrt，我误以为是类似“归一化”的除操作。具体为什么这么初始化需要看看论文。
            W_data = np.random.randn(self.I, self.O).astype(self.dtype) * np.sqrt(1 / self.I)
        self.W = Parameter(W_data, name='W')

    def forward(self, x) :
        if self.W.data is None :
            self.I = x.shape[1]
            self._init_W()
        y = linear(x, self.W, self.b)
        return y

class RNN(Layer) :
    def __init__(self, hidden_size, in_size=None) :
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = Linear(hidden_size, in_size = in_size)
        self.h2h = Linear(hidden_size, in_size = hidden_size, has_bias=False)
        self.h = None
    
    def reset_status(self) :
        self.h = None
    
    # self.h存了t时的输出，下次t+1调用时用self.h计算.这是一种串行执行的方式
    # 在这种使用方式下，必须保证就这一个对象，循环调用这一个对象的forward
    # 但是在TimeRNN中，是多个rnn对象连接在一起，rnn 1 接受的是rnn 0的输出
    # 也就必须得从rnn 0获取到输出，然后传入给rnn 1才行，rnn 1的h是从外部来的
    # 所以必须加一个入参h，为了不影响之前的用法，给h赋默认值
    # 不过，不管是几个rnn，其实都是串行，耗时一样，内存多个rnn > 一个rnn
    def forward(self, x, h=None) :
        if h is None :
            h = self.h

        if h is None :
            y = tanh(self.x2h(x))
        else :
            y = tanh(self.x2h(x) + self.h2h(h))
        self.h = y
        return y

class TimeRNN(Layer) :
    def __init__(self, T, hidden_size, in_size=None, stateful=False) :
        super().__init__()
        self.rnn_layers = []
        for i in range(T) :
            self.rnn_layers.append(RNN(hidden_size, in_size = in_size))
        self.h = None
        self.T = T
        self.stateful = stateful
    
    def set_states(self, h) :
        self.h = h

    def reset_status(self) :
        self.h = None

    def forward(self, xs) :
        BATCH, TIME, DENSE = xs.shape
        if TIME != self.T :
            raise ValueError("T error, init T:{}".format(self.T))

        h = None 
        for t in range(self.T) :
            rnn = self.rnn_layers[t]
            if h is None :
                h = rnn(xs[:, t, :])
            else :
                h = rnn(xs[:, t, :], h)
        
        if self.stateful :
            self.h = h
        
        return h


class LSTM(Layer) :
    def __init__(self, hidden_size, in_size=None) :
        super().__init__()
        self.x2f = Linear(hidden_size) 
        self.x2i = Linear(hidden_size)
        self.x2o = Linear(hidden_size)
        self.x2u = Linear(hidden_size)

        self.h2f = Linear(hidden_size, has_bias=False)
        self.h2i = Linear(hidden_size, has_bias=False)
        self.h2o = Linear(hidden_size, has_bias=False)
        self.h2u = Linear(hidden_size, has_bias=False)

        self.reset_status()
    
    def reset_status(self) :
        self.c = None
        self.h = None

    def forward(self, x) :
        f,i,o,u = None,None,None,None
        if self.h is None :
            f = sigmod_simple(self.x2f(x))
            i = sigmod_simple(self.x2i(x))
            o = sigmod_simple(self.x2o(x))
            u = tanh(self.x2u(x))
        else :
            f = sigmod_simple(self.x2f(x) + self.h2f(self.h))
            i = sigmod_simple(self.x2i(x) + self.h2i(self.h))
            o = sigmod_simple(self.x2o(x) + self.h2o(self.h))
            u = tanh(self.x2u(x) + self.h2u(self.h))
        
        if self.c is None :
            c = i * u
        else :
            c = f * self.c + i * u
        h = o * tanh(c)

        self.c = c
        self.h = h

        return h

class Embedding(Layer) :
    def __init__(self, in_size, out_size, dtype=np.float32) :
        super().__init__()
        self.I = in_size
        self.O = out_size
        self.W = Parameter(None, name='W')
        self.dtype = dtype
        self._init_W()

    def _init_W(self) :
        if Config.close_random :
            W_data = np.zeros((self.I, self.O)).astype(self.dtype) * np.sqrt(1 / self.I)
        else :
            W_data = np.random.randn(self.I, self.O).astype(self.dtype) * np.sqrt(1 / self.I)
        self.W = Parameter(W_data, name='W')

    def forward(self, idx) :
        # idx类型是Variable，get_item中的slices不能是Variable,进行了变换
        y = get_item(self.W, idx)
        return y