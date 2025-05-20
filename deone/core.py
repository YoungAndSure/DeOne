import numpy as np
import weakref
from deone.config import *
import deone.util
import logging

class Variable :
    def __init__(self, data, name=None) :
        # NOTE:如果传入的data是None，直接就没有data这个成员了，也就打印不出
        # 这个问题出现在Parameter未初始化W却要打印时
        if (data is not None) :
            if (not isinstance(data, np.ndarray)) :
                raise TypeError('{} is not supported'.format(type(data)))
            self.data = data
        else :
            self.data = None
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    @property 
    def shape(self) :
        return self.data.shape
    @property
    def ndim(self) :
        return self.data.ndim
    @property
    def dtype(self) :
        return self.data.dtype
    
    def __len__(self) :
        return len(self.data)
    
    def __repr__(self) :
        if self.data is None :
            return 'variable(None)'
        p = str(self.data)
        return 'variable(' + p + ')'

    def backward(self, retain_grad=False, create_graph=False) :
        if Config.enable_backward == False :
            #print("backward disabled")
            return

        if self.grad == None :
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()
        def add_func(f) :
            if f not in seen_set :
                funcs.append(f)
                seen_set.add(f)
            funcs.sort(key = lambda x : x.generation)
        add_func(self.creator)

        while funcs :
            func = funcs.pop()
            output_grads = func.outputs().grad
            # 在反向传播时，可以认为实际上在执行一次新的正向传播，
            # 此时关掉enable_backward，也就关掉了这次正向传播对应的反向传播，
            # 对应第一次正向传播，也就是二次求导
            with using_config('enable_backward', create_graph) :
                input_grads = func.backward(output_grads)
                if not isinstance(input_grads, tuple) :
                    input_grads = (input_grads, )
                for input, input_grad in zip(func.inputs, input_grads) :
                    if (input.grad == None) :
                        input.grad = input_grad
                    else :
                        input.grad = input.grad + input_grad
                    if (input.creator != None) :
                        add_func(input.creator)
                if retain_grad == False :
                    func.outputs().grad = None
    
    def unchain(self) :
        self.creator = None
    
    def unchain_backward(self) :
        if self.creator is not None :
            funcs = [self.creator]
            # TODO: why not?
            # self.unchain()
            while funcs :
                f = funcs.pop()
                for input in f.inputs :
                    if input.creator is not None :
                        funcs.append(input.creator)
                        input.unchain()

    def cleargrad(self) :
        self.grad = None

    def reshape(self, *shape) :
        # *会把输入变成一个tuple，所以，如果传入(6,)，这里读出来是((6,))
        # 如果传入2,3，这里读出来反倒是(2,3)，可以直接透传
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)) :
            shape = shape[0]
        import deone.function as F
        return F.reshape(self, shape)
    
    def transpose(self, *axis) :
        import deone.function as F
        return F.transpose(self, axis)
    @property
    def T(self) :
        import deone.function as F
        return F.transpose(self)

    def sum(self, dim, keepdim=False) :
        import deone.function as F
        return F.sum(self, dim, keepdim)

    def matmul(self, W) :
        import deone.function as F
        return F.matmul(self, W)

    def dim(self) :
        return self.data.ndim

def as_variable(input) :
    if not isinstance(input, Variable) :
        return Variable(input)
    return input

def as_array(x) :
    if (np.isscalar(x)) :
        return np.array(x)
    return x

class Function :
    def __call__(self, *inputs) :
        inputs = [as_variable(as_array(input)) for input in inputs]
        input_datas = [input.data for input in inputs]
        self.generation = max([input.generation for input in inputs])

        output_datas = self.forward(*input_datas)
        outputs = Variable(as_array(output_datas))
        if Config.enable_backward == True :
            outputs.creator = self
            outputs.generation = self.generation + 1

            # 真他妈巧妙，这里function持有的是虚的，向后传递的是实的，不用改动任何接口
            # 只有在用function里这个output的时候才需要加()
            # THINK: 为什么这里output是虚的，input要是实的?
            self.outputs = weakref.ref(outputs)
            self.inputs = inputs
        return outputs

    def flat_input(self, inputs) :
        result = []
        for input in inputs :
            if (isinstance(input, list) or isinstance(input, tuple)) :
                result = result + input
            else :
                result.append(input)
        return result

    # 和上一版的类似，这两个接口是放给用户写的，所以务必完全适应用户的使用习惯，和系统的适配交由框架解决
    # 这里用户就想写个公式，并不想管几个输入和几个输出的问题，所以框架用两个技巧给解决了：
    # 一个是对输入解包，一个是对输出框架层转换
    # input: array, output: array or scalar
    def forward(self, x) :
        raise NotImplementedError()

    # input: Variable, output: Varibale
    def backward(self, gy) :
        raise NotImplementedError()