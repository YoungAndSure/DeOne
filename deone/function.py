import numpy as np
import weakref
from deone.core import *
import logging

class Add(Function) :
    def forward(self, x0, x1) :
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y
    def backward(self, gy) :
        gy0 = sum_to(gy, self.x0_shape)
        gy1 = sum_to(gy, self.x1_shape)
        return (gy0, gy1)
def add(*inputs) :
    func = Add()
    return func(*inputs)

class Sub(Function) :
    def forward(self, x0, x1) :
        y = x0 - x1
        return y
    def backward(self, gy) :
        return (gy, -1 * gy)
def sub(*inputs) :
    func = Sub()
    return func(*inputs)
def rsub(*inputs) :
    func = Sub()
    return func(inputs[1], inputs[0])

class Mul(Function) :
    def forward(self, x0, x1) :
        y = x0 * x1
        return y
    def backward(self, gy) :
        return (gy * self.inputs[1], gy * self.inputs[0])
def mul(*inputs) :
    func = Mul()
    return func(*inputs)

class Square(Function) :
    def forward(self, x) :
        return x ** 2
    def backward(self, gy) :
        return 2 * self.inputs[0] * gy
def square(*x) :
    f = Square()
    return f(*x)

class Exp(Function) :
    def forward(self, x) :
        return np.exp(x)
    def backward(self, gy) :
        return exp(self.inputs[0]) * gy
def exp(*x) :
    f = Exp()
    return f(*x)

class Neg(Function) :
    def forward(self, x) :
        return -1 * x
    def backward(self, gy) :
        return -1 * gy
def neg(*inputs) :
    func = Neg()
    return func(*inputs)

class Div(Function) :
    def forward(self, x0, x1) :
        return x0 / x1
    def backward(self, gy) :
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = -1 * (x0 * gy) / (x1 * x1)
        # NOTE:惊天大bug，找了一周才发现这里没有广播
        # 通过在Varibale.backwards()中逐层打印grad对比得出
        # 除法时候输入可能是一个数除以多维数，或者多维数除以一个数，输出都是多维数，
        # 在对一个数求梯度时需要做sum_to保持形状一致
        if x0.shape != x1.shape:  # for broadcast
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1
def div(*inputs) :
    func = Div()
    return func(*inputs)
def rdiv(*inputs) :
    func = Div()
    return func(inputs[1], inputs[0])

class Pow(Function) :
    def __init__(self, c) :
        self.c = c
    def forward(self, x) :
        return np.power(x, self.c)
    def backward(self, gy) :
        return gy * self.c * pow(self.inputs[0], self.c - 1)
def pow(x, c) :
    func = Pow(c)
    return func(x)

class Sin(Function) :
  def forward(self, x) :
    return np.sin(x)
  def backward(self, gy) :
    return cos(self.inputs[0]) * gy
def sin(x) :
  func = Sin()
  return func(x)

class Cos(Function) :
  def forward(self, x) :
    return np.cos(x)
  def backward(self, gy) :
    return -sin(self.inputs[0]) * gy
def cos(x) :
  func = Cos()
  return func(x)

class Tanh(Function) :
    def forward(self, x) :
        return np.tanh(x)
    def backward(self, gy) :
        return gy * (1 - self.outputs() ** 2)
def tanh(x) :
    func = Tanh()
    return func(x)

class Sum(Function) :
    def __init__(self, axis, keepdims) :
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x) :
        self.input_shape = x.shape
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)
    def backward(self, gy) :
        # trick方法，这样反向传播没有用 dezero 实现的方法，会无法构建连接图，也就没法二次求导了
        #return Variable(np.ones(self.input_shape))
        # TODO:抄的人家的代码，没有特别明白
        gy = Utils.reshape_sum_backward(gy, self.input_shape, self.axis, self.keepdims)
        gy = broadcast_to(gy, self.input_shape)
        return gy
def sum(x, axis=None, keepdims=False) :
    func = Sum(axis, keepdims)
    return func(x)

class BroadcastTo(Function) :
    def __init__(self, output_shape) :
        self.output_shape = output_shape
    def forward(self, x) :
        self.input_shape = x.shape
        return np.broadcast_to(x, self.output_shape)
    def backward(self, gy) :
        return sum_to(gy, self.input_shape)
def broadcast_to(x, shape) :
    func = BroadcastTo(shape)
    return func(x)

class SumTo(Function) :
    def __init__(self, output_shape) :
        self.output_shape = output_shape
    def forward(self, x) :
        self.input_shape = x.shape
        return Utils.sum_to(x, self.output_shape)
    def backward(self, gy) :
        return broadcast_to(gy, self.input_shape)
def sum_to(x, shape) :
    func = SumTo(shape)
    return func(x)

class MatMul(Function) :
    def forward(self, x, W) :
        y = np.matmul(x, W)
        return y
    def backward(self, gy) :
        x = self.inputs[0]
        W = self.inputs[1]
        # 如果x是一维的，ndim == 1，x转置后就是二维的，ndim ==2
        # 而gy和y形状一样，是一维的,ndim == 1,没法和x.T相乘了,
        # 所以这里必须升一下维度，但是目前这么写没法处理三维的

        # 以上都不对，x和w的维度应该一致，不应该输入x是一维的，w是二维的，就会出现各种对不上的问题
        # 这段代码保留，理论上不会执行
        #if gy.data.ndim == 1 :
        #    gy.data = gy.data.reshape(1, len(gy.data))
        if gy.dim() == 2 :
            gx = matmul(gy, W.T)
            gW = matmul(x.T, gy)
        elif gy.dim() == 3 :
            # sorry, too ugly, the pytorch operator need not process batch dim,
            gx = matmul(gy, W.transpose(0, 2, 1))
            gW = matmul(x.transpose(0, 2, 1), gy)
        return (gx, gW)
def matmul(x, W) :
    func = MatMul()
    return func(x, W)

class Log(Function) :
    def forward(self, x) :
        return np.log(x)
    def backward(self, gy) :
        return gy * (1 / self.inputs[0])
def log(x) :
    func = Log()
    return func(x)

class MeanSquareError(Function) :
    def forward(self, x0, x1) :
        diff = x0 - x1
        diff = diff ** 2
        self.N = len(diff.data)
        return diff / self.N
    def backward(self, gy) :
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        gy0 = (2 / self.N) * (x0 - x1) * gy
        gy1 = (2 / self.N) * (x0 - x1) * -1 * gy
        return (gy0, gy1)
def mean_square_error(x0, x1) :
    func = MeanSquareError()
    return func(x0, x1)

class Linear(Function) :
    def forward(self, x, W, b=None) :
        # forward已经在一个Function里了，入出参都是np.array，不能再调用另一个Function
        # backward因为出入参是Variable类型，且需要建立反向传播的连接图，所以必须调用其他已经实现的Function
        t = np.dot(x, W)
        if b is None :
            self.b_shape = None
            return t
        self.b_shape = b.shape
        y = t + b
        return y
    def backward(self, gy) :
        # 入参是3个，返回梯度也是3个，一一对应
        gW = matmul(self.inputs[0].T, gy)
        gb = sum_to(gy, self.b_shape) if self.b_shape is not None else None
        gx = matmul(gy, self.inputs[1].T)
        return (gx, gW, gb)
def linear(*input) :
    func = Linear()
    return func(*input)

class GetItem(Function) :
    def __init__(self, slices) :
        # slices不能是个Variable，对slices求导数也没意义
        if isinstance(slices, Variable) :
            self.slices = slices.data
        else :
            self.slices = slices
    def forward(self, x) :
        return x[self.slices]
    def backward(self, gy) :
        f = GetItemGrad(self.slices, self.inputs[0].shape)
        return f(gy)
class GetItemGrad(Function) :
    def __init__(self, slices, in_shape) :
        self.slices = slices
        self.in_shape = in_shape
    def forward(self, x) :
        y = np.zeros(self.in_shape)
        np.add.at(y, self.slices, x)
        return y
    def backward(self, gy) :
        return get_item(gy, self.slices)
def get_item(x, slices) :
    func = GetItem(slices)
    return func(x)

class Clip(Function) :
    def __init__(self, x_min, x_max) :
        self.x_min = x_min
        self.x_max = x_max
    def forward(self, x) :
        return np.clip(x, self.x_min, self.x_max)
    def backward(self, gy) :
        x = self.inputs[0]
        return gy * (x.data >= self.x_min) * (x.data <= self.x_max)
def clip(x, x_min, x_max) :
    func = Clip(x_min, x_max)
    return func(x)

def max_backward_shape(x, axis) :
    if axis is None :
        axis = range(x.ndim)
    elif isinstance(axis, int) :
        axis = (axis,)

    # axis这个方向被max计算过之后shape变成了1，这里是计算了x被axis方向max之后的shape
    # 按说那不就是y的shape吗？
    shape = [ s if ax not in axis else 1 for ax,s in enumerate(x.shape)]
    return shape

class Max(Function) :
    def __init__(self, axis, keepndims=True) :
        self.axis = axis
        self.keepndims = keepndims
    def forward(self, x) :
        y = np.max(x, axis=self.axis, keepdims=self.keepndims)
        return y
    def backward(self, gy) :
        x = self.inputs[0]
        y = self.outputs()

        # NOTE:这一步暂时没发现有什么用，理论上这个shape和y的shape是一样的，没必要reshape
        shape = max_backward_shape(x, self.axis)
        gy = gy.reshape(shape)
        y = reshape(y, shape)
        # 这里==用的是np提供的广播功能
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond
def vmax(x, axis=None, keepndims=True) :
    func = Max(axis, keepndims)
    return func(x)

class ReLU(Function) :
    def forward(self, x) :
        y = np.maximum(x, 0)
        return y
    def backward(self, gy):
        x = self.inputs[0]
        mask = x.data > 0
        gx = gy * mask
        return gx
def relu(x) :
    func = ReLU()
    return func(x)

class Utils :
    @staticmethod
    def sum_to(x, shape) :
        # copy from book example
        ndim = len(shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))

        axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return y

    @staticmethod
    def reshape_sum_backward(gy, input_shape, axis, keepdims) :
        # copy from book example
        ndim = len(input_shape)
        tupled_axis = axis
        if axis is None:
            tupled_axis = None
        elif not isinstance(axis, tuple):
            tupled_axis = (axis,)

        if not (ndim == 0 or tupled_axis is None or keepdims):
            actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
            shape = list(gy.shape)
            for a in sorted(actual_axis):
                shape.insert(a, 1)
        else:
            shape = gy.shape

        gy = gy.reshape(shape)  # reshape
        return gy

class Reshape(Function) :
    def __init__(self, shape) :
        self.output_shape = shape
    def forward(self, x) :
        self.input_shape = x.shape
        y = x.reshape(self.output_shape)
        return y
    def backward(self, gy) :
        return reshape(gy, self.input_shape)
def reshape(x, shape) :
    func = Reshape(shape)
    return func(x)

class Transpose(Function) :
    def __init__(self, axis) :
        self.input_axis = axis
    def forward(self, x) :
        # 不用特殊处理一维数据, 一维实际就是不能转置的
        #if x.ndim == 1 :
        #    y = np.transpose(x).reshape(-1, 1)
        #elif x.shape[-1] == 1 :
        #    y = np.transpose(x).ravel()
        #else :
        y = np.transpose(x) if (self.input_axis is None or len(self.input_axis) == 0) else np.transpose(x, self.input_axis)
        return y
        #return np.transpose(x) if x.ndim != 1 else np.transpose(x).reshape(-1, 1)
    def backward(self, gy):
        # gy的类型是 Variable , 反向传播需要建立图，也就是要调用 deone 的方法实现
        if self.input_axis is None :
            return transpose(gy)
        inv_axis = tuple(np.argsort([ax % len(self.input_axis) for ax in self.input_axis]))
        gx = transpose(gy, inv_axis)
        return gx
def transpose(x, axis=None) :
    func = Transpose(axis)
    return func(x)
