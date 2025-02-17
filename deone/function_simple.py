#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
from deone.core import *
from deone.function import *

def sphere(x, y) :
  return x**2 + y**2

def matyas(x, y) :
  z = 0.26 * (x**2 + y**2) - 0.48*x*y
  return z

def goldstein(x, y) :
  z = (1 + (x + y + 1) ** 2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) *\
      (30 + (2*x - 3*y)**2*(18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
  return z

def my_factorial(x) :
  result = 1.0
  i = Variable(np.array(1.0))
  while i.data <= x.data :
    result *= i
    i = i + 1
  return result

def taylor_sin(x, threshold=0.0001) :
  y = 0
  for i in range(100000) :
    c = (-1) ** i / math.factorial(2 * i + 1)
    t = c * x ** (2 * i + 1)
    y = y + t
    if (abs(t.data) < threshold) :
      break
  return y

def rosenbrock(x0, x1) :
  return 100 * (x1 - (x0 ** 2)) ** 2 + (x0 - 1) ** 2

def mean_square_error_simple(x0, x1) :
  diff = x0 - x1
  diff = diff ** 2
  return diff / len(diff.data)

def linear_simple(x, W, b) :
  t = matmul(x, W)
  if b == None :
      return t
  y = t + b
  t.data = None
  return y

def sigmod_simple(x) :
  return 1.0 / (1.0 + exp(-1.0 * x))

def softmax1d(x) :
  y = exp(x)
  y_sum = sum(y)
  return y / y_sum

def softmax(x, axis=1) :
  x = as_variable(x)
  # NOTE: 这里很容易溢出，输入的x稍微大一点，y_sum就inf了，进而导致y / y_sum是nan
  y = exp(x)
  y_sum = sum(y, axis=axis, keepdims=True)
  return y / y_sum

def softmax_cross_entropy_simple(x, t) :
  x = as_variable(x)
  t = as_variable(t)
  N = x.shape[0]

  p = softmax(x)
  p = clip(p, 1e-15, 1.0)
  p = log(p)
  # np的功能，N是行数，np.arange(N)就是逐行的意思,
  # t.data是提供的切片slices，合起来也就是逐行按照t.data切片
  p = p[np.arange(N), t.data]
  y = -1 * sum(p) / N
  return y

def sigmoid_cross_entropy_simple(x, t) :
  x = as_variable(x)
  t = as_variable(t)

  p = sigmod_simple(x)
  p = clip(p, 1e-15, 1.0)
  p = log(p)
  np = log(1 - p)
  y = t * p + (1 - t) * np
  return y

def accuracy(y, t) :
  y, t = as_variable(y), as_variable(t)
  pred = y.data.argmax(axis=1).reshape(t.shape)
  result = (pred == t.data)
  acc = result.mean()
  return Variable(as_array(acc))

def dropout(x, dropout_ratio=0.5) :
  x = as_variable(x)
  if Config.train :
    # rand是0，1均匀分布，randn是正态分布
    # x.shape是个元组，传入rand需要解包
    rx = np.random.rand(*x.shape)
    mask = rx > dropout_ratio
    scale = np.array(1 / (1 - dropout_ratio)).astype(x.dtype)
    x *= mask
    x *= scale
    return x
  else :
    return x