#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
from deone.core import *
from deone.layer import *

def get_conv_outsize(input_size, kernel_size, stride, pad) :
  return ((input_size + pad * 2) - kernel_size) // stride + 1

def pair(x) :
  if isinstance(x, int) :
    return (x, x)
  elif isinstance(x, tuple) :
    assert len(x) == 2
    return x
  raise ValueError

def im2col_array(img, kernel_size, stride, pad, to_matrix=True) :
  # TODO: copy from book
  N, C, H, W = img.shape
  KH, KW = pair(kernel_size)
  SH, SW = pair(stride)
  PH, PW = pair(pad)
  OH = get_conv_outsize(H, KH, SH, PH)
  OW = get_conv_outsize(W, KW, SW, PW)

  img = np.pad(img,
                ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                mode='constant', constant_values=(0,))
  col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

  for j in range(KH):
      j_lim = j + SH * OH
      for i in range(KW):
          i_lim = i + SW * OW
          col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

  if to_matrix:
      col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

  return col

def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
  # TODO: copy from book
  N, C, H, W = img_shape
  KH, KW = pair(kernel_size)
  SH, SW = pair(stride)
  PH, PW = pair(pad)
  OH = get_conv_outsize(H, KH, SH, PH)
  OW = get_conv_outsize(W, KW, SW, PW)

  if to_matrix:
    col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
  img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                  dtype=col.dtype)
  for j in range(KH):
      j_lim = j + SH * OH
      for i in range(KW):
          i_lim = i + SW * OW
          img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
  return img[:, :, PH:H + PH, PW:W + PW]

class Im2col(Function) :
  def __init__(self, kernel_size, stride, pad, to_matrix=True) :
    self.kernel_size = kernel_size
    self.stride = stride
    self.pad = pad
    self.to_matrix = to_matrix

  def forward(self, x) :
    self.input_shape = x.shape
    y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
    return y

  def backward(self, gy):
    gx = col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)
    return gx

def im2col(x, kernel_size, stride, pad, to_matrix=True) :
  f = Im2col(kernel_size, stride, pad, to_matrix)
  return f(x)

class Col2im(Function):
  #TODO: copy from book
  def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
      super().__init__()
      self.input_shape = input_shape
      self.kernel_size = kernel_size
      self.stride = stride
      self.pad = pad
      self.to_matrix = to_matrix

  def forward(self, x):
      y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                        self.pad, self.to_matrix)
      return y

  def backward(self, gy):
      gx = im2col(gy, self.kernel_size, self.stride, self.pad,
                  self.to_matrix)
      return gx
def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)

def conv2d_simple(x, Kernel, b=None, stride=1, pad=0) :
  N, C, H, W = x.shape
  OC, C, KH, KW = Kernel.shape
  SH, SW = pair(stride)
  PH, PW = pair(pad)
  OH = get_conv_outsize(H, KH, SH, PH)
  OW = get_conv_outsize(W, KW, SW, PW)

  col = im2col(x, (KH, KW), (SH, SW), (PH, PW))
  w = Kernel.reshape(OC, -1).transpose()
  t = linear(col, w, b)
  y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
  return y

class Conv2d(Layer) :
    def __init__(self, out_channel, kernel_size, stride=1, pad=0, no_bias=True, dtype=np.float32, in_channel=None) :
      super().__init__()
      self.out_channel = out_channel
      self.kernel_size = kernel_size
      self.stride = stride
      self.pad = pad
      self.in_channel = in_channel
      self.no_bias = no_bias
      self.dtype = dtype
      self.W = Parameter(None, 'W')
      if self.in_channel is not None :
         self.init_W()

      if self.no_bias :
         self.b = None
      else :
         self.b = Parameter(np.zeros(out_channel).astype(dtype), 'b')
    
    def init_W(self) :
      KH, KW = pair(self.kernel_size)
      C, OC = self.in_channel, self.out_channel
      scale = np.sqrt(1 / (C * KH * KW))
      self.W.data = np.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
    
    def forward(self, x) :
      if self.W.data is None :
        self.in_channel = x.shape[1]
        self.init_W()

      y = conv2d_simple(x, self.W, self.b, self.stride, self.pad)
      return y

def pooling_simple(x, kernel_size, stride=1, pad=0) :
  N, C, H, W = x.shape
  KH, KW = pair(kernel_size)
  SH, SW = pair(stride)
  PH, PW = pair(pad)
  OH = get_conv_outsize(H, KH, SH, PH)
  OW = get_conv_outsize(W, KW, SW, PW)

  col = im2col(x, (KH, KW), (SH, SW), (PH, PW))
  col = col.reshape(-1, KH * KW)
  y = col.max(axis=1)
  y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
  return y