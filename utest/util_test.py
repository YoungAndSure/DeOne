#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import math

from deone import *
from deone.util import _dot_func,_dot_var

class UtilTest(unittest.TestCase) :
  def test_dot_var(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(2.0))
    y = x0 + x1
    print(_dot_var(y))
    print(_dot_func(y.creator))
  
  def test_get_dot_graph(self) :
    x0 = Variable(np.array(1.0), "x0")
    x1 = Variable(np.array(2.0), "x1")
    y = x0 + x1
    y.name = "y"
    plot_dot_graph(y, verbose=True, to_file="add.png")
  
  def test_get_dot_graph2(self) :
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein(x, y)
    plot_dot_graph(z, verbose=True, to_file="goldstein.png")
  
  def test_graph_taylor_sin(self) :
    x0 = Variable(np.array(np.pi / 4))
    y0 = taylor_sin(x0)
    x1 = Variable(np.array(np.pi / 4))
    y1 = sin(x1)
    self.assertTrue(np.allclose(y0.data, y1.data))
    plot_dot_graph(y0, verbose=True, to_file="taylor_sin.png")
    plot_dot_graph(y1, verbose=True, to_file="sin.png")

unittest.main()