
from deone.core import *
from deone.config import *
from deone.function import *
from deone.function_simple import *
from deone.function_conv import *
from deone.layer import *
from deone.model import *
from deone.optimizer import *
from deone.data_set import *
from deone.data_loader import *
from deone.util import *

def setup_variable() :
  Variable.__add__ = add
  Variable.__radd__ = add

  Variable.__mul__ = mul
  Variable.__rmul__ = mul

  Variable.__sub__ = sub
  Variable.__rsub__ = rsub

  Variable.__neg__ = neg

  Variable.__truediv__ = div
  Variable.__rtruediv__ = rdiv

  Variable.__pow__ = pow

  Variable.__array_priority__ = 200

  Variable.__getitem__ = get_item

  Variable.max = vmax

setup_variable()