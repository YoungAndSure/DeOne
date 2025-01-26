import contextlib

class Config :
  # 是否开启反向传播
  enable_backward = True
  close_random = False
  train = True

@contextlib.contextmanager
def using_config(key, value) :
  #print("set " + key, value)
  old_value = getattr(Config, key)
  setattr(Config, key, value)
  try :
    yield
  finally :
    setattr(Config, key, old_value)

def predict() :
  return using_config("enable_backward", False)

def test_mode() :
  return using_config("train", False)