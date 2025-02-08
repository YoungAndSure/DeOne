import deone.layer as L
import deone.util as util
from deone.function import *
from deone.function_simple import *
from deone.function_conv import *
from external.utils import *

class Model(L.Layer) :
  def plot(self, *input, file_name='model.png') :
    output = self.forward(*input)
    util.plot_dot_graph(output, verbose=True, to_file=file_name)

class MLP(Model) :
  def __init__(self, full_connect_layer_size, activation=sigmod_simple) :
    super().__init__()
    self.activation = activation
    self.layers=[]
    for (i, layer_size) in enumerate(full_connect_layer_size) :
      l = L.Linear(out_size = layer_size)
      # 必须得放到self._params里，遍历参数时候才能拿到，之前都是用self.xxx=x
      # 调用了=，默认调用了__setattr__把参数放进self._params里了
      # 这里是append到一个数组里了，所以必须手动调用setattr
      setattr(self, "layer_"+str(i), l)
      self.layers.append(l)

  def forward(self, x) :
    for l in self.layers[:-1] :
      x = l(x)
      x = self.activation(x)
    output_layer = self.layers[-1]
    y = output_layer(x)
    return y

class VGG16(Model):
  WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/' \
                 'releases/download/v0.1/vgg16.npz'

  def __init__(self, pretrained=False):
    super().__init__()
    self.conv1_1 = Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv1_2 = Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv2_1 = Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv2_2 = Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv3_1 = Conv2d(256, kernel_size=3, stride=1, pad=1)
    self.conv3_2 = Conv2d(256, kernel_size=3, stride=1, pad=1)
    self.conv3_3 = Conv2d(256, kernel_size=3, stride=1, pad=1)
    self.conv4_1 = Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.conv4_2 = Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.conv4_3 = Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.conv5_1 = Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.conv5_2 = Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.conv5_3 = Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.fc6 = L.Linear(4096)
    self.fc7 = L.Linear(4096)
    self.fc8 = L.Linear(1000)

    if pretrained :
      weight_file = get_file(VGG16.WEIGHTS_PATH)
      self.load_weights(weight_file)

  def forward(self, x):
    x = relu(self.conv1_1(x))
    x = relu(self.conv1_2(x))
    x = pooling_simple(x, 2, 2)
    x = relu(self.conv2_1(x))
    x = relu(self.conv2_2(x))
    x = pooling_simple(x, 2, 2)
    x = relu(self.conv3_1(x))
    x = relu(self.conv3_2(x))
    x = relu(self.conv3_3(x))
    x = pooling_simple(x, 2, 2)
    x = relu(self.conv4_1(x))
    x = relu(self.conv4_2(x))
    x = relu(self.conv4_3(x))
    x = pooling_simple(x, 2, 2)
    x = relu(self.conv5_1(x))
    x = relu(self.conv5_2(x))
    x = relu(self.conv5_3(x))
    x = pooling_simple(x, 2, 2)
    x = reshape(x, (x.shape[0], -1))
    x = dropout(relu(self.fc6(x)))
    x = dropout(relu(self.fc7(x)))
    x = self.fc8(x)
    return x
  
  @staticmethod
  def preprocess(img, size=(224,224), dtype=np.float32) :
    image = img.convert('RGB')
    if size :
      image = image.resize(size)
    image = np.asarray(image, dtype)
    # ::-1代表逆序操作，比如本来是'RGB',::-1之后是'BGR'
    image = image[:,:,::-1]
    # 这段不懂，可能是为了限制数据大小防止溢出？
    image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
    # 把通道放前面了，VGG16里是这么实现的
    image = image.transpose((2, 0, 1))
    return image

class SimpleRNN(Model) :
  def __init__(self, hidden_size, out_size) :
    super().__init__()
    self.rnn = RNN(hidden_size=hidden_size)
    self.fc = Linear(out_size=out_size)

  def reset_status(self) :
    self.rnn.reset_status()

  def forward(self, x) :
    y = self.rnn(x)
    y = self.fc(y)
    return y

class BetterRNN(Model) :
  def __init__(self, hidden_size, out_size) :
    super().__init__()
    self.rnn = LSTM(hidden_size=hidden_size)
    self.fc = Linear(out_size=out_size)

  def reset_status(self) :
    self.rnn.reset_status()

  def forward(self, x) :
    y = self.rnn(x)
    y = self.fc(y)
    return y

class Cbow(Model) :
  def __init__(self, hidden_size, vocab_size) :
    super().__init__()

    #self.l0 = Linear(out_size=hidden_size, has_bias=False)
    self.l0 = Embedding(in_size=vocab_size , out_size=hidden_size)
    self.l1 = Linear(out_size=vocab_size, has_bias=False)

  def forward(self, xs) :
    y = None
    for x in xs :
      y = self.l0(x) if y is None else y + self.l0(x)
    y /= len(xs)
    # l0 用Linear输出的y是二维的，用Embedding输出是一维的,需要加一个轴
    y = y[np.newaxis, :]
    y = self.l1(y)
    return y