#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import math

from deone import *

class AddTest(unittest.TestCase) :
  def test_backward(self) :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    outputs = add(x0, x1)
    outputs.backward()
    expected = np.array(1.0)
    self.assertEqual(x0.grad.data, expected)
    self.assertEqual(x1.grad.data, expected)
  
  def test_second_backward(self) :
    x = Variable(np.array(2.0))
    y = x ** 4 - 2 * x ** 2
    y.backward(create_graph=True)
    self.assertEqual(x.grad.data, 24.0)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    # 注意，二阶导数不在y.grad，也不在gx.grad，而是传回了x.grad
    # 二次反向传播先是传回了y.grad，然后又继续往回传到了x.grad
    self.assertEqual(x.grad.data, 44.0)

  def test_cos(self) :
    x = Variable(np.array(np.pi / 4))
    y = cos(x)
    self.assertTrue(np.allclose(y.data, 0.70710678))
    y.backward()
    self.assertTrue(np.allclose(x.grad.data, -0.70710678))
  
  def test_tan(self) :
    x = Variable(np.array(np.pi))
    y = tanh(x)
    self.assertTrue(np.allclose(y.data, 0.996272))
    y.backward()
    self.assertTrue(np.allclose(x.grad.data, 0.00744195))
  
  def test_tensor_add(self) :
    x = Variable(np.array([1.0, 2.0, 3.0]))
    y = Variable(np.array([3.0, 2.0, 1.0]))
    z = x + y
    self.assertTrue(np.array_equal(z.data, [4.0, 4.0, 4.0]))
    z.backward()
    self.assertTrue(np.array_equal(x.grad.data, [1.0, 1.0, 1.0]))
  
  def test_reshape(self) :
    t = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
    x = Variable(t)
    y = reshape(x, (6, ))
    self.assertTrue(np.array_equal(y.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    z = reshape(t, (2, 3))
    self.assertTrue(np.array_equal(z.data, t))
    z.backward()
  
  def test_variable_reshape(self) :
    x = Variable(np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]))
    y = x.reshape((6, ))
    self.assertTrue(np.array_equal(y.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

    x = Variable(np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]))
    y = x.reshape(6, )
    self.assertTrue(np.array_equal(y.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

  def test_transpose(self) :
    x = Variable(np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]))
    y = transpose(x)
    self.assertTrue(np.array_equal(y.data, [[1.0,4.0], [2.0, 5.0], [3.0, 6.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]))

    x.cleargrad()
    y = x.transpose()
    self.assertTrue(np.array_equal(y.data, [[1.0,4.0], [2.0, 5.0], [3.0, 6.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]))

    x.cleargrad()
    y = x.T
    self.assertTrue(np.array_equal(y.data, [[1.0,4.0], [2.0, 5.0], [3.0, 6.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]))

    x = Variable(np.array([1.0, 2.0, 3.0]))
    y = x.T
    # 输入是一维的，输出就也是一维的，如果想转置后成一列，输入要写成二维的
    #self.assertTrue(np.array_equal(y.data, [[1.0], [2.0], [3.0]]))
    self.assertTrue(np.array_equal(y.data, [1.0, 2.0, 3.0]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [1.0, 1.0, 1.0]))

  def test_sum(self) :
    x = Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    y = sum(x)
    self.assertTrue(np.array_equal(y.data, 21.0))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = sum(x)
    self.assertTrue(np.array_equal(y.data, 21.0))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = sum(x, keepdims=True)
    self.assertTrue(np.array_equal(y.data, [[21.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

  def test_broadcast2_sum2(self) :
    x = Variable(np.array([1.0, 2.0, 3.0]))
    y = broadcast_to(x, (2, 3))
    self.assertTrue(np.array_equal(y.data, [[1.0,2.0,3.0],[1.0,2.0,3.0]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [2.0, 2.0, 2.0]))

    x.cleargrad()
    z = sum_to(y, (1,3))
    self.assertTrue(np.array_equal(z.data, [[2.0, 4.0, 6.0]]))
    z.backward(retain_grad=True)
    self.assertTrue(np.array_equal(y.grad.data, [[1.0,1.0,1.0],[1.0,1.0,1.0]]))
  
  def test_support_broadcast2_add(self) :
    x = Variable(np.array([1.0, 2.0, 3.0]))
    y = Variable(np.array([1.0]))
    z = x + y
    self.assertTrue(np.array_equal(z.data, [2.0, 3.0, 4.0]))
    z.backward()
    self.assertTrue(np.array_equal(x.grad.data, [1.0, 1.0, 1.0]))
    self.assertTrue(np.array_equal(y.grad.data, [3.0]))
  
  def test_matmul(self) :
    x = Variable(np.array([[1.0, 2.0, 3.0]]))
    W = Variable(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    y = matmul(x, W)
    self.assertTrue(np.array_equal(y.data, np.array([[22, 28]])))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, np.array([[3.0, 7.0, 11.0]])))

  def test_mean_square_error(self) :
    x0 = Variable(np.array([4,5,6]))
    x1 = Variable(np.array([7,8,9]))
    y = mean_square_error(x0, x1)
    self.assertTrue(np.array_equal(y.data, np.array([3.0, 3.0, 3.0])))
    y.backward()
    self.assertTrue(np.array_equal(x0.grad.data, [-2.0, -2.0, -2.0]))

  def test_linear(self) :
    x0 = Variable(np.ones((100, 1)))
    W0 = Variable(np.ones((1, 1)))
    b0 = Variable(np.ones((1,)))
    y0 = linear(x0, W0, b0)
    self.assertTrue(np.array_equal(y0.data, np.ones((100, 1)) + 1))
    y0.backward()

    x1 = Variable(np.ones((100, 1)))
    W1 = Variable(np.ones((1, 1)))
    b1 = Variable(np.ones((1,)))
    y1 = linear_simple(x1, W1, b1)
    self.assertTrue(np.array_equal(y1.data, np.ones((100, 1)) + 1))
    y1.backward()

    self.assertTrue(np.array_equal(W0.grad.data, W1.grad.data))
    self.assertTrue(np.array_equal(b0.grad.data, b1.grad.data))
    self.assertTrue(np.array_equal(x0.grad.data, x1.grad.data))

  def test_sigmod_simple(self) :
    x = Variable(np.array([0, 0, 0]))
    y = sigmod_simple(x)
    self.assertTrue(np.array_equal(y.data, [0.5, 0.5, 0.5]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [0.25, 0.25, 0.25]))
  
  def test_linear_layer(self) :
    x0 = Variable(np.ones((100, 1)))
    layer = Linear(in_size = 1, out_size = 1, has_bias = True, dtype = np.float32)
    y0 = layer(x0)
    self.assertTrue(isinstance(y0, Variable))
    params_dict = {}
    layer._flatten_params(params_dict)
    self.assertTrue('W' in params_dict and isinstance(params_dict['W'], Parameter))
    self.assertTrue('b' in params_dict and isinstance(params_dict['b'], Parameter))

  def test_linear_layer2(self) :
    np.random.seed(0)
    x = np.random.rand(100, 1)
    label_y = np.sin(2 * np.pi * x)

    iters = 10000
    lr = 0.2

    I,H,O = 1, 10, 1
    def use_layer() :
      l1 = Linear(in_size = I, out_size = H)
      l2 = Linear(in_size = H, out_size = O)

      def predict(x) :
        y = l1(x)
        y = sigmod_simple(y)
        y = l2(y)
        return y

      for i in range(iters) :
        predict_y = predict(x)
        loss = mean_square_error(predict_y, label_y)
        l1.cleargrad()
        l2.cleargrad()
        loss.backward()
        for l in (l1, l2) :
          for param in l.params() :
            param.data -= lr * param.grad.data

      y = predict(x)
      loss = mean_square_error(y, label_y)
      return loss.data

    def use_function() :
      W1 = Variable(np.random.rand(I, H))
      b1 = Variable(np.zeros(H))
      W2 = Variable(np.random.rand(H, O))
      b2 = Variable(np.zeros(O))

      def predict(x) :
        y = linear(x, W1, b1)
        y = sigmod_simple(y)
        y = linear(y, W2, b2)
        return y

      for i in range(iters) :
        predict_y = predict(x)
        loss = mean_square_error(predict_y, label_y)

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()

        loss.backward()

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data

      y = predict(x)
      loss = mean_square_error(y, label_y)
      return loss.data
    
    use_layer_loss = use_layer()
    use_function_loss = use_function()
    # DONE: 不过原因找到了。1. 是__setattr__，不是__set_attr__ 2. lr和iter太低，两个方式都没有收敛，所以和label差距大
    # 3. 两个方法的迭代速度不同，layer要更快，所以一样的iter，两边的结果不同
    # 4. 所以，不能比较预测结果，应该比较loss在一个合理范围内，就说明运行正常
    # 5. 还有，in_size和out_size，需要显示指定，不然默认初始化到错误的入参了
    loss_threshold = 1e-3
    self.assertTrue((np.abs(use_layer_loss) < loss_threshold).all())
    self.assertTrue((np.abs(use_function_loss) < loss_threshold).all())

  def test_mlp(self) :
    # 提前解锁隐藏关卡了，这里不能层数太多，会overflow，结果全是nan
    layer_size = (10, 1)
    mlp_model = MLP(full_connect_layer_size = layer_size)

    np.random.seed(0)
    x = np.random.rand(100, 1)
    label_y = np.sin(2 * np.pi * x)

    lr = 0.2
    iters = 10000

    for i in range(iters) :
      py = mlp_model.forward(x)
      loss = mean_square_error(py, label_y)
      mlp_model.cleargrad()
      loss.backward()

      for param in mlp_model.params() :
        param.data -= lr * param.grad.data

    py = mlp_model.forward(x)
    loss = mean_square_error(py, label_y)
    loss_threshold = 1e-3
    self.assertTrue((np.abs(loss.data) < loss_threshold).all())

  def test_get_item(self) :
    x = Variable(np.array([[1,2,3],[4,5,6],[7,8,9]]))
    y = get_item(x, 1)
    # 数组表示切片，tuple表示取元素
    y.backward()
    self.assertTrue(np.array_equal(y.data, [4,5,6]))
    self.assertTrue(np.array_equal(x.grad.data, [[0,0,0],[1,1,1],[0,0,0]]))

    x.cleargrad()
    z = get_item(x, [0,1,1])
    z.backward()
    self.assertTrue(np.array_equal(z.data, [[1,2,3],[4,5,6],[4,5,6]]))
    self.assertTrue(np.array_equal(x.grad.data, [[1,1,1],[2,2,2],[0,0,0]]))

    x.cleargrad()
    # 等同于调用get_item(x, [0,1,1])
    z = x[[0,1,1]]
    z.backward()
    self.assertTrue(np.array_equal(z.data, [[1,2,3],[4,5,6],[4,5,6]]))
    self.assertTrue(np.array_equal(x.grad.data, [[1,1,1],[2,2,2],[0,0,0]]))

  def test_softmax1d(self) :
    x = Variable(np.array([0.1, 0.2, 0.3]))
    y = softmax1d(x)
    self.assertTrue(np.allclose(y.data, [0.30060961, 0.33222499, 0.3671654]))
    y.backward()
    self.assertTrue(np.allclose(x.grad.data, [0, 0, 0]))

  def test_softmax(self) :
    x = Variable(np.array([[-0.615, -0.427, 0.317],[-0.763, -0.249, 0.185],[-0.520, -0.962, 0.578],[-0.942, -0.503, 0.175]]))
    y = softmax(x)
    self.assertTrue(np.allclose(y.data, [[0.210, 0.254, 0.535],[0.190, 0.318, 0.491],[0.215, 0.138, 0.646],[0.178, 0.276, 0.545]], atol=0.001))
    y.backward()
    self.assertTrue(np.allclose(x.grad.data, [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]))

  def test_clip(self) :
    x = Variable(np.array(10.0))
    y = clip(x, 100, 200)
    y.backward()
    self.assertEqual(y.data, 100)
    self.assertEqual(x.grad.data, 0)

    x = Variable(np.array(150.0))
    y = clip(x, 100, 200)
    y.backward()
    self.assertEqual(y.data, 150)
    self.assertEqual(x.grad.data, 1)

    x = Variable(np.array([10.0, 150.0]))
    y = clip(x, 100, 200)
    y.backward()
    self.assertTrue(np.array_equal(y.data, [100, 150]))
    self.assertTrue(np.array_equal(x.grad.data, [0, 1]))

  def test_log(self) :
    x = Variable(np.array([1.0, 2.0]))
    y = log(x)
    self.assertTrue(np.allclose(y.data, [0, 0.693147]))
    y.backward()
    self.assertTrue(np.allclose(x.grad.data, [1.0, 0.5]))

  def test_softmax_cross_entropy_simple(self) :
    x = Variable(np.array([[0.1, 0.2]]))
    t = Variable(np.array([[0, 1]]))
    y = softmax_cross_entropy_simple(x, t)
    self.assertTrue(np.allclose(y.data, 1.38879332))
    y.backward()
    self.assertTrue(np.allclose(x.grad.data, [[-0.04995837, 0.04995837]]))

  def test_spiral(self) :
    dataset = Spiral()
    train, label = dataset[0]
    self.assertTrue(np.allclose(train, [-0.13981389, -0.00721657]))
    self.assertEqual(label, 1)
    self.assertEqual(len(dataset), 300)
  
  def test_dataset_transforms(self) :
    def f(x) :
      return x / 2
    def lf(x) :
      return x * 2

    dataset = Spiral(transforms=f, label_tansforms=lf)
    train, label = dataset[0]
    self.assertTrue(np.allclose(train, [-0.13981389 / 2, -0.00721657 / 2]))
    self.assertEqual(label, 1 * 2)
    self.assertEqual(len(dataset), 300)

  def test_accuracy(self) :
    y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
    t = np.array([1, 2, 0])
    acc = accuracy(y, t)
    self.assertTrue(np.allclose(acc.data, 0.66666))

  def test_dropout(self) :
    x = Variable(np.ones(10))
    y = dropout(x)
    self.assertEqual(x.shape, y.shape)
    with test_mode() :
      x = Variable(np.ones(10))
      y = dropout(x)
      self.assertTrue(np.array_equal(x.data, y.data))

  def test_get_conv_outsize(self) :
    H, W = 4, 4 # input_shape
    KH, KW = 3, 3 # kernel_shape
    SH, SW = 1, 1 # stride(垂直方向的步幅，水平方向的步幅)
    PH, PW = 1, 1 # padding(垂直方向的填充，水平方向的填充)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    self.assertEqual(OH, 4)
    self.assertEqual(OW, 4)

  def test_im2col(self) :
    x = Variable(np.random.rand(1, 3, 7, 7))
    y = im2col(x, kernel_size=5, stride=1, pad=0)
    self.assertTrue(np.array_equal(y.shape, (9, 75)))
    x = Variable(np.random.rand(10, 3, 7, 7))
    y = im2col(x, kernel_size=5, stride=1, pad=0)
    self.assertTrue(np.array_equal(y.shape, (90, 75)))
    y.backward()

  def test_conv2d_simple(self) :
    x = Variable(np.random.rand(10, 3, 7, 7))
    W = Variable(np.random.rand(2, 3, 5, 5))
    y = conv2d_simple(x, W)
    self.assertTrue(np.array_equal(y.shape, (10, 2, 3, 3)))
    y.backward()
  
  def test_max(self) :
    x = Variable(np.array([1,2,3]))
    y = x.max()
    self.assertTrue(np.array_equal(y.data, [3]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [0, 0, 1]))

    x = Variable(np.array([[1,2,3],[4,5,6]]))
    y = x.max(axis=1)
    self.assertTrue(np.array_equal(y.data, [[3],[6]]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [[0,0,1], [0,0,1]]))

  def test_pooling_simple(self) :
    x = Variable(np.random.rand(10, 3, 7, 7))
    y = pooling_simple(x, kernel_size=5, stride=1, pad=0)
    self.assertTrue(np.array_equal(y.shape, (10, 3, 3, 3)))

  def test_conv2d_layer(self) :
    x = Variable(np.random.rand(10, 3, 7, 7))
    conv2d = Conv2d(out_channel=3, kernel_size=5, stride=1, pad=0)
    y = conv2d(x)
    self.assertTrue(np.array_equal(y.shape, (10, 3, 3, 3)))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.shape, (10, 3, 7, 7)))

  def test_relu(self) :
    x = Variable(np.array([-1.0, 1.0]))
    y = relu(x)
    self.assertTrue(np.array_equal(y.data, [0, 1.0]))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.data, [0, 1.0]))

  def test_vgg16_model(self) :
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if cpu_count <= 2 :
      print("这么点核数跑不起来vgg16,加钱吧哥们儿")
      return
    model = VGG16(pretrained=True)
    x = Variable(np.random.rand(1, 3, 224, 224))
    y = model(x)
    self.assertTrue(np.array_equal(y.shape, (1, 1000)))
    y.backward()
    self.assertTrue(np.array_equal(x.grad.shape, (1, 3, 224, 224)))

  def test_rnn(self) :
    x0 = Variable(np.random.rand(8,16))
    rnn = RNN(10)
    y = rnn(x0)
    self.assertTrue(np.array_equal(y.shape, (8,10)))

    x1 = Variable(np.random.rand(8,16))
    z = rnn(x1)
    self.assertTrue(np.array_equal(z.shape, (8,10)))
    self.assertTrue(np.array_equal(rnn.h.shape, (8,10)))
    rnn.reset_status()
    self.assertEqual(rnn.h, None)
  
  def test_unchain(self) :
    a = Variable(np.array(1.0))
    b = Variable(np.array(2.0))
    c = a + b
    d = c + 3.0
    self.assertTrue(d.creator is not None)
    d.unchain_backward()
    self.assertTrue(d.creator is not None)
    self.assertTrue(a.creator is None)
    self.assertTrue(b.creator is None)
    self.assertTrue(c.creator is None)

  def test_seq_dataloader(self) :
    dataset = SeqDataSet(max_len=100)
    data_loader = SeqDataLoader(dataset, batch_size=10)
    for x, t in data_loader :
      self.assertTrue(np.array_equal(x, [0,9,18,27,36,45,54,63,72,81]))
      self.assertTrue(np.array_equal(t, [1,10,19,28,37,46,55,64,73,82]))
      break

  def test_embedding_layer(self) :
    layer = Embedding(in_size=10, out_size=3)
    idx = np.array([0, 0, 1])
    y = layer(idx)
    self.assertEqual(y.data.shape, (3, 3))
    y.backward()
    self.assertTrue(np.array_equal(layer.W.grad.data[0], [2,2,2]))
    self.assertTrue(np.array_equal(layer.W.grad.data[1], [1,1,1]))

  def test_embedding_dot_layer(self) :
    h = np.array([[1, 2, 3], [4, 5, 6]])
    # 反过来的
    layer = EmbeddingDot(in_size=3, out_size=10)
    idx = np.array([0, 1])
    y = layer(h, idx)
    self.assertEqual(y.data.shape, (2, 1))
    y.backward()

  def test_time_rnn(self) :
    # 3 batch, 4 time, 5 dense
    xs = Variable(np.random.rand(3, 4, 5))
    BATCH, TIME, DENSE = xs.shape
    hidden_size = 10
    time_rnn = TimeRNN(TIME, hidden_size, stateful=True)
    hs = time_rnn(xs)
    self.assertEqual(hs.shape, (BATCH, hidden_size))
    hs.backward()
    self.assertEqual(xs.grad.shape, (BATCH, TIME, DENSE))

unittest.main()
