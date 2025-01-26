import numpy as np
from external.datasets import get_spiral
from external.utils import get_file

class Dataset:
  def __init__(self, train = True, transforms=None, label_tansforms=None) :
    self.train = train
    self.data = None
    self.label = None
    self.transforms = transforms
    self.label_transforms = label_tansforms
    self.prepare()

  def __getitem__(self, index) :
    data = self.transforms(self.data[index]) if self.transforms != None else self.data[index]
    if self.label is None :
        return data, None
    label = self.label_transforms(self.label[index]) if self.label_transforms != None else self.label[index]
    return data, label
  
  def __len__(self) :
      return len(self.data)

  def prepare(self) :
      pass

class Spiral(Dataset) :
  def prepare(self) :
     self.data, self.label = get_spiral(train=True)

class BigData(Dataset) :
  def __getitem__(self, index) :
     x = np.load('data/{}.npy'.format(index))
     t = np.load('label/{}.npy'.format(index))
     return x,t

  def __len__(self) :
    # TODO:why?
    return 1000000

class ImageNet(Dataset) :
  def __init__(self) :
    NotImplemented
  @staticmethod
  def labels() :
    url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
    path = get_file(url)
    with open(path, 'r') as f :
      # 这个文件里是个字典，这里用eval直接把字典转成了对象
      labels = eval(f.read())
    return labels

class SeqDataSet(Dataset) :
  def __init__(self, max_len=100, train = True, transforms=None, label_tansforms=None) :
    self.max_len = max_len
    super().__init__(train=train, transforms=transforms, label_tansforms=label_tansforms)

  def prepare(self):
    data = range(0, self.max_len)
    self.data = data[0:-1]
    self.label = data[1:]