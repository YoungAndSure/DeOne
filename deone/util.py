import os
import subprocess
import logging

from deone.core import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(pathname)s:%(lineno)d %(funcName)s() %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def _dot_var(v, verbose=False):
  dot_var = '{} [label="{}", color=orange, style=filled]\n'

  name = v.name if v.name != None else ""
  if verbose == True and v.data is not None:
    if v.name != None :
      name += ": "
    name += str(v.shape) + ' ' + str(v.dtype)

  return dot_var.format(id(v), name)

def _dot_func(f) :
  dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
  txt = dot_func.format(id(f), f.__class__.__name__)

  dot_edge = '{}->{}\n'
  for x in f.inputs :
    txt += dot_edge.format(id(x), id(f))
  txt += dot_edge.format(id(f), id(f.outputs()))

  return txt

def get_dot_graph(output, verbose=False) :
    if Config.enable_backward == False :
        return

    funcs = []
    seen_set = set()
    def add_func(f) :
        if f not in seen_set :
            funcs.append(f)
            seen_set.add(f)
        funcs.sort(key = lambda x : x.generation)
    add_func(output.creator)

    txt = ""
    while funcs :
        func = funcs.pop()
        txt += _dot_func(func)
        txt += _dot_var(func.outputs(), verbose)
        for input in func.inputs :
            txt += _dot_var(input, verbose)
            if (input.creator != None) :
                add_func(input.creator)
    txt = 'digraph g {\n' + txt + '}'
    return txt


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)
    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)