# -*- coding: utf-8 -*-
"""
Symbolic Regression using Evolutionary Algorithms
- Pragyendra Bagediya

Finding the graph that fits the given data points
Mathematical operator encoded on a binary tree
"""

# importing all the required libraries
import numpy as np
import matplotlib.pyplot as plt
import time

# function to output children of a node
def find_children(node_index):
    """
    function: find the index position in tuple based on linear index
    args: index of the node in a linear array
    rtype: list: array of row, column index
    """
    return [(2*node_index), (2*node_index)+1]

# this function outputs the value of a node
def calc_node(tree, tree_node_def, node_index):
  """
  function: calculate the mathematical function pertaining to a given node
  args: 
        tree: the binary tree array itself
        tree_node_def: identity of the node, child or parent; leaf or node
        node_index: index of the node for calculating function
  rtype:
        string: function enclosed in parenthesis
  """
  if tree_node_def[node_index] == 'c': # if it is the last_child i.e. either a const/var it returns the same
      return "("+str(tree[node_index])+")"
  elif tree_node_def[node_index] == 'p': # if it is a parent it calculates the value
      nodes= find_children(node_index)
      # print(tree[node_index])
      if tree[node_index] in ['sin', 'cos']:
          ret_string = "(np." + tree[node_index] + "(" + calc_node(tree, tree_node_def, nodes[0]) + "))"
          return ret_string
      else:
          ret_string = "(" + calc_node(tree, tree_node_def, nodes[0]) + tree[node_index] + calc_node(tree, tree_node_def,nodes[1]) + ")"
          return ret_string
  else:
      return str(0)

# creating a tree by swaping operators and constants
def create_a_tree(tree_input, tree_node_def):
  """
  function: creating a tree for evalution of the function and loss
  args: 
        tree_input: input a tree to mutate
        tree_node_def: number of nodes in the tree
  rtype:
        list: newly created tree
  """
  # print("input to create a  tre", len(tree_input))
  tree = tree_input.copy()
  fun_avail = ['+', '-', '*', '/','sin', 'cos']
  def get_var():
    return np.random.uniform(-10,10)
  choose_one = ['q', 'cons','cons','cons']
  for i in range(len(tree_node_def)):
    if tree_node_def[i] == 'p':
      tree[i] = np.random.choice(fun_avail)
    if tree_node_def[i] == 'c':
      chosen_one = np.random.choice(choose_one)
      if chosen_one == 'q':
        tree[i] = 'q'
      else:
        tree[i] = get_var()
  return tree

#creating a completely random tree
def create_a_tree_random1(depth):
  """
  function: creating a completely new tree based on random functions
  args: depth: maximum depth of the tree
  rtype:
        List tree, List tree_node_def: the tree itself and the identity (child/parent) of the given nodes 
  """
  tree = [None]*((2**depth))
  tree_node_def = [None]*((2**depth))
  len_tree = len(tree)
  fun_avail = ['+', '-', '*', '/','sin', 'cos']
  def get_var():
    return np.random.uniform(-10,10)
  choose_one = ['p', 'p', 'p', 'c']
  choose_const = ['q', 'cons','cons','cons']
  def assign(tree_input, tree_node_def_input, idx):
    tree = tree_input.copy()
    tree_node_def = tree_node_def_input.copy()
    # print(idx)
    if idx > len(tree)-1:
      # print(idx)
      return 'Nope'
    else:
      assignment = np.random.choice(choose_one)
      # print(assignment)
      if assignment == 'c':
        choice = np.random.choice(choose_const)
        tree_node_def[idx] = 'c'
        if choice == 'q':
          tree[idx] = 'q'
        elif choice == 'cons':
          tree[idx] = get_var()
          # print(tree[idx])
      elif assignment == 'p' :
        tree_node_def[idx] = 'p'
        tree[idx] = np.random.choice(fun_avail)
        children = find_children(idx)
        # print(children)
        if assign(tree, tree_node_def, children[0]) == 'Nope':
          # print("hoho")
          choice = np.random.choice(choose_const)
          tree_node_def[idx] = 'c'
          if choice == 'q':
            tree[idx] = 'q'
            # print("hoho_q", idx)
          elif choice == 'cons':
            tree[idx] = get_var()
            # print("hoho_c", idx)
        else:
          tree, tree_node_def = assign(tree, tree_node_def, children[0])
          tree, tree_node_def = assign(tree, tree_node_def, children[1])
    return tree, tree_node_def

  tree, tree_node_def = assign(tree, tree_node_def, 1)
  return tree, tree_node_def

def calc_error(data, func_eqn):
  """
  function: calcuate the error based on the dataset and input function equation 
  args:
        data: the given data points
        func_eqn: the equation of function in string
  rtype:
        float: error val
  """
  x,y = data.T
  y_new = np.zeros_like(y)
  try:
    for i in range(len(x)):
      q = x[i]
      y_new[i] = eval(func_eqn)
      # print(np.shape(y_new))
    return np.sum(np.absolute(y_new-y))
  except:
    return float('inf')

def plot_data(data, func_eqn):
  """
  function: plot the data graph
  args:
        data: the data points
        func_eqn: 
  rtype:
        None: the plot is generated
  """
  x,y = data.T
  y_new = np.zeros_like(y)
  try:
      for i in range(len(x)):
          q = x[i]
          y_new[i] = eval(func_eqn)
          # print(np.shape(y_new))
  except:
      pass
  plt.scatter(x,y)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("Function Curve Visualisation")
  plt.plot(x,y_new,color='red')
  plt.show()

def plot_error_curve(record_error):
  """
  fucntion: plotting the error curve
  args: List record_error: list of all the errors
  ryte:
        None: the plot of the graph is generated 
  """
  fitness  = np.zeros_like(record_error)
  for i in range(len(record_error)):
    fitness[i] = (1/record_error[i])
  plt.plot(fitness)
  plt.xlabel('evaluations')
  plt.ylabel('Fitness')
  plt.title("Learning Curve")
  plt.show()

def swap(tree_in, tree_node_def_in, type_in):
  """
  function: swap a node with other branch
  args:
          tree_in : input tree structure
          tree_node_def_in: node definition of the input tree
          type_in: swapping with random or change
  """
  tree = tree_in.copy()
  tree_node_def = tree_node_def_in.copy()
  parents =[]
  child = []
  for i in range(len(tree_node_def)):
    if tree_node_def[i] == 'p':
      parents = np.append(parents, i)
    if tree_node_def[i] == 'c':
      child = np.append(child, i)

  def actual_swap(tree_in, tree_node_def_in, array, type_in):
    """"
    function: actual swap function used for recursion
    args:
          tree_in: input tree
          tree_node_def_in: definition of the nodes
          array: 
          type_in: random/simple
    rtype:
          List tree: Modified tree
    """
    if len(array) in [0,1]:
      return tree_in
    else:
      pass
    tree = tree_in.copy()
    a = int(np.random.choice(array))
    if type_in == 'random':
      b = int(np.random.choice(array))
    elif type_in == 'simple':
      try:
        b = array(int(np.where(array == a))+1)
      except:
        try:
          b = array(int(np.where(array == a) - 1))
        except:
          return tree
    temp = tree[a]
    tree[a] = tree[b]
    tree[b] = temp
    return tree

  tree = actual_swap(tree, tree_node_def_in, parents, type_in)
  tree = actual_swap(tree, tree_node_def_in, child, type_in)
  return tree

# function for making crossovers
def crossover(tree1_input, tree2_input, tree_def1_input, tree_def2_input):
  """
  function: creating crossover of two trees
  args:
          tree1_input: input tree 1
          tree2_input: input tree 2
          tree_def1_input: node definition of tree 1
          tree_def2_input: node definition of tree 2
  rtype:
        list tree1: tree swapped with a branch
        list tree_def1: definition of the nodes
  """
  tree1 = tree1_input.copy()
  tree2 = tree2_input.copy()
  tree_def1 = tree_def1_input.copy()
  tree_def2 = tree_def2_input.copy()

  def select_a_random_depth(tree1, tree2):
    def select_a_random_node(tree, depth):
      """
      function: selecting a random node of the given tree
      args:
              tree: input tree
              depth: max depth of the tree
      rtype:
              index of the randomly selected depth choice
      """
      return np.random.choice(nodes_avail(tree,depth))
    ac1 = actual_depth(tree1)
    ac2 = actual_depth(tree2)
    dep1 = np.random.choice(ac1)
    diff = ac2[-1] - (8-dep1)
    if diff <=0:
      lcl = 1
    else :
      lcl = diff
    ucl = ac2[-1]
    choice = np.arange(lcl, ucl+1,1)
    # print(choice)
    dep2 = np.random.choice(choice)
    # print(ac1)

    return select_a_random_node(tree1, dep1), select_a_random_node(tree2, dep2)

  # #step_1
  # #select a depth
  # dep1 = select_a_random_depth(tree1)
  # #step_2
  # #select a node from that depth
  # node_tree1 = select_a_random_node(tree1, dep1)
  # #step 3
  # #select a node from other tree less than depth - dep
  # node_tree2 = select_a_random_node(dep+1, depth)
  # #step 4 now join
  #   #remove the elents from tree1
  

  def remove_it_all(tree, tree_node_def, idx):
    """
    function: removing all the dta from the index of the tree
    args:
          tree: input tree
          tree_node_def: definition of the input node
          idx: index of the tree to remove the data from
    rtype:
          List: tree with the removed data
    """
    if tree_node_def[idx] == 'c':
      pass
    elif tree_node_def[idx] == 'p':
      children = find_children(idx)
      remove_it_all(tree, tree_node_def, children[0])
      remove_it_all(tree, tree_node_def, children[1])

    tree[idx] = None
    tree_node_def[idx] = None

  def add_em_up(tree1, tree2, tree_def1, tree_def2, node_tree1, node_tree2):
    """
    function: adding the two tree
    args:
          tree1: input tree 1
          tree2: input tree 2
          tree_def1: definition of intput tree 1
          tree_def2: definition of intput tree 2

          node_tree1: node to snap from tree 1
          node_tree2: node to swap from tree 2 to tree 1
    """
    if tree_def2[node_tree2] == 'c':
      tree_def1[node_tree1] = tree_def2[node_tree2]
      tree1[node_tree1] = tree2[node_tree2]
    elif tree_def2[node_tree2] == 'p':
      tree_def1[node_tree1] = tree_def2[node_tree2]
      tree1[node_tree1] = tree2[node_tree2]
      children1 = find_children(node_tree1)
      children2 = find_children(node_tree2)
      add_em_up(tree1, tree2, tree_def1, tree_def2, children1[0], children2[0])
      add_em_up(tree1, tree2, tree_def1, tree_def2, children1[1], children2[1])

  node_tree1, node_tree2 = select_a_random_depth(tree1, tree2)
  remove_it_all(tree1, tree_def1, node_tree1)
  add_em_up(tree1, tree2, tree_def1, tree_def2, node_tree1, node_tree2)
  return tree1, tree_def1

def actual_depth(tree_node_def):
  """
  function: calculate the depth of tree
  args:
        tree_node_def: input the definition of nodes of the tree
  rtype:
      numpy array: depth of the tree
  """
  last_element = 0
  depth = 0
  for i in range(len(tree_node_def)):
    if tree_node_def[i] != None:
      last_element = i
  for i in range(9):
    if last_element <= ((2**i)-1):
      depth = i
      return np.arange(1,depth+1,1)

def nodes_avail(tree_node_def, depth):
  """
  function: checks for the available nodes of a tree
  args:
        tree_node_def: the definintion of the nodes of the tree
        depth: the depth to check
  rtypes:
      List nodes: list of the available nodes to pick from
  """
  nodes = []
  lcl = ((2**(depth-1)))
  ucl = ((2**(depth)))
  pick = np.arange(lcl, ucl, 1)
  # print(pick)
  for i in pick:
    # print(i)
    # print(len(tree_node_def))
    if tree_node_def[i] != None:
      nodes.append(i)
  return nodes


if __name__ == "__main__":
  # Bool for plot animation
  show_animation=True
  # these will be used in real code
  data = np.loadtxt("data.txt", dtype = float, delimiter = ",")
  x,y = data.T

  #other required variables
  iter = 0
  record_error = []
  record_fun = []

  #parameters to end the code
  start = time.time()
  time_off = 240

  # defining characterstics of the tree
  depth = 8
  tree_size = (2**(depth))
  tree = [None]*tree_size
  tree_node_def= [None]*tree_size
  children_nodes= [3,9,10,11,17,33,65,128,129]
  parent_nodes = [1,2,4,5,8,16,32,64]
  for i in children_nodes:
    tree_node_def[i] = 'c'
  for i in parent_nodes:
    tree_node_def[i] = 'p'

  #running iteration1
  tree = create_a_tree(tree, tree_node_def)
  print(len(tree))
  func_eqn = calc_node(tree, tree_node_def, 1)
  error = calc_error(data, func_eqn)
  record_fun = func_eqn
  record_error.append(error)

  # hill climber Search
  # while time.time() - start < time_off:
  while iter<1000:
    iter +=1
    iter100 = error
    switch = 0
    neigh = []
    neigh_def = []
    neigh = tree.copy()
    neigh_def = tree_node_def.copy()
    neigh_error = []
    neigh_error = error

    new_tree, new_tree_def = create_a_tree_random1(8)
    # print("len of tree)", len(tree))
    neigh = np.vstack((neigh, new_tree))
    neigh_def = np.vstack((neigh_def, new_tree_def))
    neigh_error = np.append(neigh_error, calc_error(data, calc_node(new_tree, new_tree_def, 1)))

    # if switch == 0:
    for i in range(5):
      tree2, tree_def2 = create_a_tree_random1(8)
      # print(len(tree2))
      # print(len(neigh[0]))
      new_tree, new_tree_def = crossover(neigh[0], tree2, neigh_def[0], tree_def2)
      neigh = np.vstack((neigh, new_tree))
      neigh_def = np.vstack((neigh_def, new_tree_def))
      neigh_error = np.append(neigh_error, calc_error(data, calc_node(new_tree, new_tree_def, 1)))


    min_error = min(neigh_error)
    index = np.where(neigh_error == min_error)
    if min_error < record_error[-1]:
          record_error = np.append(record_error, min_error)
          record_fun = calc_node(neigh[index].flatten(), neigh_def[index].flatten(), 1)
          tree, tree_node_def = neigh[index], neigh_def[index]
          # switch = 0
    else:
        record_error = np.append(record_error, record_error[-1])
        curr_fun = calc_node(neigh[index].flatten(), neigh_def[index].flatten(), 1)
        pass

    if show_animation:
      plt.cla()

      #manual code
      xh,yh = data.T
      yh_new = np.zeros_like(yh)
      yh2_new = np.zeros_like(yh)
      try:
          for i in range(len(xh)):
              q = xh[i]
              yh2_new[i] = eval(curr_fun)
              yh_new[i] = eval(record_fun)
      except:
          pass

      plt.title("Function Curve Visualisation")
      plt.scatter(xh,yh, label="Given Data Points")
      plt.plot(xh,yh_new,color='green', label="Best Fit Curve Found")
      plt.plot(xh,yh2_new,color='red', label="Curve Being Evaluated")
      plt.xlim([-2.5,20])
      plt.ylim([-2.5,20])
      plt.grid(True)
      plt.legend()
      plt.pause(0.001)
    print("Record Error:", record_error[-1])
    del neigh, neigh_error

  # plt.figure()
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()

  fitness  = np.zeros_like(record_error)
  for i in range(len(record_error)):
    fitness[i] = (1/record_error[i])
    plt.plot(fitness)
    plt.xlabel('evaluations')
    plt.ylabel('Fitness')
    plt.title("Learning Curve")
    plt.show()
