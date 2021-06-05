import math
from math import sqrt
import numbers
# import numpy as np

def zeroes(height, width):
  """
  Creates a matrix of zeroes.
  """
  g = [[0.0 for _ in range(width)] for __ in range(height)]
  return Matrix(g)

# Not sure why this function is needed?
def identity(n):
  """
  Creates a n x n identity matrix.
  """
  I = zeroes(n, n)
  for i in range(n):
      I.g[i][i] = 1.0
  return I

class Matrix(object):
  # Constructor
  # This sets up the object and its state
  def __init__(self, grid):
      self.g = grid
      self.h = len(grid)
      self.w = len(grid[0])

  # Primary matrix math methods
  #############################

  def determinant(self):
    """
    Calculates the determinant of a 1x1 or 2x2 matrix.
    """
    if not self.is_square():
      raise ValueError("Cannot calculate determinant of non-square matrix.")
    if self.h > 2:
      raise NotImplementedError("Calculating determinant not implemented for matrices largerer than 2x2.")
    
    # TODO - your code here
    # For a 1x1 matrix, just return the value
    if self.h == 1:
      return self.g[0]
    elif self.h == 2:
      return  (self.g[0][0] * self.g[1][1]) -  (self.g[0][1] * self.g[1][0])
    # # Alternate Solution to support larger matrices
    # # Create a Numpy matrix and calculate the determinant
    #   np_array = np.array(self.g)
    #   return np.linalg.det(np_array)

    else:
      raise RuntimeError("Something else went wrong") 
  
  def trace(self):
    """
    Calculates the trace of a matrix (sum of diagonal entries).
    """
    if not self.is_square():
        raise ValueError("Cannot calculate the trace of a non-square matrix.")

    # TODO - your code here
    t = 0
    for i in range(self.w):
        t += self[i][i]
    return t

  def inverse(self):
    """
    Calculates the inverse of a 1x1 or 2x2 Matrix.
    """
    if not self.is_square():
      raise ValueError("Non-square Matrix does not have an inverse.")
    if self.h > 2:
      raise NotImplementedError("inversion not implemented for matrices larger than 2x2.")

    # TODO - your code here    
    inverse = []
    if self.h == 1:
      inverse.append([1 / self.g[0][0]])
      return Matrix(inverse)
    elif self.h == 2:
      # Calculate a determinant matrix
      det = self.determinant()
      # If the matrix is 2x2, check that the matrix is invertible
      if self.g[0][0] * self.g[1][1] == self.g[0][1] * self.g[1][0]:
        raise ValueError('The matrix is not invertible.')
      else:
        # Start with a zero matrix and overwrite
        inv = zeroes(self.h, self.w)
        a = self.g[0][0]
        b = self.g[0][1]
        c = self.g[1][0]
        d = self.g[1][1]
      # Populate inverse grid with inverse value
        inv[0][0] = (1/det) * d
        inv[0][1] = (1/det) * (-1 * b)
        inv[1][0] = (1/det) * (-1 * c)
        inv[1][1] = (1/det) * a
      return inv
    else:
      raise RuntimeError("Something else went wrong") 

  def T(self):
    """
    Returns a transposed copy of this Matrix.
    """
    # TODO - your code here
    # Start with a zero matrix and overwrite
    t = zeroes(self.h, self.w)
    # Loop through columns on outside loop
    for c in range(self.w):
      # Loop through columns on inner loop
      for r in range(self.h):
        # Column values will be filled by what were each row before
        t[r][c] = self[c][r]
    return t
      
  def is_square(self):
    return self.h == self.w

  # Begin Operator Overloading
  ############################
  def __getitem__(self,idx):
    """
    Defines the behavior of using square brackets [] on instances
    of this class.

    Example:

    > my_matrix = Matrix([ [1, 2], [3, 4] ])
    > my_matrix[0]
      [1, 2]

    > my_matrix[0][0]
      1
    """
    return self.g[idx]

  def __repr__(self):
    """
    Defines the behavior of calling print on an instance of this class.
    """
    s = ""
    for row in self.g:
        s += " ".join(["{} ".format(x) for x in row])
        s += "\n"
    return s

  def __add__(self,other):
    """
    Defines the behavior of the + operator
    """
    if self.h != other.h or self.w != other.w:
        raise ValueError("Matrices can only be added if the dimensions are the same") 

    # TODO - your code here
    # Start with a zero matrix and overwrite with sum, element by element
    am = zeroes(self.h, self.w)
    for i in range(self.h):
        for j in range(self.w):
          am[i][j] = self[i][j] + other[i][j]
    return am

  def __neg__(self):
    """
    Defines the behavior of - operator (NOT subtraction)

    Example:

    > my_matrix = Matrix([ [1, 2], [3, 4] ])
    > negative  = -my_matrix
    > print(negative)
      -1.0  -2.0
      -3.0  -4.0
    """
    # TODO - your code here
    # Start with a zero matrix and overwrite with negative elements
    ng = zeroes(self.h, self.w)
    for i in range(self.h):
        for j in range(self.w):
          ng[i][j] = self.g[i][j] * -1
    return ng
      
  def __sub__(self, other):
    """
    Defines the behavior of - operator (as subtraction)
    """
    # TODO - your code here
    # Start with a zero matrix and overwrite with negative elements
    sub = zeroes(self.h, self.w)
    for i in range(self.h):
      for j in range(self.w):
        sub[i][j] = self.g[i][j] - other.g[i][j]
    return sub

  def __mul__(self, other):
    """
    Defines the behavior of * operator (matrix multiplication)
    """
    # TODO - your code here
    #
    # Start with a zero matrix of a[h] * b[w]
    multigrid = zeroes(self.h, other.w)
    for r in range(multigrid.h):
      for c in range(multigrid.w):
        product = 0
        for i in range(self.w):
          product += self[r][i] * other[i][c]
        multigrid[r][c] = product
    return multigrid       

  def __rmul__(self, other):
    """
    Called when the thing on the left of the * is not a matrix.

    Example:

    > identity = Matrix([ [1,0], [0,1] ])
    > doubled  = 2 * identity
    > print(doubled)
      2.0  0.0
      0.0  2.0
    """
    # TODO - your code here
    #
    if isinstance(other, numbers.Number):
      # Start with a zero matrix
      revmulti = zeroes(self.h, self.w)
      for i in range(self.h):
        for j in range(self.w):
          # multiply each element by the scalar
          revmulti[i][j] = self.g[i][j] * other
      return revmulti         