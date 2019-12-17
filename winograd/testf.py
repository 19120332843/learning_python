from __future__ import print_function
from sympy import symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint
from operator import mul
from functools import reduce

a = (0,1,-1)
n = 2
r = 3

def At(a,m,n):
  return Matrix(m, n, lambda i,j: a[i]**j)

def A(a,m,n):
  return At(a, m-1, n).row_insert(m-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))

def F(a,n):
  return Matrix(n, 1, lambda i,j: reduce(mul, ((a[i]-a[k] if k!=i else 1) for k in range(0,n)), 1))

def Fdiag(a,n):
  f=F(a,n)
  return Matrix(n, n, lambda i,j: (f[i,0] if i==j else 0))

def FdiagPlus1(a,n):
  f = Fdiag(a,n-1)
  f = f.col_insert(n-1, zeros(n-1,1))
  f = f.row_insert(n-1, Matrix(1,n, lambda i,j: (1 if j==n-1 else 0)))
  return f

FractionsInG=0
FractionsInA=1
FractionsInB=2
FractionsInF=3

def cookToomFilter(a,n,r,fractionsIn=FractionsInG):
  alpha = n+r-1
  f = FdiagPlus1(a,alpha)
  if f[0,0] < 0:
    f[0,:] *= -1
  if fractionsIn == FractionsInG:
    AT = A(a,alpha,n).T
    G = (A(a,alpha,r).T/f).T
    BT = f * B(a,alpha).T
  elif fractionsIn == FractionsInA:
    BT = f * B(a,alpha).T
    G = A(a,alpha,r)
    AT = (A(a,alpha,n)).T/f
  elif fractionsIn == FractionsInB:
    AT = A(a,alpha,n).T
    G = A(a,alpha,r)
    BT = B(a,alpha).T
  else:
    AT = A(a,alpha,n).T
    G = A(a,alpha,r)
    BT = f * B(a,alpha).T
  return (AT,G,BT,f)
