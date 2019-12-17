# wincnn 计算说明

以F(2,3)为例

输入`wincnn.showCookToomFilter((0,1,-1), 2, 3)`；的`alpha = 2+3-1`

1. 计算步骤1，其中`a = (0,1,-1)  n = alpha-1 = 2+3-1-1 = 3`

```python
def F(a,n):
  return Matrix(n, 1, lambda i,j: reduce(mul, ((a[i]-a[k] if k!=i else 1) for k in range(0,n)), 1))
```

意思是一开始输入的`a=(0,1,-1)`，计算出一个`lambda i,j: reduce(mul, ((a[i]-a[k] if k!=i else 1) for k in range(0,n)), 1)`就是就得出一个n维向量，$n是a的维数$，以这个为例子最后得出的数是`F = [[1*1*(1-0)*(-1-0)=-1][1*(0-1)*1*(-1-1)=2][1*(0-(-1))*(1-(-1))*1=2]] = [[-1], [2], [2]]`

2. 计算步骤2,其中`a = [[-1], [2], [2]]，  n = alpha-1 = 2+3-1-1 = 3`

```python
def Fdiag(a,n):
  f=F(a,n)
  return Matrix(n, n, lambda i,j: (f[i,0] if i==j else 0))
```

填充矩阵，得到`[[-1, 0, 0], [0, 2, 0], [0, 0, 2]]`

3. 计算步骤3，其中`f = [[-1, 0, 0], [0, 2, 0], [0, 0, 2]]`

```python
  f = f.col_insert(n-1, zeros(n-1,1)) # 28
```

在f矩阵的第n-1列加一列(n-1 * 1)的0向量得`[[-1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]]`

4. 计算步骤4,其中`f = [[-1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]]`

```python
  f = f.row_insert(n-1, Matrix(1,n, lambda i,j: (1 if j==n-1 else 0))) # 29
```

往里面加了一行，在行和列相等的位置取1，其余位置取0得`[[-1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]`

5. 计算步骤5,其中`f = [[-1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]`

```python
if f[0,0] < 0:
  f[0,:] *= -1
```

如果矩阵f[0,0]位置上的数小于0，则第1行的全体数字乘-1得`[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]`

6. 计算步骤6，其中`f = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]，且fractionsIn=FractionsInG`

```python
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
```

那么这里只做：

```python
if fractionsIn == FractionsInG:
  AT = A(a,alpha,n).T
  G = (A(a,alpha,r).T/f).T
  BT = f * B(a,alpha).T
```

7. 计算步骤7，`a = (0,1,-1), m = m-1 = alpha-1 = n + r - 1 - 1 = 2+3-1-1 = 3, n = 2`

```python
def At(a,m,n):
  return Matrix(m, n, lambda i,j: a[i]**j)
```

我觉得他是计算对于位置的数值的j(j代表列数)次方，首先出来一个矩阵`[[0,0],[1,1],[-1,-1]]`，然后计算j次方`[[0**0,0**1],[1**0,1**1],[-1**0,-1**1]]`得`[[1,0],[1,1],[1,-1]]`

8. 计算步骤8，`At(a, m-1, n) = [[1,0],[1,1],[1,-1]]`

```python
def A(a,m,n):
  return At(a, m-1, n).row_insert(m-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))
```

在后面加一行，如果列号等于n-1则唯1，其他为0得`[[1,0],[1,1],[1,-1],[0,1]]`

9. 计算步骤9，`A(a,alpha,n) = [[1,0],[1,1],[1,-1],[0,1]]`

```python
AT = A(a,alpha,n).T
```

**转置得`[[1, 1, 1, 0], [0, 1, -1, 1]]`**

10. 计算步骤10，`a = (0,1,-1), m = m-1 = alpha-1 = n + r - 1 - 1 = 2+3-1-1 = 3, n = r = 3`

```python
def At(a,m,n):
  return Matrix(m, n, lambda i,j: a[i]**j)
```

同上7，先生成一个矩阵`[[0,0,0],[1,1,1],[-1,-1,-1]]`,然后算次方`[[0**0,0**1,0**2],[1**0,1**1,1**2],[-1**0,-1**1,-1**2]]`得`[[1,0,0],[1,1,1],[1,-1,1]]`

11. 计算步骤11，`At(a, m-1, n) = [[1,0,0],[1,1,1],[1,-1,1]]`

```python
def A(a,m,n):
  return At(a, m-1, n).row_insert(m-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))
```

在后面加一行，如果列号等于n-1则唯1，其他为0得`[[1,0,0], [1,1,1], [1,-1,1], [0,0,1]]`

12. 计算步骤12，`A(a,alpha,n) = [[1,0,0], [1,1,1], [1,-1,1], [0,0,1]], f = [[1,0,0,0], [0,2,0,0], [0,0,2,0], [0,0,0,1]]`

```python
G = (A(a,alpha,r).T/f).T
```

**G = (A的转置/f)的转置**

13. 计算步骤13，`a = (0,1,-1), n = alpha - 1 = 3`

```python
def Lx(a,n):
  x=symbols('x')
  return Matrix(n, 1, lambda i,j: Poly((reduce(mul, ((x-a[k] if k!=i else 1) for k in range(0,n)), 1)).expand(basic=True), x))
```

画三条线，`[[1*1*(x-1)*(x+1)],[1*x*1*(x+1)],[1*x*(x-1)*1]] -> [[x**2 - 1],[x**2 + x],[x**2 - x]] -> [[Poly(x**2 - 1, x, domain='ZZ')], [Poly(x**2 + x, x, domain='ZZ')], [Poly(x**2 - x, x, domain='ZZ')]]`根据描述就是这样

14. 计算步骤14，`a = (0,1,-1), n = alpha - 1 = 3`

```python
f = F(a, n)
```

同上`f = [[-1], [2], [2]]`

15. 计算步骤15，`a = (0,1,-1), n = alpha - 1 = 3， lx = [[x**2 - 1],[x**2 + x],[x**2 - x]]， f = [[-1], [2], [2]]`

```python
def L(a,n):
  lx = Lx(a,n)
  f = F(a, n)
  return Matrix(n, n, lambda i,j: lx[i, 0].nth(j)/f[i]).T
```

`Matrix(n, n, lambda i,j: lx[i, 0].nth(j)/f[i]) = [[(-1)/(-1),0,1/(-1)],[0,1/2,1/2],[0,(-1)/2,1/2]] = [[1,0,-1],[0,1/2,1/2],[0,(-1/2),1/2]]`

`Matrix(n, n, lambda i,j: lx[i, 0].nth(j)/f[i]).T = [[1, 0, 0], [0, 1/2, -1/2], [-1, 1/2, 1/2]]`

16. 计算步骤16， 
