[TOC]

# basic grammar

## 注释

1. 单行注释
   
    ```python
    #单行注释
    print(hello)
    ```

2. 多行注释
   
    ```python
    """
    这是多行注释
    这是多行注释
    """

    '''
    这是多行注释
    这是多行注释
    '''

    print(hello)
    ```
  
## 多行语句

1. Python 通常是一行写完一条语句，但如果语句很长，我们可以使用反斜杠( \ )来实现多行语句，例如：

    ```python
    total = a + \
            b + \
            c
    ```

2. 在 [], {}, 或 () 中的多行语句，不需要使用反斜杠( \ )。

## 标准数据类型

1. number
2. string
3. list
4. tuple
5. set
6. dictionary
   
不可变数据（3 个）：Number（数字）、String（字符串）、Tuple（元组）；

可变数据（3 个）：List（列表）、Dictionary（字典）、Set（集合）。

### [number](number.py)

1. int
2. bool: Ture
3. float
4. complex: 1.1 + 2.2j
5. 数值运算
    ```python
    5 + 4  # 加法
    4.3 - 2 # 减法
    3 * 7  # 乘法
    2 / 4  # 除法，得到一个浮点数
    2 // 4 # 除法，得到一个整数
    17 % 3 # 取余
    2 ** 5 # 乘方
    ```

### string

1. python中单引号和双引号使用完全相同；
2. 转义符 ' \ '；
3. 反斜杠可以用来转义，使用r可以让反斜杠不发生转义。。 如 r"this is a line with \n" 则\n会显示，并不是换行；
4. 按字面意义级联字符串，如"this " "is " "string"会被自动转换为this is string；
5. 字符串可以用 + 运算符连接在一起，用 * 运算符重复；
6. Python 中的字符串有两种索引方式，从左往右以 0 开始，从右往左以 -1 开始；
7. 字符串的截取的语法格式如下：`变量[头下标:尾下标:步长]`；
    
```python
str='Runoob'

print(str)                 # 输出字符串
print(str[0:-1])           # 输出第一个到倒数第二个的所有字符
print(str[0])              # 输出字符串第一个字符
print(str[2:5])            # 输出从第三个开始到第五个的字符
print(str[2:])             # 输出从第三个开始后的所有字符
print(str * 2)             # 输出字符串两次
print(str + '你好')        # 连接字符串

print('------------------------------')

print('hello\nrunoob')      # 使用反斜杠(\)+n转义特殊字符
print(r'hello\nrunoob')     # 在字符串前面添加一个 r，表示原始字符串，不会发生转义
```

### list

列表语法格式：`变量[头下标:尾下标:步长]`，索引值以 0 为开始值，-1 为从末尾的开始位置。

加号 `+` 是列表连接运算符，星号 `*` 是重复操作。[实例](testlist.py)

与Python字符串不一样的是，列表中的元素是可以改变的,[实例](testlist0.py)

### tuple

元组（tuple）与列表类似，不同之处在于元组的元素不能修改。元组写在小括号 () 里，元素之间用逗号隔开。

元组中的元素类型也可以不相同，[实例](testtuple0.py)。

```python
tup1 = ()    # 空元组
tup2 = (20,) # 一个元素，需要在元素后添加逗号
```

### [set](testset0.py)

集合（set）是由一个或数个形态各异的大小整体组成的，构成集合的事物或对象称作元素或是成员。

基本功能是进行成员关系测试和删除重复元素。

可以使用大括号 { } 或者 set() 函数创建集合，注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典。

创建格式：

```python
parame = {value01,value02,...}
#或者
set(value)
```

### [dictionary](testdictionary0.py)

列表是有序的对象集合，字典是无序的对象集合。两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。

字典是一种映射类型，字典用 { } 标识，它是一个无序的` 键(key) : 值(value) `的集合。

键(key)必须使用不可变类型。

在同一个字典中，键(key)必须是唯一的。

## 空行

函数之间或类的方法之间用空行分隔，表示一段新的代码的开始。类和函数入口之间也用一行空行分隔，以突出函数入口的开始。

空行与代码缩进不同，空行并不是Python语法的一部分。书写时不插入空行，Python解释器运行也不会出错。但是空行的作用在于分隔两段不同功能或含义的代码，便于日后代码的维护或重构。

## print输出

print 默认输出是换行的，如果要实现不换行需要在变量末尾加上 end=""

```python
x="a"
y="b"
# 换行输出
print( x )
print( y )
 
print('---------')
# 不换行输出
print( x, end=" " )
print( y, end=" " )
print()
```

## import 与 from...import

在 python 用 import 或者 from...import 来导入相应的模块。

1. 将整个模块(somemodule)导入，格式为： `import somemodule`
2. 从某个模块中导入某个函数,格式为： `from somemodule import somefunction`
3. 从某个模块中导入多个函数,格式为： `from somemodule import firstfunc, secondfunc, thirdfunc`
4. 将某个模块中的全部函数导入，格式为： `from somemodule import *`
   
## [命令行参数](https://www.runoob.com/python3/python3-command-line-arguments.html)

