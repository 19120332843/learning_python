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

## 数字类型

1. int
2. bool: Ture
3. float
4. complex: 1.1 + 2.2j
   
## 字符串

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
   
