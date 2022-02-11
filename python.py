# 字面值可以如下面这样合并
print('um''um''um''python')
print("um"'um'"python")
print(3*'um'+'python')

# 多个变量或变量和字面值之间可以通过"+"合并
a = 'py'
print(a+"thon")
# 字符串支持索引（下标访问），索引还支持负数，从右边开始计数（第一个为-1）
word = "python"
print(word[0])
print(word[-1])

# 字符串的索引（索引提取单个字符，切片提取字符串）
# note：输出结果包含切片开始，但不包含切片结束，省略开始索引时，默认值为 0，省略结束索引时，默认为到字符串的结尾
print(word[:6])
print(word[0:3])
# 字符串不能修改，是immutable（不可变的），故不能对里面的索引位置进行赋值操作

# 列表：列表是用方括号标注，逗号分隔的一组值。列表 可以包含不同类型的元素，但一般情况下，各个元素的类型相同
# 列表支持索引和切片
# 列表支持合并操作
squares = [1,4,9,16]
squares2 = squares+[25,36,49]
print(squares+[25,36,49])
# 列表是mutable的，可以对内容进行改变
squares2[4] = 24
print(squares2)

#append() 方法可以在列表结尾添加新元素,  为切片赋值可以改变列表大小，甚至清空整个列表
squares2.append(64)
print(squares2)

squares2[0:3] = [2,3,5]
print(squares2)

squares2[0:3] = []
print(squares2)

squares2[:] = []
print(squares2)

# 内置函数 len() 也支持列表, 列表还支持嵌套（即创建包含其他列表的列表）
print(len(squares2))

# 流程控制工具，if 语句包含零个或多个 elif 子句，及可选的 else 子句，for 语句是迭代列表或字符串等任意序列，元素的迭代顺序与在序列中出现的顺序一致,内置函数 range() 常用于遍历数字序列
# 在字典中循环时，用 items() 方法可同时取出键和对应的值：在序列中循环时，用 enumerate() 函数可以同时取出位置索引和对应的值：同时循环两个或多个序列时，用 zip() 函数可以将其内的元素一一匹配