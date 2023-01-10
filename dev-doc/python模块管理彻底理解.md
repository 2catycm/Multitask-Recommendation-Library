## 彻底理解Python模块

### 问题背景
- 不同目录下运行python解释器，模块不同吗？
- import .a是什么操作
- 什么情况下可以引用父目录下别的模块呢？
- 从pip拿下来的代码，是怎么被import成功的呢？命名空间不会冲突吗？

### 基础知识

- 概念
模块：一个包含Python代码的文件，以.py结尾，模块名即文件名
包：一个包含多个模块的特殊目录，包含一个特殊文件__init__.py
库：多个包和模块的集合

- 基础语句
import 模块或者包
获得了 模块名.变量名 （包括变量、函数、类），运行了模块中的代码
包则是运行了__init__.py中的代码


- 问题1：包内的模块不是包拥有的模块
import pack
pack.test.fun() #报错
import pack.test
pack.test.fun() #正确
- 特例：__all__属性
相当于 __init__.py中导入了子模块，所以拥有了子模块。


### 动态导入

```
s_obj = __import__("os")
#或者
import importlib
myos=importlib.import_module("os")
```

### 解释器从哪里找模块

sys.path


### 相对导入

.为当前目录
...为上一级目录