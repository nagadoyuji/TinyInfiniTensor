# Utils 模块详细文档

## 模块概述

Utils 模块是 TinyInfiniTensor AI 编译器的工具函数模块，提供了各种辅助功能和实用工具。该模块包含异常处理、算子工具函数、数据生成器等组件，为整个编译器提供基础支持。

## 模块结构

```
src/utils/
├── exception.cc       # 异常处理实现
└── operator_utils.cc  # 算子工具函数实现

include/utils/
├── exception.h        # 异常处理头文件
├── operator_utils.h   # 算子工具函数头文件
└── data_generator.h   # 数据生成器头文件
```

## 核心组件详解

### 1. Exception (异常处理)

**文件位置**: `src/utils/exception.cc`, `include/utils/exception.h`

**主要功能**:
- 提供统一的异常处理机制
- 支持异常信息的链式追加
- 继承自标准库的 `std::runtime_error`

**类定义**:
```cpp
class Exception : public std::runtime_error {
  protected:
    std::string info;

  public:
    Exception(const std::string &msg);
    Exception &operator<<(const std::string &str);
    const char *what() const noexcept override;
};
```

**核心方法**:

#### `Exception::Exception(const std::string &msg)`
- **功能**: 构造函数，初始化异常对象
- **参数**: `msg` - 异常消息
- **实现**: 调用父类 `std::runtime_error` 的构造函数
- **用途**: 创建包含初始消息的异常对象

**示例**:
```cpp
throw Exception("File not found");
throw Exception("Invalid argument: " + std::to_string(value));
```

#### `Exception &operator<<(const std::string &str)`
- **功能**: 链式追加异常信息
- **参数**: `str` - 要追加的字符串
- **返回值**: 异常对象的引用，支持链式调用
- **实现**: 将字符串追加到 `info` 成员变量中

**示例**:
```cpp
Exception e("Error occurred");
e << " at line " << __LINE__ << " in file " << __FILE__;
throw e;
```

**链式调用示例**:
```cpp
throw Exception("Invalid shape")
    << ": expected [2, 3], got [2, 4]"
    << " in operator " << opType.toString();
```

#### `const char *what() const noexcept`
- **功能**: 获取异常的详细信息
- **返回值**: 异常信息的 C 风格字符串
- **实现**: 返回 `info` 成员变量的 C 风格字符串
- **重写**: 重写父类的 `what()` 方法

**示例**:
```cpp
try {
    throw Exception("Test exception") << " with additional info";
} catch (const Exception &e) {
    std::cout << e.what() << std::endl;
    // 输出: "Test exception with additional info"
}
```

**设计特点**:
1. 继承标准异常类，保持兼容性
2. 支持链式调用，便于构建详细的错误信息
3. 使用 `noexcept` 修饰符，保证异常安全
4. 提供统一的异常处理接口

**在项目中的应用**:
- 断言宏 `IT_ASSERT` 使用 `Exception` 抛出异常
- 各种错误检查使用 `Exception` 报告错误
- 提供详细的错误信息，便于调试

### 2. Operator Utils (算子工具函数)

**文件位置**: `src/utils/operator_utils.cc`, `include/utils/operator_utils.h`

**主要功能**:
- 提供算子相关的工具函数
- 支持广播机制
- 处理维度索引和轴转换
- 提供内核属性的字符串表示

#### 2.1 infer_broadcast (广播推导)

**函数签名**:
```cpp
Shape infer_broadcast(const Shape &A, const Shape &B);
```

**功能**:
- 对两个张量的形状进行双向广播
- 返回广播后的形状
- 支持任意维度的广播

**参数**:
- `A` - 第一个张量的形状
- `B` - 第二个张量的形状

**返回值**:
- 广播后的形状

**作业任务**: 需要实现具体的广播推导逻辑

**参考文档**: [ONNX Broadcasting](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md)

**广播规则**:
1. 从右向左比较两个形状的维度
2. 如果维度大小相同，则输出该维度
3. 如果其中一个维度为 1，则输出另一个维度的大小
4. 如果两个维度大小不同且都不为 1，则无法广播
5. 如果一个张量的维度较少，则在左侧补 1 维度

**示例**:
```cpp
// 相同形状
A = [3, 4], B = [3, 4]
输出 = [3, 4]

// 广播（标量）
A = [3, 4], B = [1, 1]
输出 = [3, 4]

// 广播（向量）
A = [3, 4], B = [1, 4]
输出 = [3, 4]

// 广播（不同维度数）
A = [3, 4], B = [4]
输出 = [3, 4]

// 无法广播
A = [3, 4], B = [2, 4]
输出 = 错误
```

**实现思路**:
1. 确定输出形状的维度数（取两个形状的最大维度数）
2. 从右向左逐个维度比较
3. 应用广播规则
4. 检查是否可以广播
5. 返回广播后的形状

#### 2.2 get_real_axis (获取真实轴)

**函数签名**:
```cpp
int get_real_axis(const int &axis, const int &rank);
```

**功能**:
- 将负数轴索引转换为正数轴索引
- 处理轴索引的边界检查
- 支持从末尾开始的索引

**参数**:
- `axis` - 轴索引（可以是负数）
- `rank` - 张量的维度数

**返回值**:
- 转换后的正数轴索引

**实现逻辑**:
```cpp
int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}
```

**示例**:
```cpp
// 正数轴索引
axis = 1, rank = 4
输出 = 1

// 负数轴索引（从末尾开始）
axis = -1, rank = 4
输出 = 3  // 4 + (-1) = 3

axis = -2, rank = 4
输出 = 2  // 4 + (-2) = 2

// 边界检查
axis = -5, rank = 4
输出 = 错误（断言失败）

axis = 4, rank = 4
输出 = 错误（断言失败）
```

**设计特点**:
1. 支持负数轴索引，提高使用便利性
2. 进行边界检查，防止越界访问
3. 使用断言确保参数有效性

#### 2.3 locate_index (定位索引)

**函数签名**:
```cpp
Shape locate_index(size_t inputN, const Shape &shape);
```

**功能**:
- 将线性索引转换为多维坐标
- 支持任意维度的张量
- 按照行优先顺序转换

**参数**:
- `inputN` - 线性索引
- `shape` - 张量的形状

**返回值**:
- 多维坐标向量

**实现逻辑**:
```cpp
Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}
```

**示例**:
```cpp
// 2D 张量
shape = [2, 3]
inputN = 5
输出 = [1, 2]  // 1*3 + 2 = 5

// 3D 张量
shape = [2, 3, 4]
inputN = 10
输出 = [0, 2, 2]  // 0*3*4 + 2*4 + 2 = 10

// 1D 张量
shape = [5]
inputN = 3
输出 = [3]
```

**算法说明**:
1. 从右向左逐个维度计算坐标
2. 使用 `std::div` 同时计算商和余数
3. 余数作为当前维度的坐标
4. 商作为下一轮计算的输入

#### 2.4 delocate_index (反定位索引)

**函数签名**:
```cpp
size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride);
```

**功能**:
- 将多维坐标转换为线性索引
- 支持广播机制
- 使用步长计算线性索引

**参数**:
- `shapeIndex` - 多维坐标
- `shape` - 张量的形状
- `stride` - 各维度的步长

**返回值**:
- 线性索引

**实现逻辑**:
```cpp
size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}
```

**示例**:
```cpp
// 基本情况
shapeIndex = [1, 2]
shape = [2, 3]
stride = [3, 1]
输出 = 1*3 + 2*1 = 5

// 带广播
shapeIndex = [1, 2]
shape = [1, 3]  // 第一维广播
stride = [3, 1]
输出 = (1%1)*3 + 2*1 = 0*3 + 2 = 2
```

**广播处理**:
- 使用模运算处理广播维度
- `index[i] = shapeIndex[i] % shape[i]`
- 如果 `shape[i] = 1`，则 `index[i] = 0`

**步长计算**:
- 步长表示在该维度移动一个单位对应的线性索引增量
- 通常从右向左计算：`stride[i] = stride[i+1] * shape[i+1]`

#### 2.5 get_kernel_attrs_str (获取内核属性字符串)

**函数签名**:
```cpp
std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs);
```

**功能**:
- 将内核属性转换为字符串表示
- 用于调试和日志输出

**参数**:
- `kernelAttrs` - 内核属性元组

**返回值**:
- 内核属性的字符串表示

**实现逻辑**:
```cpp
std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}
```

**辅助函数**:
```cpp
std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}
```

**示例**:
```cpp
KernelAttrs attrs = {Device::CPU, OpType::Add};
std::string str = get_kernel_attrs_str(attrs);
输出 = "CPU, Add"
```

**设计特点**:
1. 提供统一的字符串表示
2. 便于调试和日志输出
3. 支持多种设备和算子类型

## 工具函数间的关系

```
Exception (异常处理)
    ├── 提供统一的异常处理机制
    └── 支持链式追加异常信息

Operator Utils (算子工具函数)
    ├── infer_broadcast (广播推导)
    ├── get_real_axis (获取真实轴)
    ├── locate_index (定位索引)
    ├── delocate_index (反定位索引)
    └── get_kernel_attrs_str (获取内核属性字符串)
```

**协作关系**:
1. `locate_index` 和 `delocate_index` 互为逆操作
2. `infer_broadcast` 用于元素级操作的形状推导
3. `get_real_axis` 用于处理负数轴索引
4. `get_kernel_attrs_str` 用于内核调试和日志

## 设计模式

### 1. 异常处理模式
- 继承标准异常类
- 提供链式调用接口
- 支持详细的错误信息构建

### 2. 工具函数模式
- 提供纯函数接口
- 无副作用，便于测试
- 支持函数组合

### 3. 策略模式
- 不同的广播策略
- 不同的索引计算策略

## 性能优化

### 1. 索引计算优化
- 预先计算步长，减少运行时计算
- 使用 `std::div` 同时计算商和余数
- 支持任意维度的索引计算

### 2. 广播优化
- 在编译时检查是否可以广播
- 避免不必要的内存拷贝
- 支持高效的广播计算

### 3. 异常处理优化
- 使用 `noexcept` 修饰符
- 避免异常处理的性能开销
- 提供详细的错误信息

## 扩展性

### 1. 添加新的异常类型
1. 继承 `Exception` 类
2. 添加特定的异常信息
3. 实现自定义的异常处理逻辑

### 2. 添加新的工具函数
1. 在 `operator_utils.h` 中声明函数
2. 在 `operator_utils.cc` 中实现函数
3. 确保函数接口的一致性

### 3. 支持新的设备类型
1. 在 `device_to_str` 函数中添加新的设备类型
2. 更新 `KernelAttrs` 的处理逻辑
3. 确保与新设备的兼容性

## 作业任务总结

### 1. infer_broadcast 实现
- **任务**: 实现 `infer_broadcast` 函数
- **要求**: 根据两个输入张量的形状，推导广播后的形状
- **参考**: ONNX Broadcasting 文档
- **关键点**:
  - 从右向左比较维度
  - 应用广播规则
  - 检查是否可以广播
  - 返回广播后的形状

## 总结

Utils 模块是 TinyInfiniTensor AI 编译器的工具函数模块，提供了各种辅助功能和实用工具。该模块包含异常处理、算子工具函数、数据生成器等组件，为整个编译器提供基础支持。

该模块的主要特点包括：
1. 提供统一的异常处理机制
2. 实现广播、索引计算等核心工具函数
3. 支持链式调用和函数组合
4. 具有良好的扩展性和可维护性

通过合理的设计和实现，该模块在保证功能完整性的同时，也提供了良好的性能和可扩展性。这些工具函数为整个编译器提供了坚实的基础支持，使得其他模块能够专注于各自的核心功能。