# DataType 数据类型系统详细文档

## 文件概述

**文件路径**: `include/core/data_type.h`

**主要功能**: 定义 TinyInfiniTensor 项目中的数据类型系统，提供类型安全的抽象，支持多种数据类型，并实现类型与索引之间的双向映射。

**设计目标**:
- 提供统一的数据类型抽象
- 支持编译时类型检查
- 实现高效的类型转换
- 兼容 ONNX 数据类型标准

## 类结构

```cpp
class DataType {
  public:
    // 静态数据类型常量
    static const DataType Undefine;
    static const DataType Float32;
    static const DataType UInt8;
    // ... 其他数据类型

    // 类型信息数组
    static constexpr size_t sizePerElement[];
    static constexpr std::string_view names[];
    static constexpr int cpuType[];

  private:
    int index;  // 数据类型索引

  public:
    // 构造函数和运算符
    DataType() = default;
    constexpr DataType(int index) : index(index) {}
    bool operator==(const DataType &rhs) const;
    bool operator<(const DataType &rhs) const;

    // 类型查询方法
    template <typename T> static int get();
    size_t getSize() const;
    string toString() const;
    int cpuTypeInt() const;
    int getIndex() const;
};
```

## 数据类型定义

### 支持的数据类型

| 索引 | 数据类型常量 | C++ 类型 | 字节大小 | 说明 |
|------|-------------|----------|---------|------|
| 0 | Undefine | - | 0 | 未定义类型 |
| 1 | Float32 | float | 4 | 32位浮点数 |
| 2 | UInt8 | uint8_t | 1 | 8位无符号整数 |
| 3 | Int8 | int8_t | 1 | 8位有符号整数 |
| 4 | UInt16 | uint16_t | 2 | 16位无符号整数 |
| 5 | Int16 | int16_t | 2 | 16位有符号整数 |
| 6 | Int32 | int32_t | 4 | 32位有符号整数 |
| 7 | Int64 | int64_t | 8 | 64位有符号整数 |
| 8 | String | std::string | 变长 | 字符串类型 |
| 9 | Bool | int8_t | 1 | 布尔类型 |
| 10 | Float16 | uint16_t | 2 | 16位浮点数 |
| 11 | Double | double | 8 | 64位浮点数 |
| 12 | UInt32 | uint32_t | 4 | 32位无符号整数 |
| 13 | UInt64 | uint64_t | 8 | 64位无符号整数 |
| 14 | PlaceHolder | - | 0 | 占位符类型 |
| 15 | PlaceHolder | - | 0 | 占位符类型 |
| 16 | BFloat16 | uint16_t | 2 | 脑浮点16位 |

### ONNX 兼容性

数据类型定义参考了 [ONNX 数据类型标准](https://onnx.ai/onnx/intro/concepts.html#element-type)，确保与 ONNX 模型的兼容性。

## 核心成员详解

### 1. 静态数据类型常量

```cpp
static const DataType Undefine;
static const DataType Float32;
static const DataType UInt8;
static const DataType Int8;
static const DataType UInt16;
static const DataType Int16;
static const DataType Int32;
static const DataType Int64;
static const DataType String;
static const DataType Bool;
static const DataType Float16;
static const DataType Double;
static const DataType UInt32;
static const DataType UInt64;
static const DataType BFloat16;
```

**功能**: 定义所有支持的数据类型常量

**使用示例**:
```cpp
DataType dtype = DataType::Float32;
if (dtype == DataType::Int32) {
    // 处理 32位整数类型
}
```

**实现细节**:
- 这些常量在对应的 `.cc` 文件中初始化
- 每个常量对应一个特定的索引值
- 提供类型安全的常量访问方式

### 2. sizePerElement 数组

```cpp
static constexpr size_t sizePerElement[]{
    0,              // Undefine
    sizeof(float),   // Float32
    sizeof(uint8_t), // UInt8
    sizeof(int8_t),  // Int8
    sizeof(uint16_t),// UInt16
    sizeof(int16_t), // Int16
    sizeof(int32_t), // Int32
    sizeof(int64_t), // Int64
    sizeof(std::string), // String
    sizeof(int8_t), // Bool
    sizeof(uint16_t),// Float16
    sizeof(double),  // Double
    sizeof(uint32_t),// UInt32
    sizeof(uint64_t),// UInt64
    0,              // PlaceHolder
    0,              // PlaceHolder
    sizeof(uint16_t) // BFloat16
};
```

**功能**: 存储每种数据类型占用的字节数

**设计特点**:
- 使用 `constexpr` 确保编译时计算
- 数组索引与数据类型索引一一对应
- 支持快速查询类型大小

**使用示例**:
```cpp
DataType dtype = DataType::Float32;
size_t size = dtype.getSize();  // 返回 4
```

**注意事项**:
- String 类型的 `sizeof(std::string)` 只返回对象本身的大小，不包括字符串内容
- PlaceHolder 类型的大小为 0
- Bool 类型使用 `int8_t` 实现，占用 1 字节

### 3. names 数组

```cpp
static constexpr std::string_view names[]{
    "Undefine",    "Float32", "UInt8",  "Int8",   "UInt16",
    "Int16",       "Int32",   "Int64",  "String", "Bool",
    "Float16",     "Double",  "UInt32", "UInt64", "PlaceHolder",
    "PlaceHolder", "BFloat16"
};
```

**功能**: 存储每种数据类型的名称字符串

**设计特点**:
- 使用 `std::string_view` 避免字符串拷贝
- 数组索引与数据类型索引一一对应
- 支持类型名称的快速查询

**使用示例**:
```cpp
DataType dtype = DataType::Float32;
std::string name = dtype.toString();  // 返回 "Float32"
```

### 4. cpuType 数组

```cpp
static constexpr int cpuType[]{
    -1,  // Undefine
    0,    // Float32 -> float
    2,    // UInt8 -> uint8_t
    3,    // Int8 -> int8_t
    4,    // UInt16 -> uint16_t
    5,    // Int16 -> int16_t
    6,    // Int32 -> int32_t
    7,    // Int64 -> int64_t
    -1,   // String
    3,    // Bool -> int8_t
    4,    // Float16 -> uint16_t
    9,    // Double -> double
    1,    // UInt32 -> uint32_t
    8,    // UInt64 -> uint64_t
    -1,   // PlaceHolder
    -1,   // PlaceHolder
    4     // BFloat16 -> uint16_t
};
```

**功能**: 存储每种数据类型对应的 CPU 类型索引

**设计特点**:
- 使用整数索引表示 CPU 类型
- -1 表示不支持的类型
- 数组索引与数据类型索引一一对应

**使用示例**:
```cpp
DataType dtype = DataType::Float32;
int cpuTypeIdx = dtype.cpuTypeInt();  // 返回 0
```

### 5. index 私有成员

```cpp
private:
    int index;
```

**功能**: 存储数据类型的索引

**设计特点**:
- 使用整数索引表示数据类型
- 私有成员，确保封装性
- 通过公共方法访问

## 核心方法详解

### 1. 构造函数

#### 默认构造函数
```cpp
DataType() = default;
```

**功能**: 默认构造函数

**注意事项**:
- 由于 JSON 库的要求，不能删除默认构造函数
- 默认构造的 `DataType` 对象可能处于未定义状态
- 建议使用预定义的常量或带索引的构造函数

**相关注释**:
```cpp
// FIXME: default ctor should be deleted but json requires it. Solution:
// https://github.com/nlohmann/json#how-can-i-use-get-for-non-default-constructiblenon-copyable-types
```

#### 带索引的构造函数
```cpp
constexpr DataType(int index) : index(index) {}
```

**功能**: 使用索引构造数据类型对象

**参数**:
- `index` - 数据类型索引

**设计特点**:
- 使用 `constexpr` 支持编译时构造
- 直接初始化索引成员
- 支持常量表达式

**使用示例**:
```cpp
constexpr DataType dtype(1);  // Float32
```

### 2. 比较运算符

#### 相等运算符
```cpp
bool operator==(const DataType &rhs) const { return index == rhs.index; }
```

**功能**: 比较两个数据类型是否相等

**参数**:
- `rhs` - 右操作数

**返回值**: 如果索引相同则返回 `true`，否则返回 `false`

**使用示例**:
```cpp
DataType dtype1 = DataType::Float32;
DataType dtype2 = DataType::Int32;
if (dtype1 == dtype2) {
    // 不会执行
}
```

#### 小于运算符
```cpp
bool operator<(const DataType &rhs) const { return index < rhs.index; }
```

**功能**: 比较两个数据类型的索引大小

**参数**:
- `rhs` - 右操作数

**返回值**: 如果当前索引小于右操作数索引则返回 `true`，否则返回 `false`

**使用示例**:
```cpp
DataType dtype1 = DataType::Float32;  // index = 1
DataType dtype2 = DataType::Int32;   // index = 6
if (dtype1 < dtype2) {
    // 会执行
}
```

**应用场景**:
- 支持数据类型的排序
- 支持在有序容器中使用 `DataType` 作为键

### 3. 类型查询方法

#### 模板方法 get
```cpp
template <typename T> static int get() {
    IT_TODO_HALT_MSG("Unsupported data type");
}
```

**功能**: 根据 C++ 类型获取对应的数据类型索引

**模板参数**:
- `T` - C++ 类型

**返回值**: 数据类型索引

**默认实现**: 抛出异常，表示不支持的数据类型

**模板特化**:
```cpp
template <> inline int DataType::get<float>() { return 0; }
template <> inline int DataType::get<uint32_t>() { return 1; }
template <> inline int DataType::get<uint8_t>() { return 2; }
template <> inline int DataType::get<int8_t>() { return 3; }
template <> inline int DataType::get<uint16_t>() { return 4; }
template <> inline int DataType::get<int16_t>() { return 5; }
template <> inline int DataType::get<int32_t>() { return 6; }
template <> inline int DataType::get<int64_t>() { return 7; }
template <> inline int DataType::get<uint64_t>() { return 8; }
template <> inline int DataType::get<double>() { return 9; }
```

**使用示例**:
```cpp
int idx = DataType::get<float>();  // 返回 0
int idx = DataType::get<int32_t>(); // 返回 6
```

**设计特点**:
- 使用模板特化实现类型到索引的映射
- 编译时类型检查
- 支持所有支持的 C++ 类型

**GCC 兼容性**:
```cpp
// Method definitions are out of declaration due to GCC bug:
// https://stackoverflow.com/questions/49707184/explicit-specialization-in-non-namespace-scope-does-not-compile-in-gcc
```

由于 GCC 的 bug，模板特化必须在类定义外部定义。

#### getSize 方法
```cpp
size_t getSize() const { return sizePerElement[index]; }
```

**功能**: 获取数据类型占用的字节数

**返回值**: 数据类型占用的字节数

**使用示例**:
```cpp
DataType dtype = DataType::Float32;
size_t size = dtype.getSize();  // 返回 4
```

#### toString 方法
```cpp
string toString() const { return string(names[index]); }
```

**功能**: 获取数据类型的名称字符串

**返回值**: 数据类型的名称字符串

**使用示例**:
```cpp
DataType dtype = DataType::Float32;
std::string name = dtype.toString();  // 返回 "Float32"
```

#### cpuTypeInt 方法
```cpp
int cpuTypeInt() const { return cpuType[index]; }
```

**功能**: 获取数据类型对应的 CPU 类型索引

**返回值**: CPU 类型索引

**使用示例**:
```cpp
DataType dtype = DataType::Float32;
int cpuTypeIdx = dtype.cpuTypeInt();  // 返回 0
```

#### getIndex 方法
```cpp
int getIndex() const { return index; }
```

**功能**: 获取数据类型的索引

**返回值**: 数据类型索引

**使用示例**:
```cpp
DataType dtype = DataType::Float32;
int idx = dtype.getIndex();  // 返回 1
```

## 辅助结构体 DT

### 定义

```cpp
template <int index> struct DT {};
```

**功能**: 实现索引到类型的映射

**模板参数**:
- `index` - 数据类型索引

### 模板特化

```cpp
template <> struct DT<0> { using t = bool; };
template <> struct DT<1> { using t = float; };
template <> struct DT<2> { using t = uint8_t; };
template <> struct DT<3> { using t = int8_t; };
template <> struct DT<4> { using t = uint16_t; };
template <> struct DT<5> { using t = int16_t; };
template <> struct DT<6> { using t = int32_t; };
template <> struct DT<7> { using t = int64_t; };
template <> struct DT<8> { using t = char; };
template <> struct DT<9> { using t = int8_t; };
template <> struct DT<10> { using t = uint16_t; };
template <> struct DT<11> { using t = double; };
template <> struct DT<12> { using t = uint32_t; };
template <> struct DT<13> { using t = uint64_t; };
template <> struct DT<16> { using t = uint16_t; };
```

**功能**: 根据索引获取对应的 C++ 类型

**使用示例**:
```cpp
using Float32Type = DT<1>::t;  // float
using Int32Type = DT<6>::t;    // int32_t
```

**设计特点**:
- 使用类型别名 `using t = ...`
- 编译时类型推导
- 支持所有支持的数据类型

## 类型映射关系

### 类型到索引的映射

| C++ 类型 | DataType::get<T>() | 索引 | DataType 常量 |
|----------|-------------------|------|---------------|
| float | DataType::get<float>() | 0 | DataType::Float32 |
| uint32_t | DataType::get<uint32_t>() | 1 | DataType::UInt32 |
| uint8_t | DataType::get<uint8_t>() | 2 | DataType::UInt8 |
| int8_t | DataType::get<int8_t>() | 3 | DataType::Int8 |
| uint16_t | DataType::get<uint16_t>() | 4 | DataType::UInt16 |
| int16_t | DataType::get<int16_t>() | 5 | DataType::Int16 |
| int32_t | DataType::get<int32_t>() | 6 | DataType::Int32 |
| int64_t | DataType::get<int64_t>() | 7 | DataType::Int64 |
| uint64_t | DataType::get<uint64_t>() | 8 | DataType::UInt64 |
| double | DataType::get<double>() | 9 | DataType::Double |

### 索引到类型的映射

| 索引 | DT<index>::t | C++ 类型 | DataType 常量 |
|------|-------------|----------|---------------|
| 0 | DT<0>::t | bool | DataType::Bool |
| 1 | DT<1>::t | float | DataType::Float32 |
| 2 | DT<2>::t | uint8_t | DataType::UInt8 |
| 3 | DT<3>::t | int8_t | DataType::Int8 |
| 4 | DT<4>::t | uint16_t | DataType::UInt16 |
| 5 | DT<5>::t | int16_t | DataType::Int16 |
| 6 | DT<6>::t | int32_t | DataType::Int32 |
| 7 | DT<7>::t | int64_t | DataType::Int64 |
| 8 | DT<8>::t | char | DataType::String |
| 9 | DT<9>::t | int8_t | DataType::Bool |
| 10 | DT<10>::t | uint16_t | DataType::Float16 |
| 11 | DT<11>::t | double | DataType::Double |
| 12 | DT<12>::t | uint32_t | DataType::UInt32 |
| 13 | DT<13>::t | uint64_t | DataType::UInt64 |
| 16 | DT<16>::t | uint16_t | DataType::BFloat16 |

## 使用示例

### 示例1: 创建数据类型对象

```cpp
// 使用预定义常量
DataType dtype1 = DataType::Float32;
DataType dtype2 = DataType::Int32;

// 使用索引构造
DataType dtype3(1);  // Float32
DataType dtype4(6);  // Int32
```

### 示例2: 比较数据类型

```cpp
DataType dtype1 = DataType::Float32;
DataType dtype2 = DataType::Int32;

if (dtype1 == dtype2) {
    // 不会执行
}

if (dtype1 < dtype2) {
    // 会执行，因为 1 < 6
}
```

### 示例3: 获取类型信息

```cpp
DataType dtype = DataType::Float32;

// 获取类型大小
size_t size = dtype.getSize();  // 4

// 获取类型名称
std::string name = dtype.toString();  // "Float32"

// 获取 CPU 类型索引
int cpuTypeIdx = dtype.cpuTypeInt();  // 0

// 获取类型索引
int idx = dtype.getIndex();  // 1
```

### 示例4: 类型到索引的映射

```cpp
// 使用模板方法 get
int idx1 = DataType::get<float>();     // 0
int idx2 = DataType::get<int32_t>();   // 6
int idx3 = DataType::get<uint8_t>();   // 2
```

### 示例5: 索引到类型的映射

```cpp
// 使用 DT 结构体
using Float32Type = DT<1>::t;   // float
using Int32Type = DT<6>::t;     // int32_t
using UInt8Type = DT<2>::t;     // uint8_t

// 使用类型
Float32Type value1 = 3.14f;
Int32Type value2 = 42;
UInt8Type value3 = 255;
```

### 示例6: 在内核中使用

```cpp
template <typename T>
void compute(DataType dtype) {
    // 获取数据指针
    T* data = tensor->getRawDataPtr<T*>();
    
    // 使用类型信息
    size_t size = dtype.getSize();
    std::string name = dtype.toString();
    
    // 处理数据
    for (size_t i = 0; i < tensor->size(); ++i) {
        data[i] = compute_value<T>(data[i]);
    }
}

// 调用
compute<float>(DataType::Float32);
compute<int32_t>(DataType::Int32);
```

## 设计模式

### 1. 类型安全的枚举模式
- 使用整数索引表示数据类型
- 提供类型安全的常量访问
- 支持编译时类型检查

### 2. 双向映射模式
- 类型到索引的映射：`DataType::get<T>()`
- 索引到类型的映射：`DT<index>::t`
- 实现高效的类型转换

### 3. 模板特化模式
- 使用模板特化实现类型映射
- 编译时类型推导
- 避免运行时类型检查

## 性能优化

### 1. 编译时计算
- 使用 `constexpr` 确保编译时计算
- 避免运行时开销
- 支持常量表达式

### 2. 数组索引访问
- 使用数组索引实现快速查找
- 时间复杂度 O(1)
- 缓存友好

### 3. 零开销抽象
- 所有操作在编译时完成
- 没有虚函数调用
- 没有运行时类型信息

## 扩展性

### 添加新的数据类型

1. **在 DataType 类中添加常量**:
```cpp
static const DataType NewType;
```

2. **更新数组**:
```cpp
static constexpr size_t sizePerElement[]{..., sizeof(new_type)};
static constexpr std::string_view names[]{..., "NewType"};
static constexpr int cpuType[]{..., cpu_type_index};
```

3. **添加模板特化**:
```cpp
template <> inline int DataType::get<new_type>() { return new_index; }
```

4. **添加 DT 特化**:
```cpp
template <> struct DT<new_index> { using t = new_type; };
```

5. **在 .cc 文件中初始化常量**:
```cpp
const DataType DataType::NewType(new_index);
```

## 注意事项

### 1. 默认构造函数
- 默认构造的 `DataType` 对象可能处于未定义状态
- 建议使用预定义的常量或带索引的构造函数

### 2. 类型映射一致性
- 确保 `DataType::get<T>()` 和 `DT<index>::t` 的映射一致
- 确保数组索引与数据类型索引一一对应

### 3. GCC 兼容性
- 模板特化必须在类定义外部定义
- 这是由于 GCC 的 bug 导致的限制

### 4. JSON 兼容性
- 默认构造函数不能删除，因为 JSON 库的要求
- 参考 [nlohmann/json 文档](https://github.com/nlohmann/json#how-can-i-use-get-for-non-default-constructiblenon-copyable-types)

## 内存布局示意图

### 1. DataType 对象内存布局

```
┌─────────────────────────────────────────────────┐
│         DataType 对象内存布局              │
├─────────────────────────────────────────────────┤
│  对象大小: sizeof(DataType) = 4 字节      │
│                                         │
│  ┌─────────────┐                        │
│  │  int index  │  (4 bytes)            │
│  │   = 1       │  (Float32 的索引)    │
│  └─────────────┘                        │
└─────────────────────────────────────────────────┘
```

**说明**:
- `DataType` 对象只包含一个 `int index` 成员
- 对象大小为 4 字节（32位系统）
- 存储的是数据类型的索引值

### 2. 静态数组内存布局

```
┌─────────────────────────────────────────────────────────────┐
│              静态数组内存布局                     │
├─────────────────────────────────────────────────────────────┤
│                                                     │
│  sizePerElement 数组 (编译时常量)                   │
│  ┌────┬────┬────┬────┬────┬────┬────┐     │
│  │ 0  │ 4  │ 1  │ 1  │ 2  │ 2  │ ... │
│  └────┴────┴────┴────┴────┴────┴────┘     │
│   [0] [1] [2] [3] [4] [5] [6] ...        │
│                                                     │
│  names 数组 (编译时常量)                          │
│  ┌──────────┬──────────┬──────────┬──────────┐ │
│  │"Undefine"│ "Float32"│ "UInt8"  │ "Int8"   │ ... │
│  └──────────┴──────────┴──────────┴──────────┘ │
│   [0]       [1]       [2]       [3] ...        │
│                                                     │
│  cpuType 数组 (编译时常量)                        │
│  ┌────┬────┬────┬────┬────┬────┬────┐     │
│  │ -1 │ 0  │ 2  │ 3  │ 4  │ 5  │ ... │
│  └────┴────┴────┴────┴────┴────┴────┘     │
│   [0] [1] [2] [3] [4] [5] [6] ...        │
│                                                     │
└─────────────────────────────────────────────────────────────┘
```

**说明**:
- 三个静态数组在编译时确定
- 数组索引与数据类型索引一一对应
- 使用 `constexpr` 确保编译时计算

### 3. 完整内存布局示例

```
┌─────────────────────────────────────────────────────────────────┐
│            完整内存布局示例                        │
├─────────────────────────────────────────────────────────────────┤
│                                                         │
│  代码段 (Code Segment)                                 │
│  ┌───────────────────────────────────────────────────┐      │
│  │ DataType::Float32 (静态常量)                │      │
│  │   index = 1                                  │      │
│  └───────────────────────────────────────────────────┘      │
│  ┌───────────────────────────────────────────────────┐      │
│  │ DataType::Int32 (静态常量)                  │      │
│  │   index = 6                                  │      │
│  └───────────────────────────────────────────────────┘      │
│                                                         │
│  只读数据段 (Read-Only Data Segment)                  │
│  ┌───────────────────────────────────────────────────┐      │
│  │ sizePerElement[] (编译时常量数组)            │      │
│  │ [0, 4, 1, 1, 2, 2, 4, 8, ...]           │      │
│  └───────────────────────────────────────────────────┘      │
│  ┌───────────────────────────────────────────────────┐      │
│  │ names[] (编译时常量数组)                     │      │
│  │ ["Undefine", "Float32", "UInt8", ...]      │      │
│  └───────────────────────────────────────────────────┘      │
│  ┌───────────────────────────────────────────────────┐      │
│  │ cpuType[] (编译时常量数组)                   │      │
│  │ [-1, 0, 2, 3, 4, 5, 6, 7, ...]          │      │
│  └───────────────────────────────────────────────────┘      │
│                                                         │
│  栈段 (Stack Segment)                                 │
│  ┌───────────────────────────────────────────────────┐      │
│  │ dtype (局部变量)                              │      │
│  │   index = 1                                  │      │
│  └───────────────────────────────────────────────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────────────┘
```

**说明**:
- 静态常量存储在代码段或只读数据段
- 静态数组存储在只读数据段
- 局部变量存储在栈段
- 所有数据类型信息在编译时确定

### 4. 张量数据内存布局

```
┌─────────────────────────────────────────────────────────────┐
│           张量数据内存布局示例                     │
├─────────────────────────────────────────────────────────────┤
│                                                     │
│  Float32 张量 [2, 3]                              │
│  ┌───┬───┬───┬───┬───┬───┐                 │
│  │1.0│2.0│3.0│4.0│5.0│6.0│ ...           │
│  └───┴───┴───┴───┴───┴───┘                 │
│   [0] [1] [2] [3] [4] [5]                      │
│                                                     │
│  每个元素占用 4 字节 (sizePerElement[1] = 4)        │
│  总内存: 2 * 3 * 4 = 24 字节                     │
│                                                     │
│  内存地址:                                          │
│  ┌───────────────────────────────────────────────┐       │
│  │ 0x1000: 1.0 (字节 0-3)                │       │
│  │ 0x1004: 2.0 (字节 4-7)                │       │
│  │ 0x1008: 3.0 (字节 8-11)               │       │
│  │ 0x100C: 4.0 (字节 12-15)              │       │
│  │ 0x1010: 5.0 (字节 16-19)              │       │
│  │ 0x1014: 6.0 (字节 20-23)              │       │
│  └───────────────────────────────────────────────┘       │
│                                                     │
└─────────────────────────────────────────────────────────────┘
```

**说明**:
- 张量数据在内存中连续存储
- 每个元素的大小由 `sizePerElement[index]` 决定
- 内存布局为行优先顺序

## 数据流程图

### 1. 类型查询流程

```
┌─────────────────────────────────────────────────────────┐
│            类型查询数据流程                        │
└─────────────────────────────────────────────────────────┘

用户代码
    │
    ▼
┌───────────────────┐
│  DataType dtype  │
│   = DataType::  │
│     Float32      │
└────────┬──────────┘
         │
         │ index = 1
         │
         ▼
┌─────────────────────────────────────────────────────┐
│         dtype.getSize() 调用                 │
├─────────────────────────────────────────────────────┤
│                                             │
│  1. 访问私有成员 index                      │
│     int idx = dtype.index;  // idx = 1        │
│                                             │
│  2. 使用索引查询 sizePerElement 数组          │
│     return sizePerElement[idx];                  │
│     // sizePerElement[1] = sizeof(float) = 4   │
│                                             │
└─────────────────────────────────────────────────────┘
         │
         │ return 4
         │
         ▼
    用户代码获得类型大小
```

### 2. 类型到索引映射流程

```
┌─────────────────────────────────────────────────────────┐
│         类型到索引映射数据流程                    │
└─────────────────────────────────────────────────────────┘

用户代码
    │
    ▼
┌───────────────────────────┐
│  DataType::get<float>()  │
└────────┬──────────────────┘
         │
         │ 编译时模板匹配
         │
         ▼
┌─────────────────────────────────────────────────────┐
│       模板特化匹配过程                       │
├─────────────────────────────────────────────────────┤
│                                             │
│  1. 尝试匹配模板特化                       │
│     template <> inline int                      │
│     DataType::get<float>() { return 0; }       │
│                                             │
│  2. 编译器选择匹配的特化                   │
│     匹配成功: float -> return 0               │
│                                             │
└─────────────────────────────────────────────────────┘
         │
         │ return 0 (编译时常量)
         │
         ▼
    用户代码获得类型索引
```

### 3. 索引到类型映射流程

```
┌─────────────────────────────────────────────────────────┐
│         索引到类型映射数据流程                    │
└─────────────────────────────────────────────────────────┘

用户代码
    │
    ▼
┌───────────────────────────┐
│  using Type = DT<1>::t │
└────────┬──────────────────┘
         │
         │ 编译时模板实例化
         │
         ▼
┌─────────────────────────────────────────────────────┐
│       模板特化实例化过程                     │
├─────────────────────────────────────────────────────┤
│                                             │
│  1. 尝试匹配模板特化                       │
│     template <> struct DT<1> {                │
│         using t = float;                        │
│     };                                        │
│                                             │
│  2. 编译器选择匹配的特化                   │
│     匹配成功: DT<1>::t = float             │
│                                             │
└─────────────────────────────────────────────────────┘
         │
         │ Type = float (编译时类型)
         │
         ▼
    用户代码获得 C++ 类型
```

### 4. 完整类型转换流程

```
┌─────────────────────────────────────────────────────────┐
│         完整类型转换数据流程                    │
└─────────────────────────────────────────────────────────┘

用户代码
    │
    ▼
┌───────────────────────────────────┐
│  DataType dtype = DataType::  │
│     Float32;                   │
└────────┬──────────────────────────┘
         │
         │ 创建 DataType 对象
         │ index = 1
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│  dtype.getSize()  │              │  dtype.toString()  │
└────────┬────────┘              └────────┬────────┘
         │                               │
         │ 查询 sizePerElement[1]        │ 查询 names[1]
         │ = 4                           │ = "Float32"
         │                               │
         ▼                               ▼
    size_t size = 4            string name = "Float32"
         │                               │
         └───────────────┬───────────────┘
                         │
                         ▼
              用户代码获得完整类型信息
```

### 5. 内核数据类型分发流程

```
┌─────────────────────────────────────────────────────────┐
│         内核数据类型分发流程                    │
└─────────────────────────────────────────────────────────┘

内核调用
    │
    ▼
┌───────────────────────────────────┐
│  compute(op, context)       │
└────────┬──────────────────────────┘
         │
         │ 获取数据类型索引
         │
         ▼
┌───────────────────────────────────┐
│  int dataTypeIdx =          │
│    op->getDType().getIndex();│
└────────┬──────────────────────────┘
         │
         │ dataTypeIdx = 1 (Float32)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│       switch (dataTypeIdx) 数据类型分发        │
├─────────────────────────────────────────────────────┤
│                                             │
│  case 1:                                    │
│      doCompute<float>(op, context);             │
│      break;                                   │
│                                             │
│  case 6:                                    │
│      doCompute<int32_t>(op, context);          │
│      break;                                   │
│                                             │
│  default:                                    │
│      IT_TODO_HALT();                           │
│                                             │
└─────────────────────────────────────────────────────┘
         │
         │ 模板实例化
         │
         ▼
┌───────────────────────────────────┐
│  template <typename T>       │
│  void doCompute(...) {       │
│      T* data = tensor->      │
│        getRawDataPtr<T*>(); │
│      // 处理数据            │
│  }                          │
└────────┬──────────────────────────┘
         │
         │ T = float
         │
         ▼
    执行具体计算
```

### 6. 张量内存分配流程

```
┌─────────────────────────────────────────────────────────┐
│         张量内存分配数据流程                    │
└─────────────────────────────────────────────────────────┘

创建张量
    │
    ▼
┌───────────────────────────────────┐
│  Tensor tensor = Tensor(      │
│    {1024, 1024},          │
│    DataType::Float32,        │
│    runtime                 │
│  );                          │
└────────┬──────────────────────────┘
         │
         │ 计算元素总数
         │ 1024 * 1024 = 1,048,576
         │
         ▼
┌───────────────────────────────────┐
│  size_t numElements =        │
│    tensor->size();          │
└────────┬──────────────────────────┘
         │
         │ 获取类型大小
         │
         ▼
┌───────────────────────────────────┐
│  size_t elementSize =        │
│    tensor->getDType()        │
│      .getSize();            │
│  // sizePerElement[1] = 4  │
└────────┬──────────────────────────┘
         │
         │ elementSize = 4
         │
         ▼
┌───────────────────────────────────┐
│  size_t totalSize =         │
│    numElements *            │
│    elementSize;             │
│  // 1,048,576 * 4 =       │
│  // 4,194,304 字节         │
└────────┬──────────────────────────┘
         │
         │ totalSize = 4,194,304
         │
         ▼
┌───────────────────────────────────┐
│  void* ptr = runtime->     │
│    alloc(totalSize);        │
└────────┬──────────────────────────┘
         │
         │ ptr = 0x10000000
         │
         ▼
    分配内存成功
```

### 7. 类型信息查询综合流程

```
┌─────────────────────────────────────────────────────────┐
│       类型信息查询综合数据流程                   │
└─────────────────────────────────────────────────────────┘

用户代码
    │
    ▼
┌───────────────────────────────────┐
│  DataType dtype =             │
│    DataType::Float32;         │
└────────┬──────────────────────────┘
         │
         │ index = 1
         │
         ├──────────┬──────────┬──────────┬──────────┐
         │          │          │          │          │
         ▼          ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│getSize()││toString()││cpuTypeInt()││getIndex()│
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │            │            │            │
     │查询数组    │查询数组    │查询数组    │直接返回
     │sizePerEl   │names       │cpuType     │index
     │ement[1]    │[1]         │[1]         │
     │= 4         │="Float32"   │= 0         │= 1
     │            │            │            │
     ▼            ▼            ▼            ▼
  size_t=4   string="Float32" int=0       int=1
     │            │            │            │
     └────────────┴────────────┴────────────┘
                  │
                  ▼
         用户代码获得完整类型信息
```

## 总结

`DataType` 类是 TinyInfiniTensor 项目中的核心数据类型系统，提供了类型安全的抽象，支持多种数据类型，并实现类型与索引之间的双向映射。

**主要特点**:
1. 支持多种数据类型，兼容 ONNX 标准
2. 实现类型到索引的双向映射
3. 使用模板特化实现编译时类型检查
4. 提供高效的类型查询和转换
5. 支持零开销抽象

**应用场景**:
- 张量的数据类型定义
- 内核的数据类型分发
- 类型安全的类型转换
- 编译时类型检查

通过合理的设计和实现，`DataType` 类在保证类型安全的同时，也提供了良好的性能和可扩展性。