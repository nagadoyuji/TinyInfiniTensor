# Operators 模块详细文档

## 模块概述

Operators 模块是 TinyInfiniTensor AI 编译器的算子实现模块，提供了各种深度学习计算中常用的算子实现。该模块中的算子都继承自 Core 模块中的 `Operator` 基类，实现了具体的计算逻辑和形状推导功能。

## 模块结构

```
src/operators/
├── matmul.cc          # 矩阵乘法算子实现
├── transpose.cc       # 转置算子实现
├── element_wise.cc    # 元素级操作算子实现
├── concat.cc          # 拼接算子实现
└── unary.cc           # 一元操作算子实现
```

## 核心算子详解

### 1. Matmul (矩阵乘法算子)

**文件位置**: `src/operators/matmul.cc`

**主要功能**:
- 实现矩阵乘法运算
- 支持矩阵转置选项
- 推导输出张量的形状
- 支持批处理矩阵乘法

**核心方法**:

#### `MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA, bool transB)`
- **功能**: 构造函数，初始化矩阵乘法算子
- **参数**:
  - `graph` - 计算图对象
  - `A` - 输入矩阵 A
  - `B` - 输入矩阵 B
  - `C` - 输出矩阵 C
  - `transA` - 是否对矩阵 A 进行转置
  - `transB` - 是否对矩阵 B 进行转置
- **算子类型**: `OpType::MatMul`
- **输入**: 2 个张量（A 和 B）
- **输出**: 1 个张量（C）

**矩阵乘法公式**:
```
C = A × B
```

如果 `transA = true`，则实际计算 `A^T × B`
如果 `transB = true`，则实际计算 `A × B^T`

#### `string MatmulObj::toString() const`
- **功能**: 将矩阵乘法算子转换为字符串表示
- **返回值**: 算子的字符串描述
- **包含信息**:
  - 算子类型
  - 转置选项（A^T 或 A，B^T 或 B）
  - 输入张量的唯一标识符
  - 输出张量的唯一标识符
  - 矩阵维度（m, n, k）

**输出格式示例**:
```
Matmul([A,B],A=1,B=2,C=3,mnk=[128,256,64])
```

#### `optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)`
- **功能**: 推导矩阵乘法后的输出形状
- **参数**: `inputs` - 输入张量向量（包含 A 和 B）
- **返回值**: 输出张量的形状向量
- **作业任务**: 需要实现具体的形状推导逻辑
- **参考文档**: [ONNX GEMM Operator](https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm)

**形状推导规则**:
1. 获取输入矩阵 A 和 B 的形状
2. 根据 `transA` 和 `transB` 选项确定实际的矩阵维度
3. 计算输出矩阵的形状：
   - 如果 A 的形状为 `[m, k]`（或 `[m, k]` 转置后为 `[k, m]`）
   - 如果 B 的形状为 `[k, n]`（或 `[k, n]` 转置后为 `[n, k]`）
   - 则输出 C 的形状为 `[m, n]`

**批处理支持**:
- 支持批处理矩阵乘法
- 如果输入包含批次维度，输出也包含相应的批次维度

**示例**:
```cpp
// 基本矩阵乘法
A: [128, 64]
B: [64, 256]
C: [128, 256]

// 带转置的矩阵乘法
A: [64, 128], transA = true
B: [256, 64], transB = true
C: [128, 256]

// 批处理矩阵乘法
A: [32, 128, 64]
B: [32, 64, 256]
C: [32, 128, 256]
```

### 2. Transpose (转置算子)

**文件位置**: `src/operators/transpose.cc`

**主要功能**:
- 实现张量维度的转置操作
- 支持自定义维度排列顺序
- 推导输出张量的形状

**核心方法**:

#### `TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output, vector<int> permute)`
- **功能**: 构造函数，初始化转置算子
- **参数**:
  - `graph` - 计算图对象
  - `input` - 输入张量
  - `output` - 输出张量
  - `permute` - 维度排列顺序
- **算子类型**: `OpType::Transpose`
- **输入**: 1 个张量
- **输出**: 1 个张量

**转置逻辑**:
- 如果 `permute` 为空，则默认为恒等排列（不转置）
- 如果 `permute` 不为空，则按照指定的顺序重新排列维度
- `permute` 的长度必须等于输入张量的维度数

#### `optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)`
- **功能**: 推导转置后的输出形状
- **参数**: `inputs` - 输入张量向量
- **返回值**: 输出张量的形状向量
- **作业任务**: 需要实现具体的形状推导逻辑
- **参考文档**: [ONNX Transpose Operator](https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21)

**形状推导规则**:
1. 获取输入张量的形状
2. 根据 `permute` 重新排列维度
3. 输出张量的第 i 维等于输入张量的第 `permute[i]` 维

**示例**:
```cpp
// 基本转置
输入: [2, 3, 4]
permute: [2, 1, 0]
输出: [4, 3, 2]

// 部分转置
输入: [2, 3, 4]
permute: [0, 2, 1]
输出: [2, 4, 3]

// 恒等排列（不转置）
输入: [2, 3, 4]
permute: [0, 1, 2]
输出: [2, 3, 4]
```

#### `std::string TransposeObj::toString() const`
- **功能**: 将转置算子转换为字符串表示
- **返回值**: 算子的字符串描述
- **包含信息**:
  - 算子类型
  - 算子唯一标识符
  - 输入张量的形状
  - 输入张量的唯一标识符
  - 输出张量的唯一标识符

**输出格式示例**:
```
Transpose[1]([2,3,4],input=1,output=2)
```

### 3. ElementWise (元素级操作算子)

**文件位置**: `src/operators/element_wise.cc`

**主要功能**:
- 实现元素级的数学运算
- 支持广播机制
- 推导输出张量的形状

**支持的运算类型**:
- 加法 (Add)
- 减法 (Sub)
- 乘法 (Mul)
- 除法 (Div)
- 最大值 (Max)
- 最小值 (Min)
- 等等

**核心方法**:

#### `ElementWiseObj::ElementWiseObj(OpType type, GraphObj *graph, Tensor input0, Tensor input1, Tensor output)`
- **功能**: 构造函数，初始化元素级操作算子
- **参数**:
  - `type` - 算子类型（如 OpType::Add、OpType::Mul 等）
  - `graph` - 计算图对象
  - `input0` - 第一个输入张量
  - `input1` - 第二个输入张量
  - `output` - 输出张量
- **算子类型**: 由 `type` 参数指定
- **输入**: 2 个张量
- **输出**: 1 个张量

#### `optional<vector<Shape>> ElementWiseObj::inferShape(const TensorVec &inputs)`
- **功能**: 推导元素级操作后的输出形状
- **参数**: `inputs` - 输入张量向量（包含 input0 和 input1）
- **返回值**: 输出张量的形状向量
- **实现方式**: 调用 `infer_broadcast` 函数进行广播推导

**广播规则**:
1. 从右向左比较两个张量的维度
2. 如果维度大小相同，则输出该维度
3. 如果其中一个维度为 1，则输出另一个维度的大小
4. 如果两个维度大小不同且都不为 1，则无法广播
5. 如果一个张量的维度较少，则在左侧补 1 维度

**示例**:
```cpp
// 相同形状
A: [3, 4]
B: [3, 4]
输出: [3, 4]

// 广播（标量）
A: [3, 4]
B: [1, 1]
输出: [3, 4]

// 广播（向量）
A: [3, 4]
B: [1, 4]
输出: [3, 4]

// 广播（不同维度数）
A: [3, 4]
B: [4]
输出: [3, 4]

// 无法广播
A: [3, 4]
B: [2, 4]
输出: 错误
```

#### `std::string ElementWiseObj::toString() const`
- **功能**: 将元素级操作算子转换为字符串表示
- **返回值**: 算子的字符串描述
- **包含信息**:
  - 算子类型
  - 算子唯一标识符
  - 输入张量的形状
  - 输入张量的唯一标识符
  - 输出张量的唯一标识符

**输出格式示例**:
```
Add[1]([3,4],[3,4],input0=1,input1=2,output=3)
```

### 4. Concat (拼接算子)

**文件位置**: `src/operators/concat.cc`

**主要功能**:
- 实现多个张量的拼接操作
- 支持指定拼接维度
- 推导输出张量的形状

**核心方法**:

#### `ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)`
- **功能**: 构造函数，初始化拼接算子
- **参数**:
  - `graph` - 计算图对象
  - `inputs` - 输入张量向量
  - `output` - 输出张量
  - `_dim` - 拼接维度
- **算子类型**: `OpType::Concat`
- **输入**: 多个张量（至少 1 个）
- **输出**: 1 个张量

**拼接逻辑**:
- 调用 `get_real_axis` 函数处理负数维度索引
- 所有输入张量必须在非拼接维度上具有相同的形状
- 在拼接维度上，输出张量的大小等于所有输入张量在该维度上的大小之和

#### `optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs)`
- **功能**: 推导拼接后的输出形状
- **参数**: `inputs` - 输入张量向量
- **返回值**: 输出张量的形状向量
- **作业任务**: 需要实现具体的形状推导逻辑
- **参考文档**: [ONNX Concat Operator](https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13)

**形状推导规则**:
1. 获取第一个输入张量的形状作为基础
2. 在拼接维度上，累加所有输入张量在该维度上的大小
3. 其他维度保持不变

**示例**:
```cpp
// 在第 0 维拼接
输入: [[1, 2], [3, 4], [5, 6]]
dim: 0
输出: [[1, 2, 3, 4, 5, 6]]

// 在第 1 维拼接
输入: [[1, 2], [3, 4], [5, 6]]
dim: 1
输出: [[1, 2, 3], [4, 5, 6]]

// 多个张量拼接
输入: [[1, 2], [3, 4]], [[5, 6], [7, 8]]
dim: 0
输出: [[1, 2, 5, 6], [3, 4, 7, 8]]
```

#### `std::string ConcatObj::toString() const`
- **功能**: 将拼接算子转换为字符串表示
- **返回值**: 算子的字符串描述
- **包含信息**:
  - 算子类型
  - 算子唯一标识符
  - 所有输入张量的形状
  - 拼接维度
  - 输入张量的唯一标识符
  - 输出张量的唯一标识符

**输出格式示例**:
```
Concat[1]([2,3],[2,3],[2,3],dim=0,input=1,2,3,output=4)
```

### 5. Unary (一元操作算子)

**文件位置**: `src/operators/unary.cc`

**主要功能**:
- 实现对单个张量的一元操作
- 支持各种数学函数
- 推导输出张量的形状

**支持的运算类型**:
- 绝对值 (Abs)
- 平方根 (Sqrt)
- 指数 (Exp)
- 对数 (Log)
- 正弦 (Sin)
- 余弦 (Cos)
- 正切 (Tan)
- 等等

**核心方法**:

#### `UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)`
- **功能**: 构造函数，初始化一元操作算子
- **参数**:
  - `type` - 算子类型（如 OpType::Abs、OpType::Sqrt 等）
  - `graph` - 计算图对象
  - `input` - 输入张量
  - `output` - 输出张量
- **算子类型**: 由 `type` 参数指定
- **输入**: 1 个张量
- **输出**: 1 个张量

**形状推导规则**:
- 一元操作不改变张量的形状
- 输出张量的形状与输入张量完全相同

**示例**:
```cpp
// 绝对值
输入: [-1, 2, -3, 4]
输出: [1, 2, 3, 4]

// 平方根
输入: [1, 4, 9, 16]
输出: [1, 2, 3, 4]

// 指数
输入: [0, 1, 2, 3]
输出: [1, 2.718, 7.389, 20.086]
```

## 算子间关系

```
Operator (基类)
    ├── Matmul (矩阵乘法)
    ├── Transpose (转置)
    ├── ElementWise (元素级操作)
    ├── Concat (拼接)
    └── Unary (一元操作)
```

**共同特性**:
1. 所有算子都继承自 `Operator` 基类
2. 都实现 `inferShape` 方法进行形状推导
3. 都实现 `toString` 方法提供字符串表示
4. 都通过 `checkValid` 方法验证有效性

## 设计模式

### 1. 策略模式
- 不同的算子实现不同的计算策略
- 通过 `OpType` 枚举区分不同的算子类型

### 2. 模板方法模式
- `Operator` 基类定义算子的通用接口
- 具体算子实现具体的计算逻辑

### 3. 工厂模式
- 通过算子类型创建对应的算子对象
- 支持动态添加新的算子类型

## 性能优化

### 1. 形状推导优化
- 在编译时完成形状推导
- 避免运行时形状检查

### 2. 内存访问优化
- 支持原地操作（in-place）
- 减少不必要的内存拷贝

### 3. 算子融合
- 支持算子融合，减少内存访问
- 提高计算效率

## 扩展性

### 1. 添加新的算子
1. 继承 `Operator` 基类
2. 实现构造函数
3. 实现 `inferShape` 方法
4. 实现 `toString` 方法
5. 在 `OpType` 枚举中添加新的算子类型

### 2. 添加新的运算类型
1. 在 `OpType` 枚举中添加新的类型
2. 实现对应的算子类
3. 实现对应的内核

## 作业任务总结

### 1. Matmul 形状推导
- **任务**: 实现 `MatmulObj::inferShape` 方法
- **要求**: 根据输入矩阵的形状和转置选项，推导输出矩阵的形状
- **参考**: ONNX GEMM Operator 文档

### 2. Transpose 形状推导
- **任务**: 实现 `TransposeObj::inferShape` 方法
- **要求**: 根据输入张量的形状和维度排列顺序，推导输出张量的形状
- **参考**: ONNX Transpose Operator 文档

### 3. Concat 形状推导
- **任务**: 实现 `ConcatObj::inferShape` 方法
- **要求**: 根据输入张量的形状和拼接维度，推导输出张量的形状
- **参考**: ONNX Concat Operator 文档

## 总结

Operators 模块是 TinyInfiniTensor AI 编译器的算子实现模块，提供了各种深度学习计算中常用的算子实现。该模块采用面向对象的设计，具有良好的扩展性和可维护性。通过合理的形状推导和优化策略，该模块在保证功能完整性的同时，也提供了良好的性能。

该模块中的算子都遵循统一的设计模式，使得添加新的算子变得简单和直观。同时，通过支持广播、转置、拼接等高级特性，该模块能够满足各种深度学习计算的需求。