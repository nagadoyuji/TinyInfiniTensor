# Kernels 模块详细文档

## 模块概述

Kernels 模块是 TinyInfiniTensor AI 编译器的内核实现模块，负责将抽象的算子转换为特定硬件平台可执行的代码。该模块采用分层设计，支持多种硬件平台，目前主要实现了 CPU 平台的内核。

## 模块结构

```
src/kernels/
└── cpu/
    ├── concat.cc          # 拼接算子的 CPU 内核实现
    ├── element_wise.cc    # 元素级操作算子的 CPU 内核实现
    ├── transpose.cc       # 转置算子的 CPU 内核实现
    └── unary.cc           # 一元操作算子的 CPU 内核实现
```

## 内核架构设计

### 1. 内核基类层次

```
Kernel (抽象基类)
    ├── CpuKernelWithoutConfig (CPU 内核基类，无配置)
    └── CpuKernelWithConfig (CPU 内核基类，有配置)
```

**设计特点**:
- 采用模板方法模式
- 提供统一的内核接口
- 支持配置参数（可选）
- 自动处理数据类型分发

### 2. 内核注册机制

**注册宏**: `REGISTER_KERNEL(Device, OpType, KernelClass, Name)`

**功能**:
- 将内核类注册到内核注册表
- 建立设备类型、算子类型与内核类的映射关系
- 支持内核的动态查找和调用

**示例**:
```cpp
REGISTER_KERNEL(Device::CPU, OpType::Add, NativeElementWise, "addNaive_CPU");
```

## CPU 内核实现详解

### 1. NaiveConcat (拼接内核)

**文件位置**: `src/kernels/cpu/concat.cc`

**主要功能**:
- 实现多个张量的拼接操作
- 支持任意维度的拼接
- 使用 OpenMP 并行化

**核心方法**:

#### `template <typename T> void doCompute(const Operator &_op, const RuntimeObj *context) const`
- **功能**: 执行拼接操作的具体实现
- **参数**:
  - `_op` - 算子对象
  - `context` - 运行时环境
- **模板参数**: `T` - 数据类型

**实现逻辑**:
1. **获取算子和张量**:
   ```cpp
   auto op = as<ConcatObj>(_op);
   auto inputs = op->getInputs(), outputs = op->getOutputs();
   auto dim = op->getDim();
   auto output = outputs[0];
   ```

2. **计算维度偏移量**:
   - `blockOffsetInner`: 拼接维度内部的块偏移量
   - `blockOffset`: 拼接维度的块偏移量

3. **遍历所有输入张量**:
   - 计算每个输入张量在拼接维度上的偏移量
   - 计算局部块偏移量
   - 计算内部偏移量

4. **并行拷贝数据**:
   ```cpp
   #pragma omp parallel for
   for (size_t iOffset = 0; iOffset < inSize; ++iOffset) {
       auto oOffset = iOffset % localBlockOffset + innerOffset +
                      iOffset / localBlockOffset * blockOffset;
       outPtr[oOffset] = inPtr[iOffset];
   }
   ```

**性能优化**:
- 使用 OpenMP 并行化数据拷贝
- 预先计算各种偏移量，减少运行时计算
- 支持任意维度的拼接

**数据流示例**:
```
输入1: [[1, 2], [3, 4]]
输入2: [[5, 6], [7, 8]]
拼接维度: 0
输出: [[1, 2, 5, 6], [3, 4, 7, 8]]

数据流:
输入1[0,0] → 输出[0,0]
输入1[0,1] → 输出[0,1]
输入2[0,0] → 输出[0,2]
输入2[0,1] → 输出[0,3]
...
```

#### `void compute(const Operator &_op, const RuntimeObj *context) const override`
- **功能**: 内核的入口函数，负责数据类型分发
- **实现方式**: 使用宏 `CASE` 进行数据类型分发

**支持的数据类型**:
- `DataType::Float32` (索引 1)
- `DataType::UInt32` (索引 12)

**注册信息**:
```cpp
REGISTER_KERNEL(Device::CPU, OpType::Concat, NaiveConcat, "ConcatNaive_CPU");
```

### 2. NativeElementWise (元素级操作内核)

**文件位置**: `src/kernels/cpu/element_wise.cc`

**主要功能**:
- 实现元素级的数学运算
- 支持广播机制
- 支持多种运算类型

**支持的运算类型**:
- 加法 (Add)
- 减法 (Sub)
- 乘法 (Mul)
- 除法 (Div)

**核心方法**:

#### 运算函数模板
```cpp
template <typename T> static T addCompute(T val0, T val1) { return val0 + val1; }
template <typename T> static T subCompute(T val0, T val1) { return val0 - val1; }
template <typename T> static T mulCompute(T val0, T val1) { return val0 * val1; }
template <typename T> static T divCompute(T val0, T val1) { return (T)(val0 / val1); }
```

#### `template <typename T> void doCompute(const Operator &_op, const RuntimeObj *context) const`
- **功能**: 执行元素级操作的具体实现
- **参数**:
  - `_op` - 算子对象
  - `context` - 运行时环境
- **模板参数**: `T` - 数据类型

**实现逻辑**:
1. **获取数据指针**:
   ```cpp
   T *inptr0 = op->getInputs(0)->getRawDataPtr<T *>();
   T *inptr1 = op->getInputs(1)->getRawDataPtr<T *>();
   T *outptr = op->getOutput()->getRawDataPtr<T *>();
   ```

2. **处理广播**:
   - 将输入张量的形状扩展到输出张量的维度
   - 计算步长（stride）用于索引计算

3. **选择运算函数**:
   ```cpp
   T (*_doCompute)(T val0, T val1);
   switch (op->getOpType().underlying()) {
       case OpType::Add: _doCompute = addCompute<T>; break;
       case OpType::Sub: _doCompute = subCompute<T>; break;
       case OpType::Mul: _doCompute = mulCompute<T>; break;
       case OpType::Div: _doCompute = divCompute<T>; break;
   }
   ```

4. **逐元素计算**:
   ```cpp
   for (size_t i = 0; i < n; ++i) {
       auto shapeIndexC = locate_index(i, shapeC);
       auto indexA = delocate_index(shapeIndexC, a, strideA);
       auto indexB = delocate_index(shapeIndexC, b, strideB);
       outptr[i] = _doCompute(inptr0[indexA], inptr1[indexB]);
   }
   ```

**广播机制**:
- 自动处理不同形状的张量
- 使用步长计算正确的索引
- 支持标量、向量、矩阵的广播

**广播示例**:
```
输入1: [3, 4]
输入2: [1, 4]
输出: [3, 4]

数据流:
输出[0,0] = 输入1[0,0] + 输入2[0,0]
输出[0,1] = 输入1[0,1] + 输入2[0,1]
输出[1,0] = 输入1[1,0] + 输入2[0,0]  // 输入2 广播
输出[1,1] = 输入1[1,1] + 输入2[0,1]  // 输入2 广播
...
```

#### `void compute(const Operator &_op, const RuntimeObj *context) const override`
- **功能**: 内核的入口函数，负责数据类型分发
- **实现方式**: 使用宏 `CASE` 进行数据类型分发

**支持的数据类型**:
- `DataType::Float32` (索引 1)
- `DataType::UInt32` (索引 12)

**注册信息**:
```cpp
REGISTER_KERNEL(Device::CPU, OpType::Add, NativeElementWise, "addNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Sub, NativeElementWise, "subNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Mul, NativeElementWise, "mulNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Div, NativeElementWise, "divNaive_CPU");
```

### 3. NaiveTranspose (转置内核)

**文件位置**: `src/kernels/cpu/transpose.cc`

**主要功能**:
- 实现张量维度的转置操作
- 支持自定义维度排列顺序
- 支持任意维度的转置

**辅助函数**:

#### `inline Shape idx2Pos(const Shape &shape, size_t idx)`
- **功能**: 将线性索引转换为多维坐标
- **参数**:
  - `shape` - 张量的形状
  - `idx` - 线性索引
- **返回值**: 多维坐标向量

**实现逻辑**:
```cpp
Shape pos = Shape(shape.size(), 0);
auto rest = idx, curDimId = shape.size() - 1;
while (rest > 0) {
    pos[curDimId] = rest % shape[curDimId];
    rest /= shape[curDimId];
    curDimId--;
}
return pos;
```

**示例**:
```cpp
shape = [2, 3, 4]
idx = 5
pos = [0, 1, 1]  // 0*3*4 + 1*4 + 1 = 5
```

**核心方法**:

#### `template <typename T> void doCompute(const Operator &_op, const RuntimeObj *context) const`
- **功能**: 执行转置操作的具体实现
- **参数**:
  - `_op` - 算子对象
  - `context` - 运行时环境
- **模板参数**: `T` - 数据类型

**实现逻辑**:
1. **获取算子和张量**:
   ```cpp
   auto op = as<TransposeObj>(_op);
   auto inputs = op->getInputs(), outputs = op->getOutputs();
   const auto &inDim = inputs[0]->getDims();
   const auto &perm = op->getPermute();
   ```

2. **遍历所有输入元素**:
   ```cpp
   for (size_t inIdx = 0; inIdx < inSize; ++inIdx) {
       auto posInput = idx2Pos(inDim, inIdx);
       int outIdx = 0;
       for (size_t j = 0, jEnd = perm.size(); j < jEnd; ++j) {
           outIdx = outIdx * inDim[perm[j]] + posInput[perm[j]];
       }
       outPtr[outIdx] = inPtr[inIdx];
   }
   ```

3. **计算输出索引**:
   - 将输入索引转换为多维坐标
   - 根据转置排列重新计算输出索引
   - 将数据从输入位置拷贝到输出位置

**转置示例**:
```
输入: [[1, 2, 3], [4, 5, 6]]
形状: [2, 3]
permute: [1, 0]
输出: [[1, 4], [2, 5], [3, 6]]

数据流:
输入[0,0] → 输出[0,0]
输入[0,1] → 输出[1,0]
输入[0,2] → 输出[2,0]
输入[1,0] → 输出[0,1]
输入[1,1] → 输出[1,1]
输入[1,2] → 输出[2,1]
```

#### `void compute(const Operator &_op, const RuntimeObj *context) const override`
- **功能**: 内核的入口函数，负责数据类型分发
- **实现方式**: 使用宏 `CASE` 进行数据类型分发

**支持的数据类型**:
- `DataType::Float32` (索引 1)
- `DataType::UInt32` (索引 12)

**注册信息**:
```cpp
REGISTER_KERNEL(Device::CPU, OpType::Transpose, NaiveTranspose, "TransposeNaive_CPU");
```

### 4. NativeUnary (一元操作内核)

**文件位置**: `src/kernels/cpu/unary.cc`

**主要功能**:
- 实现对单个张量的一元操作
- 支持多种一元运算类型

**支持的运算类型**:
- ReLU (Rectified Linear Unit)

**核心方法**:

#### 运算函数模板
```cpp
template <typename T> static T reluCompute(T val) {
    return std::max(T(0), val);
}
```

#### `template <typename T> void doCompute(const Operator &_op, const RuntimeObj *context) const`
- **功能**: 执行一元操作的具体实现
- **参数**:
  - `_op` - 算子对象
  - `context` - 运行时环境
- **模板参数**: `T` - 数据类型

**实现逻辑**:
1. **获取数据指针**:
   ```cpp
   T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
   T *outptr = op->getOutput()->getRawDataPtr<T *>();
   ```

2. **选择运算函数**:
   ```cpp
   T (*_doCompute)(T val);
   switch (op->getOpType().underlying()) {
       case OpType::Relu: _doCompute = reluCompute<T>; break;
   }
   ```

3. **逐元素计算**:
   ```cpp
   for (size_t offset = 0; offset < n; offset++) {
       outptr[offset] = _doCompute(inptr[offset]);
   }
   ```

**ReLU 示例**:
```
输入: [-1, 2, -3, 4]
输出: [0, 2, 0, 4]

数据流:
输入[0] = -1 → 输出[0] = max(0, -1) = 0
输入[1] = 2  → 输出[1] = max(0, 2) = 2
输入[2] = -3 → 输出[2] = max(0, -3) = 0
输入[3] = 4  → 输出[3] = max(0, 4) = 4
```

#### `void compute(const Operator &_op, const RuntimeObj *context) const override`
- **功能**: 内核的入口函数，负责数据类型分发
- **实现方式**: 使用宏 `CASE` 进行数据类型分发

**支持的数据类型**:
- `DataType::Float32` (索引 1)
- `DataType::UInt32` (索引 12)

**注册信息**:
```cpp
REGISTER_KERNEL(Device::CPU, OpType::Relu, NativeUnary, "reluNaive_CPU");
```

### 5. Clip (裁剪内核)

**文件位置**: `src/kernels/cpu/unary.cc`

**主要功能**:
- 实现张量元素的裁剪操作
- 将元素值限制在指定范围内

**核心方法**:

#### `template <typename T> void doCompute(const Operator &_op, const RuntimeObj *context) const`
- **功能**: 执行裁剪操作的具体实现
- **参数**:
  - `_op` - 算子对象
  - `context` - 运行时环境
- **模板参数**: `T` - 数据类型

**实现逻辑**:
1. **获取数据指针和参数**:
   ```cpp
   T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
   T *outptr = op->getOutput()->getRawDataPtr<T *>();
   auto minValue = op->getMin();
   auto maxValue = op->getMax();
   ```

2. **逐元素裁剪**:
   ```cpp
   for (size_t offset = 0; offset < n; offset++) {
       auto val = *inptr++;
       *outptr++ = (minValue && val < *minValue)   ? *minValue
                   : (maxValue && val > *maxValue) ? *maxValue
                                                   : val;
   }
   ```

**裁剪示例**:
```
输入: [1, 2, 3, 4, 5]
最小值: 2
最大值: 4
输出: [2, 2, 3, 4, 4]

数据流:
输入[0] = 1 → 输出[0] = max(1, 2) = 2
输入[1] = 2 → 输出[1] = 2
输入[2] = 3 → 输出[2] = 3
输入[3] = 4 → 输出[3] = 4
输入[4] = 5 → 输出[4] = min(5, 4) = 4
```

#### `void compute(const Operator &_op, const RuntimeObj *context) const override`
- **功能**: 内核的入口函数，负责数据类型分发
- **实现方式**: 使用宏 `CASE` 进行数据类型分发

**支持的数据类型**:
- `DataType::Float32` (索引 1)
- `DataType::UInt32` (索引 12)

**注册信息**:
```cpp
REGISTER_KERNEL(Device::CPU, OpType::Clip, Clip, "Clip_CPU");
```

## 内核调用流程

### 1. 内核查找与调用

```cpp
// 在 Runtime::run 中
void NativeCpuRuntimeObj::run(const Graph &graph) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        kernel->compute(op, this);
    }
}
```

### 2. 数据类型分发

```cpp
// 在内核的 compute 方法中
void compute(const Operator &_op, const RuntimeObj *context) const override {
    int dataTypeIdx = _op->getDType().getIndex();
    switch (dataTypeIdx) {
        CASE(1); // DataType::Float32
        break;
        CASE(12); // DataType::UInt32
        break;
        default:
            IT_TODO_HALT();
    }
}
```

### 3. 模板实例化

```cpp
// CASE 宏展开
#define CASE(N) \
    case N:     \
        doCompute<DT<N>::t>(_op, context)

// 当 dataTypeIdx = 1 时
case 1:
    doCompute<float>(_op, context);
```

## 性能优化策略

### 1. 并行化
- 使用 OpenMP 进行并行计算
- 适用于元素级操作和数据拷贝

### 2. 内存访问优化
- 预先计算偏移量和步长
- 减少运行时的索引计算
- 支持连续内存访问

### 3. 模板特化
- 使用模板实现类型特化
- 避免虚函数调用开销
- 支持编译时优化

### 4. 广播优化
- 自动处理广播机制
- 使用步长计算优化索引
- 避免不必要的内存拷贝

## 扩展性

### 1. 添加新的数据类型支持
1. 在 `DataType` 枚举中添加新的数据类型
2. 在内核的 `compute` 方法中添加对应的 `CASE` 分支
3. 确保模板函数支持新的数据类型

### 2. 添加新的硬件平台支持
1. 创建新的平台目录（如 `kernels/gpu/`）
2. 实现对应的内核类
3. 注册新的内核到内核注册表

### 3. 添加新的算子内核
1. 创建新的内核类
2. 实现 `doCompute` 方法
3. 实现 `compute` 方法
4. 注册内核到内核注册表

## 内核注册表

### 注册机制
- 使用单例模式的 `KernelRegistry`
- 维护设备类型、算子类型与内核类的映射
- 支持内核的动态查找和调用

### 注册示例
```cpp
REGISTER_KERNEL(Device::CPU, OpType::Add, NativeElementWise, "addNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Sub, NativeElementWise, "subNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Mul, NativeElementWise, "mulNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Div, NativeElementWise, "divNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Concat, NaiveConcat, "ConcatNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Transpose, NaiveTranspose, "TransposeNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Relu, NativeUnary, "reluNaive_CPU");
REGISTER_KERNEL(Device::CPU, OpType::Clip, Clip, "Clip_CPU");
```

## 总结

Kernels 模块是 TinyInfiniTensor AI 编译器的内核实现模块，负责将抽象的算子转换为特定硬件平台可执行的代码。该模块采用分层设计，支持多种硬件平台，目前主要实现了 CPU 平台的内核。

该模块的主要特点包括：
1. 采用模板方法模式，提供统一的内核接口
2. 使用内核注册机制，支持内核的动态查找和调用
3. 支持多种数据类型和算子类型
4. 采用多种性能优化策略，如并行化、内存访问优化等
5. 具有良好的扩展性，便于添加新的数据类型、硬件平台和算子内核

通过合理的架构设计和优化策略，该模块在保证功能完整性的同时，也提供了良好的性能和可扩展性。