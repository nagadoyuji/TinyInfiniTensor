# Core 模块详细文档

## 模块概述

Core 模块是 TinyInfiniTensor AI 编译器的核心基础模块，提供了整个系统的基础数据结构和核心功能。该模块包含内存管理、张量操作、运行时环境、计算图管理和算子定义等关键组件。

## 模块结构

```
src/core/
├── allocator.cc       # 内存分配器实现
├── tensor.cc          # 张量实现
├── runtime.cc         # 运行时环境实现
├── graph.cc           # 计算图管理实现
├── operator.cc        # 算子基类实现
├── data_type.cc       # 数据类型实现
└── op_type.cc         # 算子类型实现
```

## 核心组件详解

### 1. Allocator (内存分配器)

**文件位置**: `src/core/allocator.cc`

**主要功能**:
- 管理计算图中张量的内存分配与回收
- 实现高效的内存管理算法
- 支持内存对齐，优化内存访问性能
- 跟踪内存使用情况（已用内存、峰值内存）

**核心方法**:

#### `Allocator::Allocator(Runtime runtime)`
- **功能**: 构造函数，初始化内存分配器
- **参数**: `runtime` - 运行时环境对象
- **初始化内容**:
  - `used = 0`: 已使用内存为0
  - `peak = 0`: 峰值内存为0
  - `ptr = nullptr`: 内存指针为空
  - `alignment = sizeof(uint64_t)`: 内存对齐大小为8字节

#### `size_t Allocator::alloc(size_t size)`
- **功能**: 分配指定大小的内存，返回起始地址偏移量
- **参数**: `size` - 要分配的内存大小
- **返回值**: 分配的内存起始地址偏移量
- **作业任务**: 需要实现具体的内存分配算法
- **注意事项**:
  - 调用前会检查 `this->ptr == nullptr`
  - 会自动进行内存对齐处理
  - 需要更新 `used` 和 `peak` 值

#### `void Allocator::free(size_t addr, size_t size)`
- **功能**: 回收指定地址和大小的内存
- **参数**:
  - `addr` - 要回收的内存地址偏移量
  - `size` - 要回收的内存大小
- **作业任务**: 需要实现具体的内存回收算法
- **注意事项**:
  - 调用前会检查 `this->ptr == nullptr`
  - 会自动进行内存对齐处理
  - 需要更新 `used` 值

#### `void *Allocator::getPtr()`
- **功能**: 获取分配的内存指针
- **返回值**: 内存指针
- **实现逻辑**:
  - 如果指针为空，调用运行时的 `alloc` 方法分配内存
  - 打印分配信息（地址和大小）
  - 返回内存指针

#### `size_t Allocator::getAlignedSize(size_t size)`
- **功能**: 计算对齐后的内存大小
- **参数**: `size` - 原始内存大小
- **返回值**: 对齐后的内存大小
- **对齐算法**: `((size - 1) / alignment + 1) * alignment`

#### `void Allocator::info()`
- **功能**: 打印内存使用信息
- **输出内容**: 已使用内存和峰值内存

**设计思路**:
1. 使用空闲链表或伙伴系统管理内存
2. 考虑内存对齐要求（默认 `sizeof(uint64_t)`）
3. 记录内存使用情况（已用内存、峰值内存）
4. 实现高效的内存分配和回收算法

### 2. Tensor (张量)

**文件位置**: `src/core/tensor.cc`

**主要功能**:
- 存储计算数据和形状信息
- 提供数据访问和操作接口
- 支持数据打印和比较
- 管理张量的生命周期

**核心方法**:

#### `TensorObj::TensorObj(Shape shape_, DataType dtype, Runtime runtime)`
- **功能**: 构造函数，初始化张量对象
- **参数**:
  - `shape_` - 张量的形状
  - `dtype` - 张量的数据类型
  - `runtime` - 运行时环境
- **初始化内容**:
  - `dim` - 张量的维度数
  - `dtype` - 数据类型
  - `runtime` - 运行时环境
  - `shape` - 张量形状
  - `_size` - 张量元素总数（形状各维度的乘积）

#### `string TensorObj::toString() const`
- **功能**: 将张量转换为字符串表示
- **返回值**: 张量的字符串描述
- **包含信息**:
  - 张量唯一标识符（guid）
  - 张量功能标识符（fuid）
  - 张量形状
  - 数据类型
  - 运行时环境
  - 数据指针
  - 源算子和目标算子

#### `void TensorObj::setShape(Shape shape_)`
- **功能**: 设置张量的形状
- **参数**: `shape_` - 新的张量形状
- **副作用**: 更新张量的大小（`_size`）

#### `void TensorObj::printData() const`
- **功能**: 打印张量的数据
- **限制**: 仅支持 CPU 运行时
- **实现方式**: 使用宏 `TRY_PRINT` 根据数据类型选择合适的打印方法

#### `bool TensorObj::equalData(const Tensor &rhs, double relativeError) const`
- **功能**: 比较两个张量的数据是否相等
- **参数**:
  - `rhs` - 要比较的张量
  - `relativeError` - 相对误差容忍度
- **返回值**: 数据是否相等
- **限制**: 仅支持 CPU 运行时

#### `void TensorObj::setData(const std::function<void(void *, size_t, DataType)> &generator) const`
- **功能**: 使用生成器函数设置张量数据
- **参数**: `generator` - 数据生成器函数

#### `void TensorObj::setDataBlob(const Blob &blob)`
- **功能**: 设置张量的数据块
- **参数**: `blob` - 数据块对象

**数据结构**:
- `Shape`: 张量形状（各维度大小的向量）
- `DataType`: 数据类型（如 Float32、Int32 等）
- `Blob`: 数据块（包含实际的数据指针）

### 3. Runtime (运行时环境)

**文件位置**: `src/core/runtime.cc`

**主要功能**:
- 提供硬件平台的抽象接口
- 执行计算图
- 管理硬件内存分配与回收
- 注册和调用算子内核

**核心方法**:

#### `void NativeCpuRuntimeObj::run(const Graph &graph) const`
- **功能**: 执行计算图
- **参数**: `graph` - 要执行的计算图
- **执行流程**:
  1. 获取内核注册表实例
  2. 遍历计算图中的所有算子
  3. 为每个算子查找对应的内核
  4. 调用内核的 `compute` 方法执行计算

#### `string NativeCpuRuntimeObj::toString() const`
- **功能**: 返回运行时环境的字符串表示
- **返回值**: "CPU Runtime"

#### `void *NativeCpuRuntimeObj::alloc(size_t size)`
- **功能**: 分配指定大小的内存
- **参数**: `size` - 要分配的内存大小
- **返回值**: 分配的内存指针
- **实现方式**: 使用 `calloc` 分配内存，按 `uint64_t` 对齐

#### `void NativeCpuRuntimeObj::dealloc(void *ptr)`
- **功能**: 回收内存
- **参数**: `ptr` - 要回收的内存指针
- **实现方式**: 使用 `free` 回收内存

**设计特点**:
1. 提供硬件平台的抽象接口
2. 支持多种硬件平台（CPU、GPU、NPU 等）
3. 内核注册机制，支持动态加载算子内核
4. 统一的执行接口，便于扩展

### 4. Graph (计算图)

**文件位置**: `src/core/graph.cc`

**主要功能**:
- 管理计算图的结构
- 实现拓扑排序
- 提供图优化功能
- 管理张量的内存分配

**核心方法**:

#### `void GraphObj::addOperatorAndConnect(const Operator &op)`
- **功能**: 添加算子并建立连接关系
- **参数**: `op` - 要添加的算子
- **副作用**:
  - 将算子添加到计算图中
  - 建立算子与输入输出张量的连接
  - 更新算子的前驱和后继关系
  - 标记计算图需要重新排序

#### `string GraphObj::toString() const`
- **功能**: 将计算图转换为字符串表示
- **返回值**: 计算图的字符串描述
- **包含信息**:
  - 所有张量的信息
  - 所有算子的信息
  - 算子之间的依赖关系

#### `bool GraphObj::topo_sort()`
- **功能**: 对计算图进行拓扑排序
- **返回值**: 排序是否成功
- **实现算法**: Kahn 算法
- **排序规则**:
  - 按照算子的依赖关系排序
  - 确保每个算子的所有输入都已计算完成
  - 如果存在循环依赖，返回 false

#### `void GraphObj::optimize()`
- **功能**: 优化计算图
- **作业任务**: 需要实现具体的图优化算法
- **优化规则**:
  1. 去除冗余的算子（如相邻的相反操作）
  2. 合并算子（如将 transpose 融入 matmul 的属性中）

#### `void GraphObj::shape_infer()`
- **功能**: 推导计算图中所有张量的形状
- **实现流程**:
  1. 遍历计算图中的所有算子
  2. 调用每个算子的 `inferShape` 方法
  3. 更新输出张量的形状

#### `void GraphObj::dataMalloc()`
- **功能**: 为计算图中的张量分配内存
- **作业任务**: 需要实现具体的内存分配逻辑
- **实现流程**:
  1. 先进行拓扑排序
  2. 分析张量的生命周期
  3. 使用 allocator 为张量分配内存
  4. 调用 tensor 的 `setDataBlob` 函数绑定内存

#### `Tensor GraphObj::addTensor(Shape dim, DataType dtype)`
- **功能**: 添加张量到计算图
- **参数**:
  - `dim` - 张量的形状
  - `dtype` - 张量的数据类型
- **返回值**: 创建的张量对象

#### `bool GraphObj::checkValid() const`
- **功能**: 检查计算图的有效性
- **返回值**: 计算图是否有效
- **检查内容**:
  - 张量的源和目标算子关系
  - 算子的输入和输出张量关系
  - 算子的前驱和后继关系
  - 张量的功能标识符唯一性

**设计特点**:
1. 有向无环图（DAG）结构
2. 支持拓扑排序
3. 支持图优化
4. 提供有效性检查

### 5. Operator (算子基类)

**文件位置**: `src/core/operator.cc`

**主要功能**:
- 定义算子的通用接口
- 管理算子的输入输出关系
- 提供形状推导功能
- 管理算子的依赖关系

**核心方法**:

#### `OperatorObj::OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs)`
- **功能**: 构造函数，初始化算子对象
- **参数**:
  - `opType` - 算子类型
  - `inputs` - 输入张量向量
  - `outputs` - 输出张量向量

#### `void OperatorObj::removePredecessors(const Operator &op)`
- **功能**: 移除指定的前驱算子
- **参数**: `op` - 要移除的前驱算子

#### `void OperatorObj::removeSuccessors(const Operator &op)`
- **功能**: 移除指定的后继算子
- **参数**: `op` - 要移除的后继算子

#### `void OperatorObj::replaceInput(Tensor t1, Tensor t2)`
- **功能**: 替换输入张量
- **参数**:
  - `t1` - 要被替换的输入张量
  - `t2` - 新的输入张量

#### `bool OperatorObj::checkValid(GraphObj *graph)`
- **功能**: 检查算子的有效性
- **参数**: `graph` - 计算图对象
- **返回值**: 算子是否有效
- **检查内容**:
  - 形状推导是否成功
  - 输出张量的形状是否正确

**设计特点**:
1. 抽象基类，所有具体算子都继承此类
2. 提供统一的算子接口
3. 支持形状推导
4. 管理算子之间的依赖关系

## 模块间关系

```
Allocator ← Runtime ← Graph ← Operator ← Tensor
    ↓         ↓         ↓         ↓         ↓
  内存管理  硬件抽象  图管理   算子定义  数据存储
```

**依赖关系**:
1. `Tensor` 依赖于 `Runtime` 和 `Operator`
2. `Operator` 依赖于 `Tensor`
3. `Graph` 依赖于 `Operator` 和 `Tensor`
4. `Runtime` 依赖于 `Allocator`
5. `Allocator` 依赖于 `Runtime`

## 设计模式

### 1. 工厂模式
- `Runtime` 根据设备类型创建不同的运行时环境
- `KernelRegistry` 根据算子类型创建对应的内核

### 2. 观察者模式
- `Tensor` 观察其源算子和目标算子的变化
- `Graph` 观察算子和张量的变化

### 3. 策略模式
- 不同的 `Runtime` 实现不同的内存管理策略
- 不同的 `Kernel` 实现不同的计算策略

## 性能优化

### 1. 内存优化
- 使用内存池技术减少内存分配开销
- 实现内存复用，减少内存占用
- 支持内存对齐，优化内存访问性能

### 2. 计算优化
- 支持算子融合，减少内存访问
- 实现图优化，去除冗余计算
- 支持并行计算，提高计算效率

### 3. 缓存优化
- 缓存计算结果，避免重复计算
- 缓存内核对象，减少内核查找开销

## 扩展性

### 1. 支持新的数据类型
- 在 `DataType` 中添加新的数据类型定义
- 在 `Tensor` 中添加对新数据类型的支持

### 2. 支持新的硬件平台
- 继承 `Runtime` 类实现新的运行时环境
- 实现对应平台的内核

### 3. 支持新的算子
- 继承 `Operator` 类实现新的算子
- 实现对应的内核

## 总结

Core 模块是 TinyInfiniTensor AI 编译器的基础，提供了内存管理、张量操作、运行时环境、计算图管理和算子定义等核心功能。该模块采用面向对象的设计，具有良好的扩展性和可维护性。通过合理的设计模式和优化策略，该模块在保证功能完整性的同时，也提供了良好的性能。