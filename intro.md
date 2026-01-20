# TinyInfiniTensor AI编译器项目介绍

## 1. 项目结构

TinyInfiniTensor是一个轻量级AI编译器项目，采用模块化设计，主要包含以下核心目录和文件：

### 1.1 核心目录结构

```
TinyInfiniTensor/
├── src/
│   ├── core/              # 核心功能模块
│   │   ├── allocator.cc   # 内存分配器
│   │   ├── tensor.cc      # 张量实现
│   │   ├── runtime.cc     # 运行时环境
│   │   ├── graph.cc       # 计算图管理
│   │   └── operator.cc    # 算子基类
│   ├── operators/         # 具体算子实现
│   │   ├── matmul.cc      # 矩阵乘法
│   │   ├── transpose.cc   # 转置操作
│   │   ├── element_wise.cc# 元素级操作
│   │   ├── concat.cc      # 拼接操作
│   │   └── unary.cc       # 一元操作
│   ├── kernels/           # 硬件平台内核实现
│   │   └── cpu/           # CPU内核
│   └── utils/             # 工具函数
├── 3rd-party/             # 第三方依赖
│   └── googletest/        # Google测试框架
└── .github/               # CI/CD配置
```

### 1.2 核心模块说明

#### 1.2.1 核心功能模块 (core/)

- **allocator.cc**：实现内存分配器，负责管理计算图中张量的内存分配与回收
- **tensor.cc**：定义张量数据结构，存储计算数据和形状信息
- **runtime.cc**：实现运行时环境，负责在硬件上执行计算
- **graph.cc**：管理计算图，包括算子和张量的组织、拓扑排序、优化等
- **operator.cc**：定义算子基类，提供算子的通用接口和功能

#### 1.2.2 算子模块 (operators/)

- **matmul.cc**：矩阵乘法算子，实现矩阵之间的乘法运算
- **transpose.cc**：转置算子，实现张量维度的转置操作
- **element_wise.cc**：元素级操作算子，实现逐元素的计算
- **concat.cc**：拼接算子，实现张量的拼接操作
- **unary.cc**：一元操作算子，实现单输入的操作

#### 1.2.3 内核模块 (kernels/)

- **cpu/**：CPU平台的内核实现，将算子转换为CPU可执行的代码

## 2. 项目功能与架构

### 2.1 整体架构

TinyInfiniTensor采用经典的AI编译器架构，主要包含以下层次：

1. **前端**：负责构建计算图，添加算子和张量
2. **中间表示**：计算图作为中间表示，包含算子和张量的依赖关系
3. **图优化**：对计算图进行优化，提高执行效率
4. **内存分配**：为计算图分配内存，优化内存使用
5. **后端**：将优化后的计算图转换为特定硬件的执行代码

### 2.2 核心功能流程

1. **计算图构建**：用户通过API构建计算图，添加张量和算子
2. **形状推导**：根据输入张量的形状，推导出输出张量的形状
3. **拓扑排序**：对计算图中的算子进行拓扑排序，确定执行顺序
4. **图优化**：应用各种优化规则，优化计算图结构
5. **内存分配**：为计算图中的张量分配内存，优化内存使用
6. **执行**：按照拓扑顺序执行算子，完成计算任务

### 2.3 关键类与接口

#### 2.3.1 Allocator类

```cpp
class Allocator {
public:
    Allocator(Runtime runtime);
    ~Allocator();
    size_t alloc(size_t size);  // 分配内存
    void free(size_t addr, size_t size);  // 回收内存
    void *getPtr();  // 获取分配的内存指针
    void info();  // 显示内存使用情况
};
```

负责管理计算图中张量的内存分配与回收，实现高效的内存管理算法。

#### 2.3.2 Tensor类

```cpp
class TensorObj {
public:
    TensorObj(Shape shape_, DataType dtype, Runtime runtime);
    string toString() const;  // 转换为字符串表示
    void setShape(Shape shape_);  // 设置张量形状
    void printData() const;  // 打印张量数据
    void setDataBlob(const Blob &blob);  // 设置数据块
};
```

定义张量数据结构，存储计算数据和形状信息，提供数据访问和操作接口。

#### 2.3.3 Runtime类

```cpp
class NativeCpuRuntimeObj : public RuntimeObj {
public:
    void run(const Graph &graph) const;  // 执行计算图
    string toString() const;  // 转换为字符串表示
    void *alloc(size_t size);  // 分配内存
    void dealloc(void *ptr);  // 回收内存
};
```

实现运行时环境，负责在硬件上执行计算，提供内存分配和回收接口。

#### 2.3.4 Graph类

```cpp
class GraphObj {
public:
    void addOperatorAndConnect(const Operator &op);  // 添加算子并连接
    string toString() const;  // 转换为字符串表示
    bool topo_sort();  // 拓扑排序
    void optimize();  // 图优化
    void shape_infer();  // 形状推导
    void dataMalloc();  // 内存分配
    Tensor addTensor(Shape dim, DataType dtype);  // 添加张量
};
```

管理计算图，包括算子和张量的组织、拓扑排序、优化等功能。

#### 2.3.5 Operator类

```cpp
class OperatorObj {
public:
    OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs);
    void removePredecessors(const Operator &op);  // 移除前驱算子
    void removeSuccessors(const Operator &op);  // 移除后继算子
    void replaceInput(Tensor t1, Tensor t2);  // 替换输入张量
    optional<vector<Shape>> inferShape(const TensorVec &inputs);  // 形状推导
};
```

定义算子基类，提供算子的通用接口和功能，具体算子继承该类实现特定功能。

## 3. 作业完成思路

### 3.1 Allocator内存分配算法实现

**问题分析**：设计一个高效的内存分配算法，管理计算图中张量的内存分配与回收。

**实现思路**：
1. 使用空闲链表或伙伴系统管理内存
2. 考虑内存对齐要求（默认sizeof(uint64_t)）
3. 记录内存使用情况（已用内存、峰值内存）
4. 实现alloc()函数分配内存，返回起始地址偏移量
5. 实现free()函数回收内存，将内存块重新加入空闲链表

**关键代码点**：
- allocator.cc:26-37：alloc函数实现
- allocator.cc:39-47：free函数实现

### 3.2 计算图优化实现

**问题分析**：实现图优化规则，提高计算效率。

**实现思路**：
1. **去除冗余算子**：识别并删除相邻的相反操作（如两个相邻的transpose算子）
2. **合并算子**：将可合并的算子合并为一个（如将transpose融入matmul的属性中）
3. 遍历计算图，应用优化规则
4. 更新算子和张量的依赖关系

**关键代码点**：
- graph.cc:101-109：optimize函数实现

### 3.3 内存分配实现

**问题分析**：利用allocator给计算图分配内存，优化内存使用。

**实现思路**：
1. 拓扑排序计算图，确定执行顺序
2. 分析张量的生命周期，确定内存复用策略
3. 使用allocator为张量分配内存
4. 调用tensor的setDataBlob函数绑定内存
5. 优化内存使用，减少内存占用

**关键代码点**：
- graph.cc:146-157：dataMalloc函数实现

### 3.4 矩阵乘法形状推导实现

**问题分析**：实现matmul算子的形状推导功能。

**实现思路**：
1. 根据输入张量的形状和transpose属性，确定矩阵的实际形状
2. 应用矩阵乘法的形状计算规则
3. 返回输出张量的形状

**关键代码点**：
- operators/matmul.cc:24-31：inferShape函数实现

## 4. 项目扩展方向

1. **支持更多硬件平台**：添加GPU、NPU等硬件的内核实现
2. **增加更多算子**：支持更多AI计算中常用的算子
3. **增强图优化**：添加更多图优化规则，如常量折叠、算子融合等
4. **性能分析**：添加性能分析工具，评估计算效率
5. **自动调度**：实现自动调度功能，优化算子执行顺序

## 5. 总结

TinyInfiniTensor是一个轻量级的AI编译器项目，采用模块化设计，包含内存分配器、张量、运行时环境、计算图和算子等核心模块。项目提供了完整的AI编译流程，包括计算图构建、形状推导、拓扑排序、图优化、内存分配和执行等功能。

通过完成作业，我们可以深入理解AI编译器的工作原理，包括内存管理、图优化、形状推导等关键技术。这些技术对于理解现代AI框架的实现原理和优化方法具有重要意义。

该项目为学习和研究AI编译技术提供了良好的平台，可以进一步扩展支持更多硬件平台和算子，增强图优化能力，提高计算效率。