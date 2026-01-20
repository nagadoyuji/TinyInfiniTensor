# TinyInfiniTensor 类关系图

## 目录
1. [核心类层次结构](#核心类层次结构)
2. [继承关系图](#继承关系图)
3. [组合关系图](#组合关系图)
4. [详细类说明](#详细类说明)

---

## 核心类层次结构

```
Object (抽象基类)
├── OperatorObj (运算符基类)
│   ├── ElementWiseObj (二元元素运算)
│   │   ├── AddObj
│   │   ├── SubObj
│   │   ├── MulObj
│   │   └── DivObj
│   ├── UnaryObj (一元运算)
│   │   └── ReluObj
│   ├── MatmulObj (矩阵乘法)
│   ├── ConcatObj (张量拼接)
│   ├── TransposeObj (转置)
│   ├── ClipObj (裁剪)
│   └── CastObj (类型转换)
│
└── TensorObj (张量对象)

Uid (唯一标识符基类)
├── Guid (全局唯一ID)
└── Fuid (家族唯一ID)

RuntimeObj (运行时基类)
└── NativeCpuRuntimeObj (CPU运行时实现)

Kernel (计算内核基类)
└── CpuKernelWithoutConfig (无配置CPU内核)
```

---

## 继承关系图

```
┌─────────────────────────────────────────────────────────────┐
│                        Object                                │
│  - guid: Guid                                                │
│  + toString(): string (纯虚函数)                             │
│  + print(): void                                            │
│  + getGuid(): UidBaseType                                   │
└─────────────────────────────────────────────────────────────┘
                            △
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
┌─────────────────────┐           ┌─────────────────────┐
│    OperatorObj      │           │    TensorObj        │
│  - type: OpType     │           │  - dim: int         │
│  - inputs: TensorVec │           │  - dtype: DataType  │
│  - outputs: TensorVec│           │  - targets: vector   │
│  - predecessors:     │           │  - source: WRef     │
│    vector<WRef>      │           │  - data: Blob       │
│  - successors:       │           │  - runtime: Runtime │
│    vector<WRef>      │           │  - shape: Shape     │
│                     │           │  - _size: size_t     │
│  + inferShape()     │           │  - fuid: Fuid       │
│  + checkValid()     │           │                     │
│  + clone()          │           │  + size()           │
│  + getInputs()      │           │  + getBytes()       │
│  + getOutputs()     │           │  + getDims()        │
│  + getPredecessors()│           │  + setShape()       │
│  + getSuccessors()  │           │  + getRank()        │
│  + getOpType()      │           │  + setData()        │
│                     │           │  + printData()      │
└─────────────────────┘           │  + equalData()      │
         △                        │  + getRawDataPtr()  │
         │                        │  + getDType()       │
         │                        │  + getRuntime()     │
    ┌────┴────┬────────┬──────────┴──┬─────────┐
    │         │        │              │         │
┌────────┐ ┌──────┐ ┌──────┐    ┌────────┐ ┌──────┐
│Element │ │Unary │ │Matmul│    │Concat  │ │Trans-│
│WiseObj │ │ Obj  │ │ Obj  │    │  Obj   │ │pose  │
│        │ │      │ │      │    │        │ │ Obj  │
│- dim:  │ │      │ │-trans│    │- dim:  │ │-trans-│
│  int   │ │      │ │A:bool│    │  int   │ │ pose  │
│        │ │      │ │-trans│    │        │ │vector │
└────────┘ └──────┘ │B:bool│    └────────┘ └──────┘
    △         △      │-m,n,k│
    │         │      │      │
┌───┴─┬───┬──┴┐     └──────┘
│Add  │Sub│Mul│ClipObj  CastObj
│ Obj │Obj│Obj│          │
│     │   │   │          │-castType
│     │   │   │          │
└─────┴───┴───┘          └───────┘
    │   │   │
    │   │   └──ReluObj
    │   │
    │   └──DivObj
    │
└──────┐
       │
┌──────┴──────┐
│  ClipObj    │
│  CastObj    │
└─────────────┘
```

---

## 组合关系图

```
┌─────────────────────────────────────────────────────────────┐
│                      GraphObj                                 │
│  - runtime: Runtime                                          │
│  - tensors: TensorVec                                        │
│  - ops: OpVec                                                │
│  - allocator: Allocator                                       │
│                                                              │
│  + addTensor(): Tensor                                       │
│  + addOp<T>(): Ref<T>                                        │
│  + topo_sort(): bool                                         │
│  + optimize(): void                                          │
│  + shape_infer(): void                                       │
│  + dataMalloc(): void                                        │
│  + getInputs(): TensorVec                                    │
│  + getOutputs(): TensorVec                                   │
└─────────────────────────────────────────────────────────────┘
         │
         │ 拥有
         │
    ┌────┴──────────────────────────────────────────┐
    │                                             │
┌───┴──────┐                              ┌───────┴───────┐
│ TensorObj │                              │ OperatorObj   │
│           │                              │               │
│ - targets │◄─────────────────────────────│ - inputs      │
│   : vector│        (反向引用)             │ - outputs    │
│           │                              │ - predecessors│
│ - source  │─────────────────────────────►│ - successors │
│   : WRef  │        (反向引用)             │               │
│           │                              └───────────────┘
│ - data    │                                      │
│   : Blob  │                                      │ 拥有
│           │                                      │
│ - runtime │◄─────────────────────────────────────┘
│   : Runtime│
│           │
│ - fuid    │◄──────────────────┐
│   : Fuid   │                   │
│           │                   │
└───────────┘                   │
                                │
┌───────────────────────────────┴─────────────────────────────┐
│                    RuntimeObj                                │
│  - device: Device                                           │
│  + run(graph): void (纯虚函数)                              │
│  + alloc(size): void* (纯虚函数)                            │
│  + dealloc(ptr): void (纯虚函数)                            │
└─────────────────────────────────────────────────────────────┘
         △
         │ 继承
         │
┌─────────────────────────────────────────────────────────────┐
│              NativeCpuRuntimeObj                            │
│  + getInstance(): Ref<NativeCpuRuntimeObj> (静态)         │
│  + run(graph): void                                         │
│  + alloc(size): void*                                       │
│  + dealloc(ptr): void                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Allocator                               │
│  - runtime: Runtime                                         │
│  - used: size_t                                             │
│  - peak: size_t                                             │
│  - totalSize: size_t                                        │
│  - alignment: size_t                                        │
│  - ptr: void*                                               │
│  - freeBlocks: map<size_t, size_t>                          │
│                                                              │
│  + alloc(size): size_t                                      │
│  + free(addr, size): void                                   │
│  + getPtr(): void*                                          │
│  + info(): void                                             │
└─────────────────────────────────────────────────────────────┘
         │
         │ 持有
         │
┌────────┴────────┐
│   BlobObj      │
│  - runtime: Runtime                                         │
│  - ptr: void*                                                 │
│  + getPtr<T>(): T                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 详细类说明

### 1. 基础类 (core/object.h)

#### Uid
- **类型**: 基础标识符类
- **成员变量**:
  - `uid: UidBaseType` (int)
- **成员函数**:
  - `operator UidBaseType() const`: 隐式类型转换运算符
  - `operator=(const Uid &rhs) = delete`: 禁止赋值

#### Guid (继承自 Uid)
- **类型**: 全局唯一标识符
- **特点**: 每次拷贝生成新的 ID
- **成员函数**:
  - `Guid()`: 构造函数，生成新 ID
  - `Guid(const Guid &rhs)`: 拷贝构造，生成新 ID

#### Fuid (继承自 Uid)
- **类型**: 家族唯一标识符
- **特点**: 克隆的张量共享相同的 FUID
- **成员函数**:
  - `Fuid()`: 构造函数，生成新 ID
  - `Fuid(const Fuid &fuid)`: 拷贝构造，保持相同 ID

#### Object
- **类型**: 抽象基类
- **成员变量**:
  - `guid: Guid`: 全局唯一标识符
- **成员函数**:
  - `virtual ~Object()`: 虚析构函数
  - `virtual string toString() const = 0`: 纯虚函数，获取字符串表示
  - `void print()`: 打印对象信息
  - `UidBaseType getGuid() const`: 获取 GUID

---

### 2. 数据类型 (core/data_type.h)

#### DataType
- **类型**: 数据类型枚举包装类
- **静态常量**: Undefine, Float32, UInt8, Int8, UInt16, Int16, Int32, Int64, String, Bool, Float16, Double, UInt32, UInt64, BFloat16
- **成员变量**:
  - `index: int`: 数据类型索引
- **成员函数**:
  - `bool operator==(const DataType &rhs) const`: 相等比较
  - `bool operator<(const DataType &rhs) const`: 小于比较
  - `template <typename T> static int get()`: 获取类型对应的索引
  - `size_t getSize() const`: 获取每个元素的字节数
  - `string toString() const`: 获取类型名称
  - `int cpuTypeInt() const`: 获取 CPU 类型索引
  - `int getIndex() const`: 获取索引

---

### 3. 张量 (core/tensor.h)

#### TensorObj (继承自 Object)
- **类型**: 张量对象
- **成员变量**:
  - `dim: int`: 维度
  - `dtype: DataType`: 数据类型
  - `targets: vector<WRef<OperatorObj>>`: 目标运算符（使用此张量的运算符）
  - `source: WRef<OperatorObj>`: 源运算符（产生此张量的运算符）
  - `data: Blob`: 数据块
  - `runtime: Runtime`: 运行时
  - `shape: Shape`: 张量形状
  - `_size: size_t`: 张量元素总数（缓存）
  - `fuid: Fuid`: 家族唯一 ID
- **成员函数**:
  - `TensorObj(Shape shape, DataType dtype, Runtime runtime)`: 构造函数
  - `string toString() const override`: 获取字符串表示
  - `size_t size() const`: 获取元素总数
  - `size_t getBytes() const`: 获取字节数
  - `Shape getDims() const`: 获取形状
  - `void setShape(Shape shape_)`: 设置形状
  - `size_t getRank() const`: 获取秩（维度数）
  - `UidBaseType getFuid() const`: 获取 FUID
  - `void setData(generator) const`: 设置数据
  - `void setDataBlob(const Blob &blob)`: 设置数据块
  - `void printData() const`: 打印数据
  - `bool equalData(const Tensor &rhs, double relativeError = 1e-6) const`: 比较数据
  - `template <typename T> bool equalData(const vector<T> &dataVector)`: 比较数据
  - `template <typename T> T getRawDataPtr() const`: 获取原始数据指针
  - `DataType getDType() const`: 获取数据类型
  - `Runtime getRuntime() const`: 获取运行时
  - `OpVec getTargets() const`: 获取目标运算符
  - `Operator getSource() const`: 获取源运算符

---

### 4. 运行时 (core/runtime.h)

#### Device
- **类型**: 设备枚举
- **枚举值**: CPU

#### RuntimeObj
- **类型**: 运行时抽象基类
- **继承**: `std::enable_shared_from_this<RuntimeObj>`
- **成员变量**:
  - `device: Device`: 设备类型
- **成员函数**:
  - `virtual void run(const Graph &graph) const = 0`: 运行计算图（纯虚函数）
  - `virtual void *alloc(size_t size) = 0`: 分配内存（纯虚函数）
  - `virtual void dealloc(void *ptr) = 0`: 释放内存（纯虚函数）
  - `bool isCpu() const`: 是否为 CPU
  - `virtual string toString() const = 0`: 获取字符串表示（纯虚函数）

#### NativeCpuRuntimeObj (继承自 RuntimeObj)
- **类型**: 原生 CPU 运行时实现
- **成员函数**:
  - `static Ref<NativeCpuRuntimeObj> &getInstance()`: 获取单例
  - `void dealloc(void *ptr) override`: 释放内存
  - `void run(const Graph &graph) const override`: 运行计算图
  - `void *alloc(size_t size) override`: 分配内存
  - `string toString() const override`: 获取字符串表示

---

### 5. 运算符 (core/operator.h)

#### OperatorObj (继承自 Object)
- **类型**: 运算符抽象基类
- **成员变量**:
  - `type: OpType`: 运算符类型
  - `inputs: TensorVec`: 输入张量
  - `outputs: TensorVec`: 输出张量
  - `predecessors: vector<WRef<OperatorObj>>`: 前驱运算符
  - `successors: vector<WRef<OperatorObj>>`: 后继运算符
- **成员函数**:
  - `OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs)`: 构造函数
  - `virtual optional<vector<Shape>> inferShape(const TensorVec &inputs) = 0`: 推断形状（纯虚函数）
  - `virtual vector<DataType> inferDataType(const TensorVec &inputs) const`: 推断数据类型
  - `bool checkValid(GraphObj *graph)`: 检查有效性
  - `const TensorVec &getInputs() const`: 获取输入张量
  - `const TensorVec &getOutputs() const`: 获取输出张量
  - `Tensor getInputs(size_t i) const`: 获取第 i 个输入
  - `Tensor getOutput() const`: 获取输出（单个输出）
  - `Tensor getOutput(size_t i) const`: 获取第 i 个输出
  - `OpVec getPredecessors() const`: 获取前驱运算符
  - `OpVec getSuccessors() const`: 获取后继运算符
  - `OpType getOpType() const`: 获取运算符类型
  - `DataType getDType() const`: 获取输入数据类型
  - `DataType getOutDType() const`: 获取输出数据类型
  - `virtual int numInputs() const = 0`: 输入数量（纯虚函数）
  - `virtual int numOutputs() const = 0`: 输出数量（纯虚函数）
  - `virtual Operator clone(const TensorVec &newInputs, const TensorVec &newOutputs) const = 0`: 克隆运算符（纯虚函数）

---

### 6. 计算图 (core/graph.h)

#### GraphObj (继承自 Object)
- **类型**: 计算图
- **成员变量**:
  - `runtime: Runtime`: 运行时
  - `tensors: TensorVec`: 张量列表
  - `ops: OpVec`: 运算符列表
  - `allocator: Allocator`: 内存分配器
  - `sorted: bool`: 是否已拓扑排序
- **成员函数**:
  - `GraphObj(Runtime runtime)`: 构造函数
  - `string toString() const override`: 获取字符串表示
  - `Runtime getRuntime() const`: 获取运行时
  - `Tensor addTensor(Shape dim, DataType dtype)`: 添加张量
  - `Tensor addTensor(const Tensor &tensor)`: 添加张量
  - `TensorVec addTensor(const TensorVec &tensors)`: 添加多个张量
  - `void removeOperator(Operator op)`: 移除运算符
  - `void removeTensor(Tensor tensor)`: 移除张量
  - `const TensorVec &getTensors() const`: 获取张量列表
  - `const OpVec &getOperators() const`: 获取运算符列表
  - `Tensor getTensor(int) const`: 获取张量
  - `bool topo_sort()`: 拓扑排序
  - `void optimize()`: 优化
  - `void shape_infer()`: 形状推断
  - `void dataMalloc()`: 数据内存分配
  - `template <typename T, typename... Args> Ref<T> addOp(Args &&...args)`: 添加运算符（自动创建输出）
  - `template <typename T, typename... Args> Ref<T> addOpWithOutputs(Args &&...args)`: 添加运算符（指定输出）
  - `TensorVec getInputs() const`: 获取输入张量
  - `TensorVec getOutputs() const`: 获取输出张量
  - `bool checkValid() const`: 检查有效性

---

### 7. 内存分配器 (core/allocator.h)

#### Allocator
- **类型**: 内存分配器
- **成员变量**:
  - `runtime: Runtime`: 运行时
  - `used: size_t`: 已使用内存
  - `peak: size_t`: 峰值内存
  - `totalSize: size_t`: 总大小
  - `alignment: size_t`: 对齐大小
  - `ptr: void*`: 实际分配的内存指针
  - `freeBlocks: map<size_t, size_t>`: 空闲块（起始地址 -> 大小）
- **成员函数**:
  - `Allocator(Runtime runtime)`: 构造函数
  - `virtual ~Allocator()`: 析构函数
  - `size_t alloc(size_t size)`: 分配内存（首次适应算法）
  - `void free(size_t addr, size_t size)`: 释放内存（合并相邻空闲块）
  - `void *getPtr()`: 执行实际内存分配
  - `void info()`: 打印内存信息

---

### 8. 数据块 (core/blob.h)

#### BlobObj
- **类型**: 数据块
- **成员变量**:
  - `runtime: Runtime`: 运行时
  - `ptr: void*`: 数据指针
- **成员函数**:
  - `BlobObj(Runtime runtime, void *ptr)`: 构造函数
  - `template <typename T> T getPtr() const`: 获取指针

---

### 9. 内核 (core/kernel.h)

#### Kernel
- **类型**: 计算内核抽象基类
- **成员函数**:
  - `virtual void compute(const Operator &op, const RuntimeObj *context) const = 0`: 计算（纯虚函数）

#### KernelRegistry
- **类型**: 内核注册表（单例）
- **成员变量**:
  - `kernels: map<KernelAttrs, KernelRecord>`: 内核映射
  - `nKernels: int`: 内核数量
- **成员函数**:
  - `bool registerKernel(const KernelAttrs &key, Kernel *kernel, string name)`: 注册内核
  - `Kernel *getKernel(const KernelAttrs &kernelAttrs) const`: 获取内核
  - `const KernelRecord &getKernelItem(const KernelAttrs &kernelAttrs) const`: 获取内核记录

#### CpuKernelWithoutConfig (继承自 Kernel)
- **类型**: 无配置的 CPU 内核
- **成员函数**:
  - `virtual void compute(const Operator &op, const RuntimeObj *context) const = 0`: 计算（纯虚函数）

---

### 10. 运算符类型 (core/op_type.h)

#### OpType
- **类型**: 运算符类型枚举包装类
- **枚举值**: Unknown, Add, Cast, Clip, Concat, Div, Mul, MatMul, Relu, Sub, Transpose
- **成员函数**:
  - `bool operator==(OpType others) const`: 相等比较
  - `bool operator!=(OpType others) const`: 不等比较
  - `bool operator<(OpType others) const`: 小于比较
  - `const char *toString() const`: 获取类型名称

---

### 11. 具体运算符 (operators/*.h)

#### MatmulObj (继承自 OperatorObj)
- **类型**: 矩阵乘法运算符
- **成员变量**:
  - `transA: bool`: 是否转置 A
  - `transB: bool`: 是否转置 B
  - `m, n, k: int`: 辅助属性
- **成员函数**:
  - `MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA = false, bool transB = false)`: 构造函数
  - `string toString() const override`: 获取字符串表示
  - `optional<vector<Shape>> inferShape(const TensorVec &inputs) override`: 推断形状
  - `int numInputs() const override`: 输入数量
  - `int numOutputs() const override`: 输出数量
  - `bool getTransA() const`: 获取 transA
  - `bool getTransB() const`: 获取 transB
  - `void setTransA(bool transA)`: 设置 transA
  - `void setTransB(bool transB)`: 设置 transB
  - `int getM() const`: 获取 m
  - `int getN() const`: 获取 n
  - `int getK() const`: 获取 k

#### ElementWiseObj (继承自 OperatorObj)
- **类型**: 二元元素运算基类
- **成员变量**:
  - `dim: int`: 维度
- **成员函数**:
  - `ElementWiseObj(OpType type, GraphObj *graph, Tensor input0, Tensor input1, Tensor output)`: 构造函数
  - `optional<vector<Shape>> inferShape(const TensorVec &inputs) override`: 推断形状
  - `string toString() const override`: 获取字符串表示
  - `int numInputs() const override`: 输入数量（2）
  - `int numOutputs() const override`: 输出数量（1）

**派生类**: AddObj, SubObj, MulObj, DivObj

#### UnaryObj (继承自 OperatorObj)
- **类型**: 一元运算基类
- **成员函数**:
  - `UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)`: 构造函数
  - `optional<vector<Shape>> inferShape(const TensorVec &inputs) override`: 推断形状
  - `string toString() const override`: 获取字符串表示
  - `int numInputs() const override`: 输入数量（1）
  - `int numOutputs() const override`: 输出数量（1）

**派生类**: ReluObj

#### ConcatObj (继承自 OperatorObj)
- **类型**: 张量拼接运算符
- **成员变量**:
  - `dim: int`: 拼接维度
- **成员函数**:
  - `ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int dim)`: 构造函数
  - `optional<vector<Shape>> inferShape(const TensorVec &inputs) override`: 推断形状
  - `string toString() const override`: 获取字符串表示
  - `int numInputs() const override`: 输入数量
  - `int numOutputs() const override`: 输出数量（1）
  - `int getDim() const`: 获取拼接维度

#### TransposeObj (继承自 OperatorObj)
- **类型**: 转置运算符
- **成员变量**:
  - `transposePermute: vector<int>`: 维度排列
- **成员函数**:
  - `TransposeObj(GraphObj *graph, Tensor input, Tensor output, vector<int> permute)`: 构造函数
  - `optional<vector<Shape>> inferShape(const TensorVec &inputs) override`: 推断形状
  - `string toString() const override`: 获取字符串表示
  - `int numInputs() const override`: 输入数量（1）
  - `int numOutputs() const override`: 输出数量（1）
  - `std::vector<int> getPermute() const`: 获取维度排列

#### ClipObj (继承自 OperatorObj)
- **类型**: 裁剪运算符
- **成员变量**:
  - `minValue: std::optional<float>`: 最小值
  - `maxValue: std::optional<float>`: 最大值
- **成员函数**:
  - `ClipObj(GraphObj *graph, Tensor input, Tensor output, std::optional<float> min, std::optional<float> max)`: 构造函数
  - `optional<vector<Shape>> inferShape(const TensorVec &inputs) override`: 推断形状
  - `string toString() const override`: 获取字符串表示
  - `std::optional<float> getMin() const`: 获取最小值
  - `std::optional<float> getMax() const`: 获取最大值
  - `int numInputs() const override`: 输入数量（1）
  - `int numOutputs() const override`: 输出数量（1）

#### CastObj (继承自 OperatorObj)
- **类型**: 类型转换运算符
- **成员变量**:
  - `castType: CastType`: 转换类型
- **成员函数**:
  - `CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)`: 构造函数
  - `optional<vector<Shape>> inferShape(const TensorVec &inputs) override`: 推断形状
  - `vector<DataType> inferDataType(const TensorVec &inputs) const override`: 推断数据类型
  - `string toString() const override`: 获取字符串表示
  - `CastType getType() const`: 获取转换类型
  - `DataType getOutputDataType() const`: 获取输出数据类型
  - `int numInputs() const override`: 输入数量（1）
  - `int numOutputs() const override`: 输出数量（1）

---

## 类型别名

```cpp
// 基础类型
using UidBaseType = int;
using ShapeElem = int;
using Shape = vector<ShapeElem>;

// 智能指针
using Tensor = Ref<TensorObj>;
using Operator = Ref<OperatorObj>;
using Graph = Ref<GraphObj>;
using Runtime = Ref<RuntimeObj>;
using Blob = Ref<BlobObj>;

// 容器
using TensorVec = vector<Tensor>;
using OpVec = vector<Operator>;

// 内核属性
using KernelAttrs = std::tuple<Device, OpType::underlying_t>;
```

---

## 设计模式

### 1. 单例模式
- `NativeCpuRuntimeObj::getInstance()`
- `KernelRegistry::getInstance()`

### 2. 工厂模式
- `GraphObj::addOp<T>()`: 动态创建运算符
- `KernelRegistry::getKernel()`: 获取对应的内核

### 3. 模板方法模式
- `OperatorObj::inferShape()`: 子类实现具体逻辑
- `Kernel::compute()`: 子类实现具体计算

### 4. 访问者模式
- `KernelRegistry`: 根据属性访问不同的内核实现

### 5. 观察者模式
- `TensorObj::targets` / `source`: 张量与运算符之间的双向引用

---

## 关键设计决策

1. **使用智能指针**: 使用 `std::shared_ptr` 和 `std::weak_ptr` 管理对象生命周期，避免内存泄漏
2. **引用计数**: 通过 `Ref<T>` 和 `WRef<T>` 实现自动内存管理
3. **拓扑排序**: 计算图支持拓扑排序，确保运算符按正确顺序执行
4. **内存池**: 使用 `Allocator` 实现内存池，支持内存重用和碎片整理
5. **类型安全**: 使用 `DataType` 枚举包装类，提供类型安全的数据类型操作
6. **可扩展性**: 通过继承和虚函数支持新的运算符和内核类型
