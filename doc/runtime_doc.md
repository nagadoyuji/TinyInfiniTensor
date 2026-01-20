# Runtime 运行时系统

## 概述

`Runtime` 是 TinyInfiniTensor AI 编译器的**运行时系统**，负责管理计算图的执行、内存分配和设备管理。它采用**抽象基类 + 具体实现**的设计模式，支持多种硬件设备的运行时实现。

## 文件位置

- 头文件: [include/core/runtime.h](file:///d:/workspace/TinyInfiniTensor/include/core/runtime.h)
- 实现文件: [src/core/runtime.cc](file:///d:/workspace/TinyInfiniTensor/src/core/runtime.cc)

## 核心设计

### 1. 类型别名和前向声明

```cpp
class TensorObj;
class OperatorObj;
class GraphObj;
class RuntimeObj;
class BlobObj;

using Tensor = Ref<TensorObj>;
using Operator = Ref<OperatorObj>;
using Graph = Ref<GraphObj>;
using Runtime = Ref<RuntimeObj>;
using Blob = Ref<BlobObj>;

using TensorVec = vector<Tensor>;
using OpVec = vector<Operator>;
```

**作用：**
- **前向声明**：避免循环依赖，减少编译时间
- **类型别名**：简化代码，提高可读性
- **智能指针包装**：使用 `Ref`（`std::shared_ptr`）管理对象生命周期

**设计优势：**
- ✅ 解耦：头文件之间相互独立
- ✅ 简洁：使用简短的类型名
- ✅ 安全：自动内存管理

### 2. Device 设备枚举

```cpp
enum class Device
{
    CPU = 1
};
```

**作用：**
- 定义支持的硬件设备类型
- 使用 `enum class` 避免命名空间污染
- 为未来扩展 GPU、TPU 等设备预留空间

**扩展示例：**
```cpp
enum class Device {
    CPU = 1,
    CUDA = 2,
    ROCm = 3,
    Metal = 4,
    Vulkan = 5
};
```

## RuntimeObj 抽象基类

### 类定义

```cpp
class RuntimeObj : public std::enable_shared_from_this<RuntimeObj>
{
  protected:
    Device device;

  public:
    explicit RuntimeObj(Device device) : device(device) {}
    RuntimeObj(RuntimeObj &other) = delete;
    RuntimeObj &operator=(RuntimeObj const &) = delete;
    virtual ~RuntimeObj() {}

    virtual void run(const Graph &graph) const = 0;
    virtual void *alloc(size_t size) = 0;
    virtual void dealloc(void *ptr) = 0;

    bool isCpu() const { return true; }
    virtual string toString() const = 0;
};
```

### 继承 enable_shared_from_this

```cpp
class RuntimeObj : public std::enable_shared_from_this<RuntimeObj>
```

**作用：**
- 允许对象在成员函数中获取指向自身的 `shared_ptr`
- 解决"从 this 指针创建 shared_ptr"的问题

**为什么需要？**
```cpp
class RuntimeObj {
  public:
    void someMethod() {
        // 问题：如何从 this 创建 shared_ptr？
        // shared_ptr<RuntimeObj> ptr(this);  // 危险！会导致双重释放
    }
};

// 解决方案：继承 enable_shared_from_this
class RuntimeObj : public std::enable_shared_from_this<RuntimeObj> {
  public:
    void someMethod() {
        // 安全：获取指向自身的 shared_ptr
        shared_ptr<RuntimeObj> ptr = shared_from_this();
    }
};
```

**使用示例：**
```cpp
// 创建 Runtime 对象
Runtime runtime = make_ref<NativeCpuRuntimeObj>();

// 在成员函数中获取 shared_ptr
void RuntimeObj::createTensor() {
    // 获取指向自身的 shared_ptr
    Runtime self = shared_from_this();
    
    // 传递给其他函数
    Tensor tensor = make_ref<TensorObj>(self);
}
```

### 构造函数

```cpp
explicit RuntimeObj(Device device) : device(device) {}
```

**作用：**
- 初始化运行时对象，指定设备类型
- `explicit` 防止隐式转换

**使用示例：**
```cpp
// 正确：显式构造
RuntimeObj runtime(Device::CPU);

// 错误：隐式转换（被 explicit 禁止）
// RuntimeObj runtime = Device::CPU;
```

### 删除的拷贝操作

```cpp
RuntimeObj(RuntimeObj &other) = delete;
RuntimeObj &operator=(RuntimeObj const &) = delete;
```

**作用：**
- 禁止拷贝构造和拷贝赋值
- 确保运行时对象的唯一性
- 防止意外的对象复制

**为什么删除？**
```cpp
// 如果允许拷贝
RuntimeObj runtime1(Device::CPU);
RuntimeObj runtime2 = runtime1;  // 拷贝

// 问题：两个对象管理相同的资源，可能导致：
// 1. 内存重复释放
// 2. 资源状态不一致
// 3. 线程安全问题

// 解决方案：禁止拷贝，只允许移动或使用智能指针
Runtime runtime1 = make_ref<NativeCpuRuntimeObj>();
Runtime runtime2 = runtime1;  // OK：共享同一个对象
```

### 纯虚函数接口

#### 1. run - 执行计算图

```cpp
virtual void run(const Graph &graph) const = 0;
```

**作用：**
- 执行整个计算图
- `const` 修饰：执行过程不修改运行时对象状态

**实现示例（在 NativeCpuRuntimeObj 中）：**
```cpp
void NativeCpuRuntimeObj::run(const Graph &graph) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        kernel->compute(op, this);
    }
}
```

**执行流程：**
```
1. 获取内核注册表
2. 遍历计算图中的所有操作符
3. 为每个操作符查找对应的内核
4. 执行内核计算
```

#### 2. alloc - 内存分配

```cpp
virtual void *alloc(size_t size) = 0;
```

**作用：**
- 分配指定大小的内存
- 返回指向内存的指针
- 不同设备有不同的分配策略

**实现示例（在 NativeCpuRuntimeObj 中）：**
```cpp
void *NativeCpuRuntimeObj::alloc(size_t size) {
    return calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                  sizeof(uint64_t));
}
```

**对齐优化：**
```cpp
// 普通分配
void *ptr = malloc(size);

// 对齐分配（性能更好）
void *ptr = calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                   sizeof(uint64_t));

// 示例：size = 100
// (100 + 8 - 1) / 8 = 107 / 8 = 13
// 分配 13 * 8 = 104 字节（8字节对齐）
```

#### 3. dealloc - 内存释放

```cpp
virtual void dealloc(void *ptr) = 0;
```

**作用：**
- 释放之前分配的内存
- 与 `alloc` 配对使用

**实现示例（在 NativeCpuRuntimeObj 中）：**
```cpp
void NativeCpuRuntimeObj::dealloc(void *ptr) {
    return free(ptr);
}
```

### 辅助函数

#### isCpu - 检查是否为 CPU 设备

```cpp
bool isCpu() const { return true; }
```

**作用：**
- 检查运行时是否为 CPU 设备
- 当前实现总是返回 `true`（因为只有 CPU）

**未来扩展：**
```cpp
bool isCpu() const {
    return device == Device::CPU;
}

bool isCuda() const {
    return device == Device::CUDA;
}
```

#### toString - 字符串表示

```cpp
virtual string toString() const = 0;
```

**作用：**
- 返回运行时的字符串表示
- 用于调试和日志输出

**实现示例：**
```cpp
string NativeCpuRuntimeObj::toString() const {
    return "CPU Runtime";
}
```

## NativeCpuRuntimeObj CPU 运行时实现

### 类定义

```cpp
class NativeCpuRuntimeObj : public RuntimeObj
{
  public:
    NativeCpuRuntimeObj() : RuntimeObj(Device::CPU) {}

    static Ref<NativeCpuRuntimeObj> &getInstance()
    {
      static Ref<NativeCpuRuntimeObj> instance =
          make_ref<NativeCpuRuntimeObj>();
      return instance;
    }
    void dealloc(void *ptr) override;
    void run(const Graph &graph) const override;
    void *alloc(size_t size) override;
    string toString() const override;
};
```

### 单例模式

```cpp
static Ref<NativeCpuRuntimeObj> &getInstance()
{
    static Ref<NativeCpuRuntimeObj> instance =
        make_ref<NativeCpuRuntimeObj>();
    return instance;
}
```

**作用：**
- 实现单例模式，确保全局只有一个 CPU 运行时实例
- 使用 `static` 局部变量保证线程安全（C++11）
- 返回引用，避免不必要的拷贝

**单例模式的优势：**
- ✅ 全局唯一：确保整个程序使用同一个运行时
- ✅ 延迟初始化：第一次调用时才创建
- ✅ 线程安全：C++11 保证静态局部变量的初始化是线程安全的
- ✅ 自动清理：程序结束时自动销毁

**使用示例：**
```cpp
// 获取 CPU 运行时单例
Runtime runtime = NativeCpuRuntimeObj::getInstance();

// 使用运行时
runtime->run(graph);
void *ptr = runtime->alloc(1024);
runtime->dealloc(ptr);
```

**单例模式的实现原理：**
```cpp
// 第一次调用
Runtime runtime1 = NativeCpuRuntimeObj::getInstance();
// 1. 检查 instance 是否已创建
// 2. 如果没有，创建新的 NativeCpuRuntimeObj
// 3. 返回 instance 的引用

// 第二次调用
Runtime runtime2 = NativeCpuRuntimeObj::getInstance();
// 1. 检查 instance 是否已创建
// 2. 已存在，直接返回
// 3. runtime1 和 runtime2 指向同一个对象

assert(runtime1 == runtime2);  // true
```

### 构造函数

```cpp
NativeCpuRuntimeObj() : RuntimeObj(Device::CPU) {}
```

**作用：**
- 调用基类构造函数，指定设备为 CPU
- 默认构造函数，不需要参数

### 内存管理实现

#### alloc 实现

```cpp
void *NativeCpuRuntimeObj::alloc(size_t size) {
    return calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                  sizeof(uint64_t));
}
```

**详细分析：**

1. **对齐计算**
```cpp
size_t alignedSize = (size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
// 示例：
// size = 100
// alignedSize = (100 + 8 - 1) / 8 = 107 / 8 = 13
```

2. **内存分配**
```cpp
void *ptr = calloc(alignedSize, sizeof(uint64_t));
// 分配 13 * 8 = 104 字节
// calloc 会将内存初始化为 0
```

3. **为什么使用 calloc 而非 malloc？**
```cpp
// malloc：不初始化内存
void *ptr = malloc(size);  // 内存内容不确定

// calloc：初始化为 0
void *ptr = calloc(count, size);  // 内存清零

// 在 AI 计算中，初始化为 0 更安全
```

#### dealloc 实现

```cpp
void NativeCpuRuntimeObj::dealloc(void *ptr) {
    return free(ptr);
}
```

**作用：**
- 释放内存
- 与 `alloc` 配对使用

**使用示例：**
```cpp
Runtime runtime = NativeCpuRuntimeObj::getInstance();

// 分配内存
void *ptr = runtime->alloc(1024);

// 使用内存
float *data = static_cast<float*>(ptr);
data[0] = 1.0f;

// 释放内存
runtime->dealloc(ptr);
```

### 计算图执行实现

```cpp
void NativeCpuRuntimeObj::run(const Graph &graph) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        kernel->compute(op, this);
    }
}
```

**执行流程详解：**

1. **获取内核注册表**
```cpp
const auto &kernelRegistry = KernelRegistry::getInstance();
// 获取全局唯一的内核注册表单例
```

2. **遍历操作符**
```cpp
for (auto &op : graph->getOperators()) {
    // 遍历计算图中的所有操作符
}
```

3. **构建内核属性**
```cpp
auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
// device: 设备类型（CPU）
// op->getOpType().underlying(): 操作符类型的底层值
```

4. **查找内核**
```cpp
Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
// 根据设备类型和操作符类型查找对应的内核实现
```

5. **执行内核**
```cpp
kernel->compute(op, this);
// 调用内核的 compute 方法执行计算
// op: 操作符对象
// this: 运行时对象（用于内存分配等）
```

**执行流程图：**
```
计算图执行流程:

┌─────────────────────────────────────────┐
│  开始执行计算图                          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  获取内核注册表                          │
│  KernelRegistry::getInstance()           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  遍历计算图中的操作符                    │
│  for (auto &op : graph->getOperators()) │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  构建内核属性                            │
│  KernelAttrs{device, opType}            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  查找对应的内核实现                     │
│  kernelRegistry.getKernel(kernelAttrs)  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  执行内核计算                           │
│  kernel->compute(op, this)              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  还有更多操作符？                        │
│  是 → 继续遍历                           │
│  否 → 结束执行                           │
└─────────────────────────────────────────┘
```

### toString 实现

```cpp
string NativeCpuRuntimeObj::toString() const {
    return "CPU Runtime";
}
```

**作用：**
- 返回运行时的字符串表示
- 用于调试和日志

**使用示例：**
```cpp
Runtime runtime = NativeCpuRuntimeObj::getInstance();
std::cout << runtime->toString() << std::endl;  // 输出: CPU Runtime
```

## 使用示例

### 1. 基本使用

```cpp
#include "core/runtime.h"
#include "core/graph.h"

void basicUsage() {
    // 获取 CPU 运行时
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    // 创建计算图
    Graph graph = make_ref<GraphObj>(runtime);

    // 执行计算图
    runtime->run(graph);

    // 打印运行时信息
    std::cout << runtime->toString() << std::endl;
}
```

### 2. 内存管理

```cpp
void memoryManagement() {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    // 分配内存
    size_t size = 1024;
    void *ptr = runtime->alloc(size);

    // 使用内存
    float *data = static_cast<float*>(ptr);
    for (size_t i = 0; i < size / sizeof(float); ++i) {
        data[i] = static_cast<float>(i);
    }

    // 释放内存
    runtime->dealloc(ptr);
}
```

### 3. 创建张量

```cpp
void createTensor() {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    // 创建张量
    Shape shape = {2, 3, 4};
    Tensor tensor = make_ref<TensorObj>(shape, DataType::Float32, runtime);

    // 张量会自动分配内存
    float *data = tensor->getData<float>();
    data[0] = 1.0f;
}
```

### 4. 执行计算图

```cpp
void executeGraph() {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    // 创建计算图
    Graph graph = make_ref<GraphObj>(runtime);

    // 添加操作符
    Tensor input1 = make_ref<TensorObj>(Shape{2, 2}, DataType::Float32, runtime);
    Tensor input2 = make_ref<TensorObj>(Shape{2, 2}, DataType::Float32, runtime);
    Tensor output = make_ref<TensorObj>(Shape{2, 2}, DataType::Float32, runtime);

    Operator addOp = make_ref<AddObj>(TensorVec{input1, input2}, output);

    // 执行计算图
    runtime->run(graph);
}
```

### 5. 使用 shared_from_this

```cpp
class TensorObj : public Object {
  private:
    Runtime runtime;

  public:
    TensorObj(Runtime rt) : runtime(rt) {}

    void allocateMemory() {
        // 在成员函数中使用 shared_from_this
        // 如果需要传递 Runtime 给其他对象
        Runtime selfRuntime = runtime->shared_from_this();
        
        // 或者如果 TensorObj 也继承 enable_shared_from_this
        // Tensor self = shared_from_this();
    }
};
```

## 设计模式

### 1. 抽象工厂模式

```cpp
// 抽象基类定义接口
class RuntimeObj {
  public:
    virtual void *alloc(size_t size) = 0;
    virtual void dealloc(void *ptr) = 0;
};

// 具体实现
class NativeCpuRuntimeObj : public RuntimeObj {
  public:
    void *alloc(size_t size) override {
        return calloc(...);
    }
    void dealloc(void *ptr) override {
        return free(ptr);
    }
};
```

### 2. 单例模式

```cpp
class NativeCpuRuntimeObj : public RuntimeObj {
  public:
    static Ref<NativeCpuRuntimeObj> &getInstance() {
        static Ref<NativeCpuRuntimeObj> instance =
            make_ref<NativeCpuRuntimeObj>();
        return instance;
    }
};
```

### 3. 策略模式

```cpp
// 不同的运行时实现不同的内存分配策略
class CpuRuntime {
    void *alloc(size_t size) {
        return calloc(...);  // CPU 策略
    }
};

class CudaRuntime {
    void *alloc(size_t size) {
        return cudaMalloc(...);  // CUDA 策略
    }
};
```

## 内存管理

### 内存分配策略

```cpp
void *alloc(size_t size) {
    // 1. 计算对齐后的大小
    size_t alignedSize = (size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    
    // 2. 分配对齐的内存
    void *ptr = calloc(alignedSize, sizeof(uint64_t));
    
    // 3. 返回指针
    return ptr;
}
```

### 内存对齐

**为什么需要对齐？**
- 性能：CPU 访问对齐的内存更快
- 兼容性：某些指令要求内存对齐
- SIMD：向量化操作需要内存对齐

**对齐示例：**
```cpp
// 不对齐
size = 100
分配 100 字节

// 对齐到 8 字节
size = 100
alignedSize = (100 + 7) / 8 = 13
分配 13 * 8 = 104 字节

// 对齐到 16 字节
size = 100
alignedSize = (100 + 15) / 16 = 7
分配 7 * 16 = 112 字节
```

## 执行流程

### 完整的计算图执行流程

```
1. 创建运行时
   Runtime runtime = NativeCpuRuntimeObj::getInstance();

2. 创建计算图
   Graph graph = make_ref<GraphObj>(runtime);

3. 创建张量
   Tensor input = make_ref<TensorObj>(shape, dtype, runtime);
   Tensor output = make_ref<TensorObj>(shape, dtype, runtime);

4. 创建操作符
   Operator op = make_ref<AddObj>(TensorVec{input, input}, output);

5. 执行计算图
   runtime->run(graph);
   
   5.1 获取内核注册表
   5.2 遍历操作符
   5.3 查找内核
   5.4 执行内核
   5.5 重复直到所有操作符执行完毕
```

## 扩展性

### 添加新的设备运行时

```cpp
// 1. 添加设备枚举
enum class Device {
    CPU = 1,
    CUDA = 2
};

// 2. 创建新的运行时类
class CudaRuntimeObj : public RuntimeObj {
  public:
    CudaRuntimeObj() : RuntimeObj(Device::CUDA) {}

    static Ref<CudaRuntimeObj> &getInstance() {
        static Ref<CudaRuntimeObj> instance =
            make_ref<CudaRuntimeObj>();
        return instance;
    }

    void *alloc(size_t size) override {
        void *ptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }

    void dealloc(void *ptr) override {
        cudaFree(ptr);
    }

    void run(const Graph &graph) const override {
        // CUDA 特定的执行逻辑
    }

    string toString() const override {
        return "CUDA Runtime";
    }
};
```

### 使用新的运行时

```cpp
// 使用 CPU 运行时
Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
cpuRuntime->run(graph);

// 使用 CUDA 运行时
Runtime cudaRuntime = CudaRuntimeObj::getInstance();
cudaRuntime->run(graph);
```

## 最佳实践

### 1. 使用单例获取运行时

```cpp
// 推荐：使用单例
Runtime runtime = NativeCpuRuntimeObj::getInstance();

// 不推荐：直接创建
// Runtime runtime = make_ref<NativeCpuRuntimeObj>();
```

### 2. 配对使用 alloc 和 dealloc

```cpp
// 推荐：配对使用
void *ptr = runtime->alloc(size);
runtime->dealloc(ptr);

// 不推荐：忘记释放
void *ptr = runtime->alloc(size);
// 忘记调用 dealloc(ptr)
```

### 3. 使用智能指针管理内存

```cpp
// 推荐：使用智能指针
std::unique_ptr<void, std::function<void(void*)>> ptr(
    runtime->alloc(size),
    [runtime](void* p) { runtime->dealloc(p); }
);

// 不推荐：手动管理
void *ptr = runtime->alloc(size);
// ... 使用 ptr
runtime->dealloc(ptr);  // 容易忘记或出错
```

### 4. 检查设备类型

```cpp
// 推荐：检查设备类型
if (runtime->isCpu()) {
    // CPU 特定逻辑
}

// 未来扩展
if (runtime->isCuda()) {
    // CUDA 特定逻辑
}
```

## 常见问题

### Q1: 为什么要继承 enable_shared_from_this？

**A:** 允许在成员函数中安全地获取指向自身的 shared_ptr，避免双重释放问题。

### Q2: 为什么要删除拷贝构造函数？

**A:** 确保运行时对象的唯一性，防止资源管理混乱。

### Q3: 为什么要使用单例模式？

**A:** 确保全局只有一个运行时实例，简化资源管理，提高效率。

### Q4: 为什么要使用 calloc 而非 malloc？

**A:** calloc 会将内存初始化为 0，在 AI 计算中更安全。

### Q5: 如何添加新的设备支持？

**A:** 继承 RuntimeObj，实现纯虚函数，添加设备枚举值。

## 相关文件

- [include/core/graph.h](file:///d:/workspace/TinyInfiniTensor/include/core/graph.h) - 计算图定义
- [include/core/tensor.h](file:///d:/workspace/TinyInfiniTensor/include/core/tensor.h) - 张量定义
- [include/core/kernel.h](file:///d:/workspace/TinyInfiniTensor/include/core/kernel.h) - 内核定义
- [src/core/runtime.cc](file:///d:/workspace/TinyInfiniTensor/src/core/runtime.cc) - 运行时实现

## 总结

`Runtime` 系统是 TinyInfiniTensor AI 编译器的核心组件，具有以下特点：

1. **抽象设计**：通过抽象基类定义统一接口
2. **单例模式**：确保运行时对象的唯一性
3. **内存管理**：提供对齐的内存分配和释放
4. **设备抽象**：支持多种硬件设备
5. **智能指针**：使用 shared_ptr 自动管理生命周期
6. **可扩展性**：易于添加新的设备支持

这个设计为 AI 编译器提供了强大而灵活的运行时基础。
