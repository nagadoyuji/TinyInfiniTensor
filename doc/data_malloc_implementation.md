# dataMalloc() 实现文档

## 目录
1. [函数概述](#函数概述)
2. [内存管理机制](#内存管理机制)
3. [实现步骤](#实现步骤)
4. [代码详解](#代码详解)
5. [示例分析](#示例分析)

---

## 函数概述

### 功能
`dataMalloc()` 为计算图中的所有张量分配内存，并将张量与内存块绑定。

### 调用时机
在执行计算之前调用，确保所有张量都有可用的内存。

### 执行顺序
1. 拓扑排序：确保算子按正确顺序执行
2. 内存分配：为所有张量分配内存
3. 内存绑定：将张量与内存块绑定

---

## 内存管理机制

### Allocator 工作原理

```
Allocator 使用统一的内存池管理所有张量的内存：

1. alloc(size): 分配指定大小的内存，返回偏移量
   - 使用首次适应算法
   - 支持内存重用
   - 自动对齐

2. getPtr(): 获取实际分配的内存指针
   - 首次调用时真正分配内存
   - 后续调用返回相同指针

3. 内存布局:
   [张量1][张量2][张量3]...[空闲块]
   ↑       ↑       ↑
   offset1 offset2 offset3
```

### Blob 作用

```
Blob 是内存块的封装，包含：
- runtime: 运行时（用于内存管理）
- ptr: 指向实际数据的指针

张量通过 Blob 访问内存：
Tensor.data → Blob.ptr → 实际数据
```

---

## 实现步骤

### 步骤 1: 拓扑排序

```cpp
IT_ASSERT(topo_sort() == true);
```

**目的**: 确保算子按正确的依赖顺序执行。

**为什么需要拓扑排序**:
```
错误顺序:
  Op2 (依赖 Op1) → Op1 (无依赖)
  执行时 Op2 会使用未计算的 Op1 输出

正确顺序:
  Op1 (无依赖) → Op2 (依赖 Op1)
  执行时 Op1 先计算，Op2 使用正确的输入
```

### 步骤 2: 为所有张量分配内存

```cpp
std::map<Tensor, size_t> tensorOffsets;
for (auto tensor : tensors)
{
    size_t offset = allocator.alloc(tensor->getBytes());
    tensorOffsets[tensor] = offset;
}
```

**说明**:
- 遍历所有张量
- 为每个张量分配内存
- 记录每个张量的偏移量
- 使用 `map` 存储偏移量，便于后续查找

**内存分配示例**:
```
张量列表: [A, B, C, D]
大小: [24, 32, 16, 40] 字节

分配过程:
1. A: offset = 0, size = 24
2. B: offset = 24, size = 32
3. C: offset = 56, size = 16
4. D: offset = 72, size = 40

内存布局:
[  A  ][  B  ][ C ][  D  ]
 ↑      ↑      ↑     ↑
 0     24    56    72
```

### 步骤 3: 获取实际内存指针

```cpp
void *basePtr = allocator.getPtr();
```

**说明**:
- 首次调用时，`allocator` 会真正分配内存
- 后续调用返回相同的指针
- 返回的是内存池的起始地址

### 步骤 4: 为张量绑定内存

```cpp
for (auto tensor : tensors)
{
    size_t offset = tensorOffsets[tensor];
    void *tensorPtr = static_cast<char *>(basePtr) + offset;
    Blob blob = make_ref<BlobObj>(runtime, tensorPtr);
    tensor->setDataBlob(blob);
}
```

**说明**:
- 使用之前记录的偏移量
- 计算张量的实际内存地址
- 创建 Blob 对象封装内存指针
- 将 Blob 绑定到张量

**内存绑定示例**:
```
basePtr = 0x1000

张量 A (offset=0):
  tensorPtr = 0x1000 + 0 = 0x1000
  Blob.ptr = 0x1000

张量 B (offset=24):
  tensorPtr = 0x1000 + 24 = 0x1018
  Blob.ptr = 0x1018

张量 C (offset=56):
  tensorPtr = 0x1000 + 56 = 0x1038
  Blob.ptr = 0x1038
```

---

## 代码详解

### 完整实现

```cpp
void GraphObj::dataMalloc()
{
    // 1. 拓扑排序
    IT_ASSERT(topo_sort() == true);

    // 2. 为所有张量分配内存，记录偏移量
    std::map<Tensor, size_t> tensorOffsets;
    for (auto tensor : tensors)
    {
        size_t offset = allocator.alloc(tensor->getBytes());
        tensorOffsets[tensor] = offset;
    }

    // 3. 获取实际分配的内存指针
    void *basePtr = allocator.getPtr();

    // 4. 为每个张量创建 Blob 并绑定内存
    for (auto tensor : tensors)
    {
        size_t offset = tensorOffsets[tensor];
        void *tensorPtr = static_cast<char *>(basePtr) + offset;
        Blob blob = make_ref<BlobObj>(runtime, tensorPtr);
        tensor->setDataBlob(blob);
    }

    // 5. 打印内存信息
    allocator.info();
}
```

### 关键点解释

#### 1. 为什么使用 map 存储偏移量？

```cpp
std::map<Tensor, size_t> tensorOffsets;
```

**原因**:
- `allocator.alloc()` 是顺序调用的
- 需要记录每个张量的偏移量
- 使用 map 便于后续查找

**替代方案**:
```cpp
// 方案 1: 使用 vector
vector<size_t> offsets;
for (auto tensor : tensors)
{
    offsets.push_back(allocator.alloc(tensor->getBytes()));
}
// 问题: 需要按顺序访问，不够灵活

// 方案 2: 直接在张量中存储偏移量
// 问题: 需要修改 TensorObj 的设计
```

#### 2. 为什么使用 char* 进行指针运算？

```cpp
void *tensorPtr = static_cast<char *>(basePtr) + offset;
```

**原因**:
- `void*` 不能直接进行指针运算
- `char*` 的字节大小为 1，适合按字节偏移
- 计算后可以安全地转换回 `void*`

**示例**:
```cpp
void *base = 0x1000;
size_t offset = 24;

// 错误: void* 不能直接加
// void *ptr = base + offset;  // 编译错误

// 正确: 使用 char*
void *ptr = static_cast<char *>(base) + offset;  // 0x1018
```

#### 3. Blob 的作用

```cpp
Blob blob = make_ref<BlobObj>(runtime, tensorPtr);
tensor->setDataBlob(blob);
```

**作用**:
- 封装内存指针
- 关联运行时（用于内存管理）
- 提供类型安全的访问接口

**张量访问数据**:
```cpp
// 通过 Blob 访问
Blob blob = tensor->data;
float *data = blob->getPtr<float*>();

// 直接访问（等价）
float *data = tensor->getRawDataPtr<float*>();
```

---

## 示例分析

### 示例 1: 简单计算图

**计算图**:
```
Tensor A (shape: [2, 3], dtype: Float32)
    │
    ▼
MatMul(A, B)
    │
    ▼
Tensor C (shape: [2, 2], dtype: Float32)
```

**执行 dataMalloc()**:

```cpp
// 1. 拓扑排序
ops = [MatMul]  // 已经是有序的

// 2. 分配内存
tensorOffsets = {}
tensorOffsets[A] = allocator.alloc(2*3*4) = allocator.alloc(24) = 0
tensorOffsets[B] = allocator.alloc(3*2*4) = allocator.alloc(24) = 24
tensorOffsets[C] = allocator.alloc(2*2*4) = allocator.alloc(16) = 48

// 3. 获取内存指针
basePtr = 0x1000  // 首次调用，实际分配 64 字节

// 4. 绑定内存
A->setDataBlob(Blob(0x1000))
B->setDataBlob(Blob(0x1018))
C->setDataBlob(Blob(0x1030))

// 5. 打印信息
allocator.info()
// 输出: Used memory: 64, peak memory: 64
```

**内存布局**:
```
[  A  ][  B  ][  C  ]
 ↑      ↑      ↑
 0     24    48
(十进制) (十六进制)
0x1000  0x1018  0x1030
```

### 示例 2: 复杂计算图

**计算图**:
```
Tensor A (shape: [2, 3], dtype: Float32)
    │
    ▼
MatMul(A, B)
    │
    ▼
Tensor C (shape: [2, 2], dtype: Float32)
    │
    ▼
Add(C, D)
    │
    ▼
Tensor E (shape: [2, 2], dtype: Float32)
```

**执行 dataMalloc()**:

```cpp
// 1. 拓扑排序
ops = [MatMul, Add]  // 已经是有序的

// 2. 分配内存
tensorOffsets[A] = allocator.alloc(24) = 0
tensorOffsets[B] = allocator.alloc(24) = 24
tensorOffsets[C] = allocator.alloc(16) = 48
tensorOffsets[D] = allocator.alloc(16) = 64
tensorOffsets[E] = allocator.alloc(16) = 80

// 3. 获取内存指针
basePtr = 0x1000  // 实际分配 96 字节

// 4. 绑定内存
A->setDataBlob(Blob(0x1000))
B->setDataBlob(Blob(0x1018))
C->setDataBlob(Blob(0x1030))
D->setDataBlob(Blob(0x1040))
E->setDataBlob(Blob(0x1050))

// 5. 打印信息
allocator.info()
// 输出: Used memory: 96, peak memory: 96
```

**内存布局**:
```
[  A  ][  B  ][ C ][ D ][ E ]
 ↑      ↑      ↑     ↑    ↑
 0     24    48    64   80
(十进制) (十六进制)
0x1000  0x1018  0x1030 0x1040 0x1050
```

### 示例 3: 内存重用

**计算图**:
```
Tensor A (shape: [2, 3], dtype: Float32)
    │
    ▼
MatMul(A, B)
    │
    ▼
Tensor C (shape: [2, 2], dtype: Float32)
    │
    ▼
Relu(C)
    │
    ▼
Tensor D (shape: [2, 2], dtype: Float32)
```

**执行 dataMalloc()**:

```cpp
// 1. 拓扑排序
ops = [MatMul, Relu]  // 已经是有序的

// 2. 分配内存
tensorOffsets[A] = allocator.alloc(24) = 0
tensorOffsets[B] = allocator.alloc(24) = 24
tensorOffsets[C] = allocator.alloc(16) = 48
// 注意: D 的大小与 C 相同，可以重用 C 的内存
tensorOffsets[D] = allocator.alloc(16) = 48  // 重用 C 的偏移量

// 3. 获取内存指针
basePtr = 0x1000  // 实际分配 64 字节

// 4. 绑定内存
A->setDataBlob(Blob(0x1000))
B->setDataBlob(Blob(0x1018))
C->setDataBlob(Blob(0x1030))
D->setDataBlob(Blob(0x1030))  // 与 C 共享内存

// 5. 打印信息
allocator.info()
// 输出: Used memory: 48, peak memory: 64
```

**内存布局**:
```
[  A  ][  B  ][ C/D ]
 ↑      ↑      ↑
 0     24    48
(十进制) (十六进制)
0x1000  0x1018  0x1030

注意: C 和 D 共享内存，因为它们的生命周期不重叠
```

---

## 总结

### 实现要点

1. **拓扑排序**: 确保算子按正确顺序执行
2. **内存分配**: 使用 Allocator 统一管理所有张量的内存
3. **偏移记录**: 使用 map 存储每个张量的偏移量
4. **内存绑定**: 通过 Blob 将张量与内存绑定
5. **内存重用**: Allocator 自动重用释放的内存块

### 优化效果

| 特性 | 说明 |
|------|------|
| 统一管理 | 所有张量使用同一个内存池 |
| 内存重用 | 自动重用释放的内存块 |
| 对齐支持 | 自动按指定对齐分配内存 |
| 碎片整理 | 合并相邻的空闲块 |

### 参考资源

- [Allocator 实现](d:\workspace\TinyInfiniTensor\src\core\allocator.cc)
- [Tensor 数据访问](d:\workspace\TinyInfiniTensor\include\core\tensor.h)
- [Blob 封装](d:\workspace\TinyInfiniTensor\include\core\blob.h)
