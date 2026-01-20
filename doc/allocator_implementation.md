# 内存分配器作业实现

## 概述

本文档详细说明了 TinyInfiniTensor AI 编译器中内存分配器的实现，包括空闲块管理和块合并算法。

## 实现内容

### 1. 数据结构设计

在 `include/core/allocator.h` 中添加：

```cpp
// 空闲块管理数据结构
// key: 起始地址（偏移量）
// value: 块大小
std::map<size_t, size_t> freeBlocks;

// 总内存大小
size_t totalSize;
```

**设计理由：**
- 使用 `std::map` 按起始地址排序，便于查找相邻块
- `key` 为起始地址，`value` 为块大小
- 自动排序，便于合并操作

### 2. 内存分配算法

在 `src/core/allocator.cc` 的 `alloc()` 方法中实现：

```cpp
size_t Allocator::alloc(size_t size)
{
    IT_ASSERT(this->ptr == nullptr);
    // 对齐大小
    size = this->getAlignedSize(size);

    size_t addr = 0;
    bool found = false;

    // 在空闲块中查找合适的块（首次适应算法）
    for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
        if (it->second >= size) {
            // 找到合适的块
            addr = it->first;
            size_t blockSize = it->second;

            // 从空闲块中移除
            freeBlocks.erase(it);

            // 如果块比需要的大，分割剩余部分
            if (blockSize > size) {
                size_t remaining = blockSize - size;
                freeBlocks[addr + size] = remaining;
            }

            found = true;
            break;
        }
    }

    // 如果没有找到合适的块，分配新内存
    if (!found) {
        addr = this->used;
        this->used += size;
        if (this->used > this->peak) {
            this->peak = this->used;
        }
    }

    return addr;
}
```

**算法说明：**

#### 首次适应（First Fit）算法

1. **查找空闲块**：遍历空闲块列表，找到第一个足够大的块
2. **使用该块**：如果找到，使用该块
3. **分割剩余部分**：如果块比需要的大，分割剩余部分
4. **分配新内存**：如果没找到，从末尾分配新内存

**流程图：**

```
开始分配
    │
    ▼
对齐大小
    │
    ▼
遍历空闲块
    │
    ├─ 找到合适的块？
    │   │
    │   ├─ 是 → 使用该块
    │   │       │
    │   │       ├─ 需要分割？
    │   │       │   │
    │   │       │   ├─ 是 → 分割剩余部分
    │   │       │   └─ 否 → 直接使用
    │   │       │
    │   │       └─ 返回地址
    │   │
    │   └─ 否 → 继续遍历
    │
    └─ 遍历完成？
        │
        ├─ 找到 → 返回地址
        │
        └─ 未找到 → 分配新内存
            │
            └─ 返回新地址
```

**示例：**

```cpp
// 空闲块列表：{[0, 100], [200, 50], [300, 80]}
// 请求分配：60 字节

// 1. 查找第一个 >= 60 的块
// 2. 找到 [0, 100]（足够大）
// 3. 使用该块，地址 = 0
// 4. 分割剩余部分：[60, 40]
// 5. 结果：返回 0，空闲块变为 {[60, 40], [200, 50], [300, 80]}
```

### 3. 内存释放算法

在 `src/core/allocator.cc` 的 `free()` 方法中实现：

```cpp
void Allocator::free(size_t addr, size_t size)
{
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);

    // 添加释放的块到空闲块列表
    freeBlocks[addr] = size;

    // 合并相邻的空闲块
    // 1. 检查前一个块（是否有块的结束地址等于当前块的开始地址）
    auto prevIt = freeBlocks.lower_bound(addr);
    if (prevIt != freeBlocks.begin()) {
        --prevIt;
        size_t prevAddr = prevIt->first;
        size_t prevSize = prevIt->second;

        // 如果前一个块紧邻当前块
        if (prevAddr + prevSize == addr) {
            // 合并前一个块和当前块
            size_t mergedSize = prevSize + size;
            freeBlocks.erase(prevIt);
            freeBlocks.erase(addr);
            freeBlocks[prevAddr] = mergedSize;
            addr = prevAddr;
            size = mergedSize;
        }
    }

    // 2. 检查后一个块（是否有块的开始地址等于当前块的结束地址）
    auto nextIt = freeBlocks.upper_bound(addr);
    if (nextIt != freeBlocks.end()) {
        size_t nextAddr = nextIt->first;

        // 如果后一个块紧邻当前块
        if (addr + size == nextAddr) {
            // 合并当前块和后一个块
            size_t mergedSize = size + nextIt->second;
            freeBlocks.erase(nextIt);
            freeBlocks.erase(addr);
            freeBlocks[addr] = mergedSize;
        }
    }
}
```

**算法说明：**

#### 块合并算法

1. **添加到空闲列表**：将释放的块添加到空闲块列表
2. **检查前一个块**：是否有块的结束地址等于当前块的开始地址
3. **检查后一个块**：是否有块的开始地址等于当前块的结束地址
4. **合并相邻块**：如果相邻，合并为一个更大的块

**流程图：**

```
开始释放
    │
    ▼
对齐大小
    │
    ▼
添加到空闲块列表
    │
    ▼
检查前一个块
    │
    ├─ 前一个块紧邻？
    │   │
    │   ├─ 是 → 合并前一个块和当前块
    │   │       │
    │   │       └─ 更新地址和大小
    │   │
    │   └─ 否 → 继续检查
    │
    ▼
检查后一个块
    │
    ├─ 后一个块紧邻？
    │   │
    │   ├─ 是 → 合并当前块和后一个块
    │   │
    │   └─ 否 → 完成
    │
    └─ 完成
```

**示例：**

```cpp
// 空闲块列表：{[0, 100], [200, 50], [300, 80]}
// 释放块：[100, 50]

// 1. 添加到空闲块列表
//    {[0, 100], [100, 50], [200, 50], [300, 80]}

// 2. 检查前一个块 [0, 100]
//    0 + 100 == 100（紧邻）
//    合并：[0, 150]
//    {[0, 150], [200, 50], [300, 80]}

// 3. 检查后一个块 [200, 50]
//    150 != 200（不紧邻）
//    不合并

// 结果：{[0, 150], [200, 50], [300, 80]}
```

**合并多个块的示例：**

```cpp
// 空闲块列表：{[0, 100], [200, 50], [300, 80]}
// 释放块：[100, 50]

// 1. 添加到空闲块列表
//    {[0, 100], [100, 50], [200, 50], [300, 80]}

// 2. 检查前一个块 [0, 100]
//    0 + 100 == 100（紧邻）
//    合并：[0, 150]
//    {[0, 150], [200, 50], [300, 80]}

// 3. 检查后一个块 [200, 50]
//    150 != 200（不紧邻）
//    不合并

// 再释放块：[150, 50]

// 1. 添加到空闲块列表
//    {[0, 150], [150, 50], [200, 50], [300, 80]}

// 2. 检查前一个块 [0, 150]
//    0 + 150 == 150（紧邻）
//    合并：[0, 200]
//    {[0, 200], [200, 50], [300, 80]}

// 3. 检查后一个块 [200, 50]
//    200 == 200（紧邻）
//    合并：[0, 250]
//    {[0, 250], [300, 80]}

// 结果：{[0, 250], [300, 80]}
```

## 算法分析

### 首次适应（First Fit）算法

**优点：**
- ✅ 简单易实现
- ✅ 分配速度快
- ✅ 适合小内存块

**缺点：**
- ❌ 可能产生外部碎片
- ❌ 大块可能被分割

**时间复杂度：**
- 查找：O(n)，n 为空闲块数量
- 插入：O(log n)，使用 map 的有序插入

### 块合并算法

**优点：**
- ✅ 减少外部碎片
- ✅ 提高内存利用率
- ✅ 便于后续分配

**缺点：**
- ❌ 释放时需要额外时间
- ❌ 需要维护有序结构

**时间复杂度：**
- 查找相邻块：O(log n)
- 合并操作：O(log n)

## 完整示例

### 示例1：分配和释放

```cpp
Allocator allocator(runtime);

// 初始状态
// freeBlocks: {}
// used: 0

// 分配 100 字节
size_t addr1 = allocator.alloc(100);
// freeBlocks: {}
// used: 100
// addr1: 0

// 分配 50 字节
size_t addr2 = allocator.alloc(50);
// freeBlocks: {}
// used: 150
// addr2: 100

// 释放 addr1 (100 字节)
allocator.free(addr1, 100);
// freeBlocks: {[0, 100]}
// used: 150

// 分配 60 字节（重用空闲块）
size_t addr3 = allocator.alloc(60);
// freeBlocks: {[60, 40]}
// used: 150
// addr3: 0

// 释放 addr2 (50 字节)
allocator.free(addr2, 50);
// freeBlocks: {[60, 40], [100, 50]}

// 释放 addr3 (60 字节)
allocator.free(addr3, 60);
// freeBlocks: {[0, 100], [100, 50]}
// 合并后: {[0, 150]}
```

### 示例2：块分割

```cpp
// 空闲块：{[0, 100]}
// 请求：30 字节

// 1. 找到 [0, 100]（足够大）
// 2. 使用地址 0
// 3. 分割剩余部分：[30, 70]
// 4. 结果：返回 0，空闲块变为 {[30, 70]}
```

### 示例3：块合并

```cpp
// 空闲块：{[0, 100], [200, 50]}
// 释放：[100, 50]

// 1. 添加：{[0, 100], [100, 50], [200, 50]}
// 2. 检查前一个块 [0, 100]
//    0 + 100 == 100（紧邻）
//    合并：[0, 150]
// 3. 检查后一个块 [200, 50]
//    150 != 200（不紧邻）
// 4. 结果：{[0, 150], [200, 50]}
```

## 优化建议

### 1. 使用最佳适应（Best Fit）

```cpp
// 查找最小的足够大的块
auto bestIt = freeBlocks.end();
for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
    if (it->second >= size) {
        if (bestIt == freeBlocks.end() || it->second < bestIt->second) {
            bestIt = it;
        }
    }
}

if (bestIt != freeBlocks.end()) {
    // 使用 bestIt
}
```

### 2. 使用伙伴系统（Buddy System）

```cpp
// 按大小分级管理空闲块
std::map<size_t, std::list<std::pair<size_t, size_t>>> buddyBlocks;

// 每个级别存储相同大小的块
// 分配时向上取整到 2 的幂次方
// 释放时合并伙伴块
```

### 3. 使用红黑树优化

```cpp
// 使用 std::map 的红黑树特性
// 查找、插入、删除都是 O(log n)
// 自动维护有序性
```

## 测试用例

### 测试1：基本分配

```cpp
void testBasicAlloc() {
    Allocator allocator(runtime);

    size_t addr1 = allocator.alloc(100);
    ASSERT_EQ(addr1, 0);

    size_t addr2 = allocator.alloc(50);
    ASSERT_EQ(addr2, 104);  // 对齐后的大小

    allocator.free(addr1, 100);
    allocator.free(addr2, 50);
}
```

### 测试2：块分割

```cpp
void testBlockSplit() {
    Allocator allocator(runtime);

    allocator.alloc(100);
    allocator.free(0, 100);

    size_t addr = allocator.alloc(30);
    ASSERT_EQ(addr, 0);

    // 检查剩余块
    // 应该有 [32, 72]（对齐后）
}
```

### 测试3：块合并

```cpp
void testBlockMerge() {
    Allocator allocator(runtime);

    allocator.alloc(100);
    allocator.alloc(50);
    allocator.free(0, 100);
    allocator.free(104, 50);

    // 应该合并为 [0, 152]
}
```

### 测试4：重用空闲块

```cpp
void testReuseFreeBlock() {
    Allocator allocator(runtime);

    size_t addr1 = allocator.alloc(100);
    allocator.free(addr1, 100);

    size_t addr2 = allocator.alloc(50);
    ASSERT_EQ(addr2, 0);  // 重用空闲块
}
```

## 总结

本实现完成了内存分配器的核心功能：

1. **空闲块管理**：使用 `std::map` 存储空闲块
2. **首次适应算法**：快速查找合适的空闲块
3. **块分割**：大块分割为小块和剩余部分
4. **块合并**：释放时自动合并相邻的空闲块
5. **内存对齐**：确保分配的内存对齐到指定边界

这个实现简单高效，适合 AI 编译器的内存管理需求。
