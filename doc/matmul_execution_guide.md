# Matmul 算子执行指南

## 目录
1. [Matmul 算子概述](#matmul-算子概述)
2. [基本概念](#基本概念)
3. [执行流程](#执行流程)
4. [代码示例](#代码示例)
5. [形状推断详解](#形状推断详解)
6. [转置操作](#转置操作)
7. [批量广播](#批量广播)

---

## Matmul 算子概述

### 功能
Matmul（矩阵乘法）算子执行两个矩阵的乘法运算，支持：
- **批量矩阵乘法**：支持高维张量的批量矩阵乘法
- **转置操作**：可以在计算前对输入矩阵进行转置
- **广播**：支持不同批次大小的广播

### 数学定义
对于两个矩阵 A 和 B：
```
C = A × B

其中：
- A 的形状: (m, k)
- B 的形状: (k, n)
- C 的形状: (m, n)

计算公式: C[i][j] = Σ(A[i][p] × B[p][j]), p ∈ [0, k)
```

---

## 基本概念

### 1. 矩阵维度
```
m: 输出矩阵的行数（来自 A）
n: 输出矩阵的列数（来自 B）
k: 共享维度（A 的列数 = B 的行数）
```

### 2. 转置标志
```
transA: 是否转置矩阵 A
  - false: A 保持原样，形状为 (m, k)
  - true:  A 转置，形状为 (k, m)

transB: 是否转置矩阵 B
  - false: B 保持原样，形状为 (k, n)
  - true:  B 转置，形状为 (n, k)
```

### 3. 批量维度
```
高维张量可以看作多个矩阵的集合：
- 形状 (batch, m, k): batch 个 (m, k) 矩阵
- 形状 (batch1, batch2, m, k): batch1×batch2 个 (m, k) 矩阵
```

---

## 执行流程

### 完整执行步骤

```
1. 创建计算图
   └─> GraphObj(runtime)

2. 添加输入张量
   └─> addTensor(shape, dtype)
       └─> Tensor A, Tensor B

3. 创建 Matmul 算子
   └─> addOp<MatmulObj>(A, B, C, transA, transB)
       ├─> 检查输入有效性
       ├─> 推断输出形状
       └─> 创建输出张量 C

4. 形状推断
   └─> inferShape(inputs)
       ├─> 验证输入维度
       ├─> 计算 m, n, k
       └─> 返回输出形状

5. 内存分配
   └─> dataMalloc()
       └─> 为所有张量分配内存

6. 执行计算
   └─> runtime->run(graph)
       └─> 按拓扑顺序执行算子
           └─> kernel->compute(op, runtime)
```

---

## 代码示例

### 示例 1: 基本矩阵乘法

```cpp
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"

using namespace infini;

int main() {
    // 1. 创建运行时和计算图
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    // 2. 创建输入张量
    // A: 形状 (3, 5) - 3行5列的矩阵
    auto A = g->addTensor(Shape{3, 5}, DataType::Float32);
    
    // B: 形状 (5, 2) - 5行2列的矩阵
    auto B = g->addTensor(Shape{5, 2}, DataType::Float32);

    // 3. 创建 Matmul 算子
    // 输出 C 的形状: (3, 2)
    auto matmul = g->addOp<MatmulObj>(A, B, nullptr);

    // 4. 获取输出张量
    auto C = matmul->getOutput();

    // 5. 设置输入数据
    A->setData([](void *ptr, size_t size, DataType dtype) {
        float *data = static_cast<float *>(ptr);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<float>(i);  // 0, 1, 2, ...
        }
    });

    B->setData([](void *ptr, size_t size, DataType dtype) {
        float *data = static_cast<float *>(ptr);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<float>(i + 10);  // 10, 11, 12, ...
        }
    });

    // 6. 分配内存
    g->dataMalloc();

    // 7. 执行计算
    runtime->run(g);

    // 8. 打印结果
    C->printData();

    return 0;
}
```

**形状变化**:
```
A: (3, 5)  ──┐
               │ Matmul
B: (5, 2)  ──┤
               │
               ▼
C: (3, 2)
```

---

### 示例 2: 带转置的矩阵乘法

```cpp
// 创建输入张量
auto A = g->addTensor(Shape{5, 3}, DataType::Float32);  // (5, 3)
auto B = g->addTensor(Shape{5, 2}, DataType::Float32);  // (5, 2)

// 创建 Matmul 算子，转置 A
// transA = true: A 转置为 (3, 5)
// transB = false: B 保持 (5, 2)
auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, false);

// 输出形状: (3, 2)
auto C = matmul->getOutput();
```

**形状变化**:
```
A: (5, 3)  ──┐ (transA=true) → (3, 5)
               │
B: (5, 2)  ──┤
               │
               ▼
C: (3, 2)
```

---

### 示例 3: 批量矩阵乘法

```cpp
// 创建批量输入张量
auto A = g->addTensor(Shape{2, 3, 5}, DataType::Float32);  // 2个 (3, 5) 矩阵
auto B = g->addTensor(Shape{2, 5, 2}, DataType::Float32);  // 2个 (5, 2) 矩阵

// 创建 Matmul 算子
auto matmul = g->addOp<MatmulObj>(A, B, nullptr);

// 输出形状: (2, 3, 2) - 2个 (3, 2) 矩阵
auto C = matmul->getOutput();
```

**形状变化**:
```
A: (2, 3, 5)  ──┐
                  │ Matmul (逐批)
B: (2, 5, 2)  ──┤
                  │
                  ▼
C: (2, 3, 2)
```

---

### 示例 4: 批量广播

```cpp
// 创建输入张量，批次大小不同
auto A = g->addTensor(Shape{3, 3, 5}, DataType::Float32);  // 3个 (3, 5) 矩阵
auto B = g->addTensor(Shape{1, 5, 2}, DataType::Float32);  // 1个 (5, 2) 矩阵

// 创建 Matmul 算子
// B 会广播到与 A 相同的批次大小
auto matmul = g->addOp<MatmulObj>(A, B, nullptr);

// 输出形状: (3, 3, 2) - 3个 (3, 2) 矩阵
auto C = matmul->getOutput();
```

**形状变化**:
```
A: (3, 3, 5)  ──┐
                  │ Matmul (B 广播)
B: (1, 5, 2)  ──┤
                  │
                  ▼
C: (3, 3, 2)
```

---

## 形状推断详解

### 基本规则

对于形状为 (..., m, k) 的 A 和 (..., k, n) 的 B：

1. **检查维度兼容性**:
   ```
   A 的倒数第二维 = B 的倒数第一维
   即: A.shape[-2] == B.shape[-1]
   ```

2. **计算输出形状**:
   ```
   output.shape = (..., m, n)
   其中:
   - ... 是广播后的批量维度
   - m = A.shape[-2] (如果 transA=false) 或 A.shape[-1] (如果 transA=true)
   - n = B.shape[-1] (如果 transB=false) 或 B.shape[-2] (如果 transB=true)
   ```

### 形状推断代码

```cpp
optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
    auto A = inputs[0];
    auto B = inputs[1];
    
    // 获取形状
    auto shapeA = A->getDims();
    auto shapeB = B->getDims();
    
    // 计算 m, n, k
    int m = shapeA[transA ? 0 : 1];  // A 的行数
    int n = shapeB[transB ? 1 : 0];  // B 的列数
    int k = transA ? shapeA[1] : shapeA[0];  // 共享维度
    
    // 验证兼容性
    int expected_k = transB ? shapeB[0] : shapeB[1];
    IT_ASSERT(k == expected_k, "Matmul: Input tensors must have compatible shapes.");
    
    // 构造输出形状
    Shape outputShape = shapeA;
    outputShape[shapeA.size() - 2] = m;
    outputShape[shapeA.size() - 1] = n;
    
    return {{outputShape}};
}
```

---

## 转置操作

### 转置的影响

```
transA = false:
  A 保持原样: (m, k)
  计算: C = A × B

transA = true:
  A 转置: (k, m) → (m, k)
  计算: C = A^T × B

transB = false:
  B 保持原样: (k, n)
  计算: C = A × B

transB = true:
  B 转置: (n, k) → (k, n)
  计算: C = A × B^T
```

### 转置示例

```cpp
// 示例 1: A 转置
auto A = g->addTensor(Shape{5, 3}, DataType::Float32);  // (5, 3)
auto B = g->addTensor(Shape{5, 2}, DataType::Float32);  // (5, 2)
auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, false);
// A 转置为 (3, 5)，输出: (3, 2)

// 示例 2: B 转置
auto A = g->addTensor(Shape{3, 5}, DataType::Float32);  // (3, 5)
auto B = g->addTensor(Shape{2, 5}, DataType::Float32);  // (2, 5)
auto matmul = g->addOp<MatmulObj>(A, B, nullptr, false, true);
// B 转置为 (5, 2)，输出: (3, 2)

// 示例 3: 两个都转置
auto A = g->addTensor(Shape{5, 3}, DataType::Float32);  // (5, 3)
auto B = g->addTensor(Shape{2, 5}, DataType::Float32);  // (2, 5)
auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
// A 转置为 (3, 5)，B 转置为 (5, 2)，输出: (3, 2)
```

---

## 批量广播

### 广播规则

```
1. 从右向左对齐形状
2. 对于每个维度：
   - 如果两个维度相同，保持该维度
   - 如果一个维度为 1，广播到另一个维度
   - 否则，广播失败
```

### 广播示例

```cpp
// 示例 1: 批次广播
auto A = g->addTensor(Shape{3, 3, 5}, DataType::Float32);  // (3, 3, 5)
auto B = g->addTensor(Shape{1, 5, 2}, DataType::Float32);  // (1, 5, 2)
auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
// B 广播到 (3, 5, 2)，输出: (3, 3, 2)

// 示例 2: 多维广播
auto A = g->addTensor(Shape{2, 3, 3, 5}, DataType::Float32);  // (2, 3, 3, 5)
auto B = g->addTensor(Shape{1, 3, 5, 2}, DataType::Float32);  // (1, 3, 5, 2)
auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
// B 广播到 (2, 3, 5, 2)，输出: (2, 3, 3, 2)
```

---

## 完整示例：从创建到执行

```cpp
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include <iostream>

using namespace infini;

int main() {
    // 1. 初始化运行时
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    
    // 2. 创建计算图
    Graph g = make_ref<GraphObj>(runtime);
    
    // 3. 定义输入数据
    std::vector<float> dataA = {
        1, 2, 3,  // A 的第1行
        4, 5, 6   // A 的第2行
    };  // A: (2, 3)
    
    std::vector<float> dataB = {
        7,  8,   // B 的第1行
        9,  10,  // B 的第2行
        11, 12   // B 的第3行
    };  // B: (3, 2)
    
    // 4. 创建张量
    auto A = g->addTensor(Shape{2, 3}, DataType::Float32);
    auto B = g->addTensor(Shape{3, 2}, DataType::Float32);
    
    // 5. 设置数据
    A->setData([&dataA](void *ptr, size_t size, DataType dtype) {
        std::memcpy(ptr, dataA.data(), size * sizeof(float));
    });
    
    B->setData([&dataB](void *ptr, size_t size, DataType dtype) {
        std::memcpy(ptr, dataB.data(), size * sizeof(float));
    });
    
    // 6. 创建 Matmul 算子
    auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
    auto C = matmul->getOutput();
    
    // 7. 打印形状信息
    std::cout << "A shape: " << vecToString(A->getDims()) << std::endl;
    std::cout << "B shape: " << vecToString(B->getDims()) << std::endl;
    std::cout << "C shape: " << vecToString(C->getDims()) << std::endl;
    
    // 8. 分配内存
    g->dataMalloc();
    
    // 9. 执行计算
    runtime->run(g);
    
    // 10. 打印结果
    std::cout << "\nResult C:" << std::endl;
    C->printData();
    
    return 0;
}
```

**预期输出**:
```
A shape: [2,3]
B shape: [3,2]
C shape: [2,2]

Result C:
[58, 64]
[139, 154]
```

**计算过程**:
```
C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
```

---

## 总结

### Matmul 算子的关键点

1. **形状推断**: 根据 m, n, k 计算输出形状
2. **转置支持**: 可以在计算前对输入矩阵进行转置
3. **批量操作**: 支持高维张量的批量矩阵乘法
4. **广播机制**: 支持不同批次大小的广播
5. **内存管理**: 通过计算图统一管理内存分配

### 执行流程

```
创建图 → 添加张量 → 创建算子 → 形状推断 → 内存分配 → 执行计算
```

### 参考资源

- [ONNX MatMul 文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm)
- [ONNX 广播规范](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md)
