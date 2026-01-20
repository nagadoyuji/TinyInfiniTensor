# Graph 优化规则实现文档

## 目录
1. [优化规则概述](#优化规则概述)
2. [规则1: 去除冗余的 Transpose 算子](#规则1-去除冗余的-transpose-算子)
3. [规则2: 将 Transpose 合并到 Matmul](#规则2-将-transpose-合并到-matmul)
4. [实现细节](#实现细节)
5. [示例分析](#示例分析)

---

## 优化规则概述

### 目标
通过图优化减少计算量，提高执行效率。

### 两条优化规则

1. **去除冗余算子**: 两个相邻的 Transpose 算子做相反操作时，可以全部删除
2. **合并算子**: Matmul 的输入存在 Transpose（交换最后两维）时，可以融入 Matmul 的 transA/transB 属性

---

## 规则1: 去除冗余的 Transpose 算子

### 原理

如果两个相邻的 Transpose 算子做的是相反的操作，它们的组合相当于恒等变换，可以全部删除。

### 数学原理

```
设 T1 和 T2 是两个 Transpose 算子
如果 T2(T1(x)) = x，则 T1 和 T2 是相反的操作

对于维度排列：
- T1 的 permute: [0, 1, ..., i, j, ...]
- T2 的 permute: [0, 1, ..., j, i, ...]

如果 T2(T1(x)) = x，则可以删除 T1 和 T2
```

### 实现步骤

```cpp
// 1. 遍历所有算子
for (int i = 0; i < (int)ops.size(); ++i)
{
    auto op = ops[i];
    
    // 2. 检查当前和下一个算子是否都是 Transpose
    if (i + 1 < (int)ops.size() &&
        op->getType() == OpType::Transpose &&
        ops[i + 1]->getType() == OpType::Transpose)
    {
        // 3. 获取两个 Transpose 的 permute
        auto transpose1 = as<TransposeObj>(op);
        auto transpose2 = as<TransposeObj>(ops[i + 1]);
        
        auto permute1 = transpose1->getPermute();
        auto permute2 = transpose2->getPermute();
        
        // 4. 检查是否为相反的操作
        bool isOpposite = true;
        for (size_t j = 0; j < permute1.size(); ++j)
        {
            // 如果 permute2[permute1[j]] != j，说明不是相反操作
            if (permute2[permute1[j]] != j)
            {
                isOpposite = false;
                break;
            }
        }
        
        // 5. 如果是相反操作，删除两个 Transpose
        if (isOpposite)
        {
            ops.erase(ops.begin() + i + 1);  // 删除第二个
            ops.erase(ops.begin() + i);      // 删除第一个
            --i;  // 回退索引，继续检查
        }
    }
}
```

### 示例

**优化前**:
```
Tensor A (shape: [2, 3, 4])
    │
    ▼
Transpose1 (permute: [0, 2, 1])  // 交换最后两维
    │
    ▼
Tensor B (shape: [2, 4, 3])
    │
    ▼
Transpose2 (permute: [0, 2, 1])  // 再次交换最后两维
    │
    ▼
Tensor C (shape: [2, 3, 4])
```

**优化后**:
```
Tensor A (shape: [2, 3, 4])
    │
    ▼
Tensor C (shape: [2, 3, 4])
```

**验证**:
```
permute1 = [0, 2, 1]
permute2 = [0, 2, 1]

检查:
- j=0: permute2[permute1[0]] = permute2[0] = 0 ✓
- j=1: permute2[permute1[1]] = permute2[2] = 1 ✓
- j=2: permute2[permute1[2]] = permute2[1] = 2 ✓

是相反操作，可以删除
```

---

## 规则2: 将 Transpose 合并到 Matmul

### 原理

如果 Matmul 的输入是一个 Transpose 算子，且该 Transpose 只交换最后两个维度，可以将 Transpose 的效果融入到 Matmul 的 transA 或 transB 属性中。

### 数学原理

```
原始计算:
    Transpose(A) × B = (A^T) × B

优化后:
    MatMul(A, B, transA=true) = A^T × B

两种计算等价，但优化后少了一次 Transpose 操作
```

### 关键函数: isSwapLastTwoDims

```cpp
bool GraphObj::isSwapLastTwoDims(TransposeObj *transpose)
{
    auto permute = transpose->getPermute();
    int rank = permute.size();
    
    // 1. 维度必须 >= 2
    if (rank < 2)
    {
        return false;
    }
    
    // 2. 检查除最后两个维度外的所有维度是否保持不变
    // 对于 rank 维度的张量，正常 permute 是 [0, 1, ..., rank-2, rank-1]
    // 如果只交换最后两个维度，permute 应该是 [0, 1, ..., rank-1, rank-2]
    for (int i = 0; i < rank - 2; ++i)
    {
        if (permute[i] != i)
        {
            return false;
        }
    }
    
    // 3. 检查最后两个维度是否交换
    if (permute[rank - 2] != rank - 1 || permute[rank - 1] != rank - 2)
    {
        return false;
    }
    
    return true;
}
```

### 实现步骤

```cpp
// 1. 遍历所有算子
for (int i = 0; i < (int)ops.size(); ++i)
{
    auto op = ops[i];
    
    // 2. 检查是否为 Matmul 算子
    if (op->getType() == OpType::MatMul)
    {
        auto matmul = as<MatmulObj>(op);
        auto inputs = matmul->getInputs();
        
        // 3. 检查第一个输入是否为 Transpose
        if (inputs[0] && inputs[0]->getSource())
        {
            auto source = inputs[0]->getSource();
            if (source->getType() == OpType::Transpose)
            {
                auto transpose = as<TransposeObj>(source);
                
                // 4. 检查是否只交换最后两个维度
                if (isSwapLastTwoDims(transpose))
                {
                    // 5. 将 Transpose 的效果融入到 Matmul 的 transA 中
                    bool newTransA = !matmul->getTransA();
                    matmul->setTransA(newTransA);
                    
                    // 6. 更新 Matmul 的输入为 Transpose 的输入
                    auto transposeInput = transpose->getInputs()[0];
                    matmul->inputs[0] = transposeInput;
                    
                    // 7. 移除 Transpose 算子
                    removeOperator(transpose);
                    
                    // 8. 更新张量的连接关系
                    transposeInput->removeTarget(transpose);
                    transposeInput->addTarget(matmul);
                    
                    --i;  // 回退索引，继续检查
                    continue;
                }
            }
        }
        
        // 9. 检查第二个输入是否为 Transpose（类似处理）
        if (inputs[1] && inputs[1]->getSource())
        {
            auto source = inputs[1]->getSource();
            if (source->getType() == OpType::Transpose)
            {
                auto transpose = as<TransposeObj>(source);
                
                if (isSwapLastTwoDims(transpose))
                {
                    // 将 Transpose 的效果融入到 Matmul 的 transB 中
                    bool newTransB = !matmul->getTransB();
                    matmul->setTransB(newTransB);
                    
                    // 更新 Matmul 的输入为 Transpose 的输入
                    auto transposeInput = transpose->getInputs()[0];
                    matmul->inputs[1] = transposeInput;
                    
                    // 移除 Transpose 算子
                    removeOperator(transpose);
                    
                    // 更新张量的连接关系
                    transposeInput->removeTarget(transpose);
                    transposeInput->addTarget(matmul);
                    
                    --i;
                    continue;
                }
            }
        }
    }
}
```

### 示例

**优化前**:
```
Tensor A (shape: [3, 5])
    │
    ▼
Transpose (permute: [0, 2, 1])  // 交换最后两维
    │
    ▼
Tensor A' (shape: [5, 3])
    │
    ▼
MatMul(A', B, transA=false)  // 不转置 A
    │
    ▼
Tensor C (shape: [5, 2])
```

**优化后**:
```
Tensor A (shape: [3, 5])
    │
    ▼
MatMul(A, B, transA=true)  // 转置 A
    │
    ▼
Tensor C (shape: [5, 2])
```

**验证**:
```
permute = [0, 2, 1]
rank = 3

检查:
- i=0: permute[0] = 0 ✓ (保持不变)
- i=1: permute[1] = 2 ≠ 1 ✗ (不检查，因为 i < rank-2)

最后两个维度:
- permute[1] = 2 = rank-1 ✓
- permute[2] = 1 = rank-2 ✓

是只交换最后两个维度，可以合并

优化结果:
- 原始: Transpose(A) × B (transA=false)
- 优化后: A × B (transA=true)
- 等价，但少了一次 Transpose 操作
```

---

## 实现细节

### 1. 迭代优化

使用 `while (modified)` 循环，确保所有可能的优化都被应用：

```cpp
bool modified = true;
while (modified)
{
    modified = false;
    for (int i = 0; i < (int)ops.size(); ++i)
    {
        // 尝试优化
        if (canOptimize(op))
        {
            applyOptimization(op);
            modified = true;
            --i;  // 回退索引
            continue;
        }
    }
}
```

### 2. 索引回退

当删除或修改算子后，使用 `--i` 回退索引：

```cpp
--i;  // 回退索引，继续检查新位置
continue;
```

**为什么需要回退**:
```
初始: [Op1, Op2, Op3, Op4]
i=0: 检查 Op1
i=1: 检查 Op2，删除 Op2
    删除后: [Op1, Op3, Op4]
    如果不回退，i=2，会跳过 Op3
    回退后，i=1，继续检查 Op3 ✓
```

### 3. 类型转换

使用 `as<T>()` 进行安全的类型转换：

```cpp
auto matmul = as<MatmulObj>(op);
auto transpose = as<TransposeObj>(source);
```

### 4. 连接关系更新

删除算子后，必须更新张量的连接关系：

```cpp
// 更新张量的连接关系
transposeInput->removeTarget(transpose);  // 移除对 Transpose 的引用
transposeInput->addTarget(matmul);      // 添加对 Matmul 的引用
```

---

## 示例分析

### 完整示例 1: 去除冗余 Transpose

**计算图**:
```
Tensor A (shape: [2, 3, 4])
    │
    ▼
Transpose1 (permute: [0, 2, 1])
    │
    ▼
Tensor B (shape: [2, 4, 3])
    │
    ▼
Transpose2 (permute: [0, 2, 1])
    │
    ▼
Tensor C (shape: [2, 3, 4])
    │
    ▼
MatMul(C, D)
```

**优化过程**:
```
第1轮:
  i=0: 检查 Transpose1
  i=1: 检查 Transpose2
       - Transpose1.permute = [0, 2, 1]
       - Transpose2.permute = [0, 2, 1]
       - 检查: permute2[permute1[j]] == j?
         j=0: permute2[0] = 0 ✓
         j=1: permute2[2] = 1 ✓
         j=2: permute2[1] = 2 ✓
       - 是相反操作，删除 Transpose1 和 Transpose2
       - 回退 i=0

优化后:
Tensor A (shape: [2, 3, 4])
    │
    ▼
MatMul(A, D)
```

### 完整示例 2: 合并 Transpose 到 Matmul

**计算图**:
```
Tensor A (shape: [3, 5])
    │
    ▼
Transpose (permute: [0, 2, 1])
    │
    ▼
Tensor A' (shape: [5, 3])
    │
    ▼
MatMul(A', B, transA=false)
    │
    ▼
Tensor C (shape: [5, 2])
```

**优化过程**:
```
第1轮:
  i=0: 检查 Transpose
  i=1: 检查 MatMul
       - inputs[0] = A'
       - A'.source = Transpose
       - Transpose.permute = [0, 2, 1]
       - 检查 isSwapLastTwoDims:
         rank = 3
         i=0: permute[0] = 0 ✓
         最后两维: permute[1]=2=rank-1 ✓, permute[2]=1=rank-2 ✓
       - 是只交换最后两个维度
       - 合并: newTransA = !false = true
       - 更新 MatMul.inputs[0] = A
       - 删除 Transpose
       - 更新连接: A.removeTarget(Transpose), A.addTarget(MatMul)
       - 回退 i=0

优化后:
Tensor A (shape: [3, 5])
    │
    ▼
MatMul(A, B, transA=true)
    │
    ▼
Tensor C (shape: [5, 2])
```

### 完整示例 3: 复杂优化

**计算图**:
```
Tensor A (shape: [2, 3, 5])
    │
    ▼
Transpose1 (permute: [0, 2, 1])
    │
    ▼
Tensor A' (shape: [2, 5, 3])
    │
    ▼
Transpose2 (permute: [0, 2, 1])
    │
    ▼
Tensor A'' (shape: [2, 3, 5])
    │
    ▼
MatMul(A'', B, transA=false)
```

**优化过程**:
```
第1轮:
  i=0: 检查 Transpose1
  i=1: 检查 Transpose2
       - 是相反操作，删除 Transpose1 和 Transpose2
       - 回退 i=0

优化后:
Tensor A (shape: [2, 3, 5])
    │
    ▼
MatMul(A, B, transA=false)
```

---

## 总结

### 优化效果

| 优化规则 | 优化前 | 优化后 | 效果 |
|---------|--------|--------|------|
| 去除冗余 Transpose | Transpose1 → Transpose2 | 直接连接 | 减少 1 次内存访问 |
| 合并 Transpose 到 Matmul | Transpose → MatMul | MatMul(transA=true) | 减少 1 次内存访问 |

### 实现要点

1. **迭代优化**: 使用 `while (modified)` 确保所有优化都被应用
2. **索引回退**: 删除算子后使用 `--i` 回退索引
3. **连接关系**: 删除算子后必须更新张量的连接关系
4. **类型安全**: 使用 `as<T>()` 进行安全的类型转换
5. **条件检查**: 使用 `isSwapLastTwoDims()` 精确判断 Transpose 的类型

### 优化限制

1. **只交换最后两维**: 只能合并交换最后两个维度的 Transpose
2. **相邻算子**: 只能优化相邻的算子
3. **图结构**: 优化后必须保持图的正确性（拓扑顺序、连接关系）

### 参考资源

- [ONNX Transpose 文档](https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21)
- [ONNX MatMul 文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm)
