# TinyInfiniTensor 项目文档索引

## 项目概述

TinyInfiniTensor 是一个轻量级 AI 编译器项目，采用模块化设计，提供了完整的深度学习计算图编译和执行功能。该项目包含了从底层内存管理到高层算子实现的完整技术栈。

## 文档结构

```
doc/
├── README.md                  # 本文件：文档索引
├── core_module.md            # Core 模块详细文档
├── operators_module.md        # Operators 模块详细文档
├── kernels_module.md         # Kernels 模块详细文档
└── utils_module.md           # Utils 模块详细文档
```

## 模块文档

### 1. Core 模块

**文档**: [core_module.md](core_module.md)

**概述**: Core 模块是 TinyInfiniTensor AI 编译器的核心基础模块，提供了整个系统的基础数据结构和核心功能。

**主要组件**:
- **Allocator (内存分配器)**: 管理计算图中张量的内存分配与回收
- **Tensor (张量)**: 存储计算数据和形状信息
- **Runtime (运行时环境)**: 提供硬件平台的抽象接口
- **Graph (计算图)**: 管理计算图的结构和优化
- **Operator (算子基类)**: 定义算子的通用接口

**关键特性**:
- 高效的内存管理算法
- 支持多种数据类型
- 硬件平台抽象
- 计算图优化
- 拓扑排序

**适用场景**:
- 需要深入理解编译器核心架构的开发者
- 需要扩展内存管理功能的开发者
- 需要添加新硬件平台支持的开发者

---

### 2. Operators 模块

**文档**: [operators_module.md](operators_module.md)

**概述**: Operators 模块是 TinyInfiniTensor AI 编译器的算子实现模块，提供了各种深度学习计算中常用的算子实现。

**主要算子**:
- **Matmul (矩阵乘法)**: 实现矩阵乘法运算，支持转置和批处理
- **Transpose (转置)**: 实现张量维度的转置操作
- **ElementWise (元素级操作)**: 实现元素级的数学运算，支持广播
- **Concat (拼接)**: 实现多个张量的拼接操作
- **Unary (一元操作)**: 实现对单个张量的一元操作

**关键特性**:
- 统一的算子接口
- 自动形状推导
- 支持广播机制
- 算子优化

**适用场景**:
- 需要添加新算子的开发者
- 需要理解算子形状推导的开发者
- 需要优化算子性能的开发者

---

### 3. Kernels 模块

**文档**: [kernels_module.md](kernels_module.md)

**概述**: Kernels 模块是 TinyInfiniTensor AI 编译器的内核实现模块，负责将抽象的算子转换为特定硬件平台可执行的代码。

**主要内核**:
- **NaiveConcat (拼接内核)**: 实现拼接操作的 CPU 内核
- **NativeElementWise (元素级操作内核)**: 实现元素级操作的 CPU 内核
- **NaiveTranspose (转置内核)**: 实现转置操作的 CPU 内核
- **NativeUnary (一元操作内核)**: 实现一元操作的 CPU 内核
- **Clip (裁剪内核)**: 实现裁剪操作的 CPU 内核

**关键特性**:
- 硬件平台抽象
- 内核注册机制
- 数据类型分发
- 并行化优化

**适用场景**:
- 需要添加新硬件平台支持的开发者
- 需要优化内核性能的开发者
- 需要理解内核调用流程的开发者

---

### 4. Utils 模块

**文档**: [utils_module.md](utils_module.md)

**概述**: Utils 模块是 TinyInfiniTensor AI 编译器的工具函数模块，提供了各种辅助功能和实用工具。

**主要组件**:
- **Exception (异常处理)**: 提供统一的异常处理机制
- **infer_broadcast (广播推导)**: 实现张量广播的形状推导
- **get_real_axis (获取真实轴)**: 处理负数轴索引
- **locate_index (定位索引)**: 将线性索引转换为多维坐标
- **delocate_index (反定位索引)**: 将多维坐标转换为线性索引

**关键特性**:
- 统一的异常处理
- 高效的索引计算
- 广播机制支持
- 链式调用接口

**适用场景**:
- 需要理解工具函数的开发者
- 需要添加新工具函数的开发者
- 需要处理异常的开发者

---

## 模块间关系

```
┌─────────────────────────────────────────────────────────┐
│                    TinyInfiniTensor                     │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   ┌────────┐      ┌──────────┐      ┌──────────┐
   │  Core  │◄─────│Operators │◄─────│ Kernels  │
   └────────┘      └──────────┘      └──────────┘
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ▼
                      ┌────────┐
                      │ Utils  │
                      └────────┘
```

**依赖关系**:
1. **Core 模块** 是基础，其他模块都依赖于它
2. **Operators 模块** 依赖于 Core 模块的算子基类
3. **Kernels 模块** 依赖于 Operators 模块的算子定义
4. **Utils 模块** 为所有模块提供工具函数支持

---

## 开发指南

### 新手入门

1. **阅读顺序**:
   - 首先阅读 [Core 模块文档](core_module.md)，了解基础架构
   - 然后阅读 [Operators 模块文档](operators_module.md)，了解算子实现
   - 接着阅读 [Kernels 模块文档](kernels_module.md)，了解内核实现
   - 最后阅读 [Utils 模块文档](utils_module.md)，了解工具函数

2. **实践建议**:
   - 从简单的算子开始（如 ElementWise）
   - 逐步理解形状推导机制
   - 学习内核注册和调用流程
   - 掌握工具函数的使用

### 进阶开发

1. **添加新算子**:
   - 参考 [Operators 模块文档](operators_module.md)
   - 继承 `Operator` 基类
   - 实现 `inferShape` 方法
   - 实现对应的内核

2. **优化性能**:
   - 参考 [Kernels 模块文档](kernels_module.md)
   - 使用并行化技术
   - 优化内存访问模式
   - 实现算子融合

3. **扩展硬件支持**:
   - 参考 [Core 模块文档](core_module.md) 和 [Kernels 模块文档](kernels_module.md)
   - 实现新的 Runtime 类
   - 实现对应的内核
   - 注册到内核注册表

### 作业完成

1. **内存分配算法**:
   - 参考 [Core 模块文档](core_module.md) 中的 Allocator 部分
   - 实现高效的内存分配和回收算法
   - 考虑内存对齐和复用

2. **图优化算法**:
   - 参考 [Core 模块文档](core_module.md) 中的 Graph 部分
   - 实现去除冗余算子和算子合并
   - 考虑优化规则的优先级

3. **形状推导**:
   - 参考 [Operators 模块文档](operators_module.md)
   - 实现 Matmul、Transpose、Concat 的形状推导
   - 参考 ONNX 算子文档

4. **广播推导**:
   - 参考 [Utils 模块文档](utils_module.md)
   - 实现双向广播推导
   - 参考 ONNX Broadcasting 文档

---

## 技术要点

### 设计模式

1. **工厂模式**: Runtime 和 Kernel 的创建
2. **策略模式**: 不同的算子和内核实现
3. **观察者模式**: Tensor 和 Operator 的关系
4. **模板方法模式**: Kernel 的基类设计

### 性能优化

1. **内存优化**: 内存池、内存复用、内存对齐
2. **计算优化**: 算子融合、并行计算、向量化
3. **缓存优化**: 缓存计算结果、缓存内核对象

### 扩展性

1. **模块化设计**: 各模块职责清晰，易于扩展
2. **接口统一**: 提供统一的接口，便于添加新功能
3. **注册机制**: 支持动态注册和查找

---

## 常见问题

### Q1: 如何添加新的数据类型？

**A**: 参考 [Core 模块文档](core_module.md) 中的 DataType 部分：
1. 在 `DataType` 枚举中添加新的数据类型
2. 在 `Tensor` 中添加对新数据类型的支持
3. 在内核中添加对应的 `CASE` 分支

### Q2: 如何实现算子融合？

**A**: 参考 [Core 模块文档](core_module.md) 中的 Graph 优化部分：
1. 在 `Graph::optimize()` 中实现融合逻辑
2. 识别可以融合的算子模式
3. 创建新的融合算子
4. 更新计算图结构

### Q3: 如何支持新的硬件平台？

**A**: 参考 [Core 模块文档](core_module.md) 和 [Kernels 模块文档](kernels_module.md)：
1. 继承 `Runtime` 类实现新的运行时环境
2. 实现对应的内核类
3. 注册到内核注册表
4. 更新设备类型枚举

### Q4: 如何调试形状推导错误？

**A**: 参考 [Operators 模块文档](operators_module.md)：
1. 使用 `IT_ASSERT` 检查形状有效性
2. 打印中间结果进行调试
3. 参考算子的 ONNX 文档
4. 使用单元测试验证

---

## 参考资料

### 官方文档
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [ONNX Broadcasting](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md)

### 项目文档
- [项目介绍](../intro.md)
- [各模块详细文档](#模块文档)

### 技术博客
- AI 编译器设计原理
- 深度学习框架架构
- 计算图优化技术

---

## 贡献指南

### 代码规范
- 遵循项目现有的代码风格
- 添加详细的注释和文档
- 编写单元测试
- 更新相关文档

### 提交流程
1. Fork 项目
2. 创建功能分支
3. 提交代码
4. 发起 Pull Request
5. 等待代码审查

---

## 版本历史

### v1.0.0
- 初始版本
- 实现 Core、Operators、Kernels、Utils 模块
- 支持 CPU 平台
- 实现基础算子和内核

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发起 Discussion
- 发送邮件

---

**最后更新**: 2026-01-19