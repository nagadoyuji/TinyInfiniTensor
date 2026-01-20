# Operator Utils 函数文档

## 1. Shape infer_broadcast(const Shape &A, const Shape &B)

### 作用
对两个张量形状进行双向广播，返回广播后的形状。

### 参数
- `A`：第一个张量的形状
- `B`：第二个张量的形状

### 返回值
广播后的形状，满足 ONNX 广播规范。

### 实现说明
该函数根据 ONNX 广播规范实现张量形状的双向广播：
1. 从末尾开始对齐两个形状
2. 对于每个维度，遵循广播规则：
   - 如果两个维度相同，保持该维度
   - 如果一个维度为 1，扩展到另一个维度的大小
   - 否则，广播失败

### 使用示例
```cpp
Shape A = {2, 1, 3};  // 形状 (2, 1, 3)
Shape B = {1, 4, 3};  // 形状 (1, 4, 3)
Shape result = infer_broadcast(A, B);  // 结果: {2, 4, 3}
```

## 2. int get_real_axis(const int &axis, const int &rank)

### 作用
将用户输入的轴索引转换为实际的轴索引（支持负索引）。

### 参数
- `axis`：用户输入的轴索引（可以是负数，如 -1 表示最后一个轴）
- `rank`：张量的秩（维度数）

### 返回值
转换后的实际轴索引（范围：0 <= real_axis < rank）。

### 实现说明
1. 验证输入轴的有效性：`-rank <= axis <= rank - 1`
2. 如果轴为负数，转换为正数：`newAxis = rank + axis`
3. 返回转换后的轴索引

### 使用示例
```cpp
int rank = 4;  // 4维张量
int axis1 = 2;  // 直接使用正索引
int axis2 = -1; // 使用负索引表示最后一个轴

int realAxis1 = get_real_axis(axis1, rank);  // 结果: 2
int realAxis2 = get_real_axis(axis2, rank);  // 结果: 3
```

## 3. Shape locate_index(size_t inputN, const Shape &shape)

### 作用
将线性索引转换为多维索引。

### 参数
- `inputN`：线性索引（从 0 开始）
- `shape`：张量的形状（如 {2, 3, 4}）

### 返回值
多维索引数组，与 shape 维度对应。

### 实现说明
1. 创建一个与 shape 相同维度的结果数组
2. 从右到左（最后一个维度到第一个维度）计算每个维度的索引：
   - 当前维度索引 = inputN % 当前维度大小
   - inputN = inputN / 当前维度大小
3. 返回计算得到的多维索引数组

### 使用示例
```cpp
Shape shape = {2, 3, 4};  // 形状 (2, 3, 4)
size_t linearIndex = 10;

Shape multiIndex = locate_index(linearIndex, shape);  // 结果: {0, 2, 2}
// 解释: 0 * 3*4 + 2 * 4 + 2 = 0 + 8 + 2 = 10
```

## 4. size_t delocate_index(const Shape &shapeIndex, const Shape &shape, const Shape &stride)

### 作用
将多维索引转换为线性索引（考虑广播）。

### 参数
- `shapeIndex`：多维索引数组
- `shape`：张量的形状
- `stride`：张量的步长数组

### 返回值
线性索引。

### 实现说明
1. 验证输入参数的维度一致性
2. 对于每个维度：
   - 将索引对形状大小取模（考虑广播）
   - 计算该维度贡献的线性偏移：`index[i] * stride[i]`
3. 将所有维度的偏移相加，得到最终的线性索引

### 使用示例
```cpp
Shape shape = {2, 3, 4};    // 形状 (2, 3, 4)
Shape stride = {12, 4, 1};  // 步长 (3*4, 4, 1)
Shape multiIndex = {0, 2, 2};  // 多维索引

size_t linearIndex = delocate_index(multiIndex, shape, stride);  // 结果: 10
// 计算: 0*12 + 2*4 + 2*1 = 0 + 8 + 2 = 10
```

## 5. std::string device_to_str(Device device)

### 作用
将 Device 枚举转换为字符串表示。

### 参数
- `device`：设备枚举（如 Device::CPU）

### 返回值
设备名称的字符串表示。

### 实现说明
根据 Device 枚举值返回对应的字符串：
- Device::CPU -> "CPU"
- 其他设备类型：抛出异常（未实现）

### 使用示例
```cpp
Device device = Device::CPU;
std::string deviceStr = device_to_str(device);  // 结果: "CPU"
```

## 6. std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs)

### 作用
将 KernelAttrs 转换为字符串表示。

### 参数
- `kernelAttrs`：内核属性元组（包含 Device 和 OpType）

### 返回值
内核属性的字符串表示。

### 实现说明
1. 从 kernelAttrs 中提取设备类型和运算符类型
2. 将设备类型转换为字符串
3. 将运算符类型转换为字符串
4. 返回组合后的字符串（格式："设备类型, 运算符类型"）

### 使用示例
```cpp
KernelAttrs attrs = {Device::CPU, OpType::MatMul};
std::string attrsStr = get_kernel_attrs_str(attrs);  // 结果: "CPU, MatMul"
```

## 代码结构

```cpp
// operator_utils.cc 函数依赖关系
//
// ┌─────────────────────────┐
// │ infer_broadcast()       │
// └─────────────────────────┘
//
// ┌─────────────────────────┐
// │ get_real_axis()         │
// └─────────────────────────┘
//
// ┌─────────────────────────┐
// │ locate_index()          │
// └─────────────────────────┘
//
// ┌─────────────────────────┐
// │ delocate_index()        │
// └─────────────────────────┘
//
// ┌─────────────────────────┐      ┌─────────────────────────┐
// │ device_to_str()         │──────┤ get_kernel_attrs_str()  │
// └─────────────────────────┘      └─────────────────────────┘
```

## 使用场景

这些函数在 InfiniTensor 中主要用于：

1. **张量操作**：
   - 广播操作的形状计算
   - 索引转换（线性索引 ↔ 多维索引）

2. **运算符实现**：
   - 处理用户输入的轴参数
   - 计算张量元素的内存位置

3. **调试和日志**：
   - 设备类型和内核属性的字符串表示
   - 方便调试和日志记录