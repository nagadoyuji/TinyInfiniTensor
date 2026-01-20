#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // 1. 比较 A 和 B 的维度数
    // 2. 从后往前遍历维度，若对应维度相等或其中一个为 1，则继续遍历；否则，若其中一个为 1，则广播该维度；否则，报错
    // 3. 返回广播后的形状
    int rankA = A.size(), rankB = B.size();
    IT_ASSERT(rankA >= 0 && rankB >= 0);
    
    int maxRank = std::max(rankA, rankB);
    Shape ans(maxRank);
    
    // 从右到左遍历维度
    for (int i = 0; i < maxRank; ++i) {
        int dimA = 1, dimB = 1;
        
        // 计算从右到左的索引
        int indexA = rankA - 1 - i;
        int indexB = rankB - 1 - i;
        
        if (indexA >= 0) {
            dimA = A[indexA];
        }
        
        if (indexB >= 0) {
            dimB = B[indexB];
        }
        
        IT_ASSERT(dimA == dimB || dimA == 1 || dimB == 1,
                  "Broadcast: Incompatible shapes for broadcast.");
        ans[maxRank - 1 - i] = std::max(dimA, dimB);
    }

    // =================================== 作业 ===================================
    
    return ans;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 0);
    if (rank == 0) {
        return 0;
    }
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
