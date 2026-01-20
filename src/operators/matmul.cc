#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // 1. 检查输入张量的秩是否为 2
        // 2. 检查输入张量的形状是否符合 matmul 操作的要求
        // 3. 返回 matmul 操作后的 shape
        auto A = inputs[0];
        auto B = inputs[1];
        int rankA = A->getRank();
        int rankB = B->getRank();
        
        IT_ASSERT(rankA == rankB, "Matmul: Input tensors must have same ranks.");
        
        Shape ans(rankA);
        
        // 计算最后两个维度
        int dimA_m = A->getDims()[rankA - 2];
        int dimA_k = A->getDims()[rankA - 1];
        int dimB_k = B->getDims()[rankB - 2];
        int dimB_n = B->getDims()[rankB - 1];
        
        // 考虑转置
        if (transA) {
            m = dimA_k;
            k = dimA_m;
        } else {
            m = dimA_m;
            k = dimA_k;
        }
        
        if (transB) {
            IT_ASSERT(k == dimB_n, "Matmul: Input tensors must have compatible shapes.");
            n = dimB_k;
        } else {
            IT_ASSERT(k == dimB_k, "Matmul: Input tensors must have compatible shapes.");
            n = dimB_n;
        }
        
        // 处理批量维度（广播）
        for (int i = 0; i < rankA - 2; ++i) {
            int dimA = A->getDims()[i];
            int dimB = B->getDims()[i];
            IT_ASSERT(dimA == dimB || dimA == 1 || dimB == 1,
                      "Matmul: Batch dimensions must be broadcastable.");
            ans[i] = std::max(dimA, dimB);
        }
        
        ans[rankA - 2] = m;
        ans[rankA - 1] = n;

        // =================================== 作业 ===================================
        return {{ans}};
    }

} // namespace infini