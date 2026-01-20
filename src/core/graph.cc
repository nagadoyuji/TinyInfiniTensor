#include "core/graph.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include "core/common.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                // 该算子不在flag中，且该算子的所有输入的源算子在flag中
                // 则将该算子加入sorted中
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        //1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        //2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        bool modified = true;
        while (modified)
        {
            modified = false;
            for (int i = 0; i < (int)ops.size(); ++i)
            {
                auto op = ops[i];
                
                // 规则1: 去除冗余的相邻 Transpose 算子
                if (i + 1 < (int)ops.size() &&
                    op->getOpType() == OpType::Transpose &&
                    ops[i + 1]->getOpType() == OpType::Transpose)
                {
                    auto transpose1 = as<TransposeObj>(op);
                    auto transpose2 = as<TransposeObj>(ops[i + 1]);
                    
                    auto permute1 = transpose1->getPermute();
                    auto permute2 = transpose2->getPermute();

                    // 检查是否为相反的操作
                    bool isOpposite = true;
                    for (size_t j = 0; j < permute1.size(); ++j)
                    {
                        if (static_cast<size_t>(permute2[permute1[j]]) != j)
                        {
                            isOpposite = false;
                            break;
                        }
                    }
                    
                    if (isOpposite)
                    {
                         // 获取两个 Transpose 算子的输出张量
                        auto output1 = transpose1->getOutputs()[0];
                        auto output2 = transpose2->getOutputs()[0];
                        
                        // 获取第一个 Transpose 的输入张量
                        auto input1 = transpose1->getInputs()[0];
                        
                        // 更新后继操作符的输入
                        auto successors = transpose2->getSuccessors();
                        for (auto &succ : successors)
                        {
                            auto succInputs = succ->getInputs();
                            for (size_t j = 0; j < succInputs.size(); ++j)
                            {
                                if (succInputs[j] == output2)
                                {
                                    succ->inputs[j] = input1;
                                    break;
                                }
                            }
                        }
                        
                        for (auto &succ : transpose1->getSuccessors())
                        {
                            succ->removePredecessors(transpose1);
                        }
                        for (auto &pred : transpose1->getPredecessors())
                        {
                            pred->removeSuccessors(transpose1);
                        }
                        for (auto &succ : transpose2->getSuccessors())
                        {
                            succ->removePredecessors(transpose2);
                        }
                        for (auto &pred : transpose2->getPredecessors())
                        {
                            pred->removeSuccessors(transpose2);
                        }
                        
                        // 移除两个 Transpose 算子
                        ops.erase(ops.begin() + i + 1);
                        ops.erase(ops.begin() + i);
                        
                        // 移除中间张量
                        removeTensor(output1);
                        removeTensor(output2);
                        
                        // 更新张量的连接关系
                        input1->removeTarget(transpose1);
                        for (auto &succ : successors)
                        {
                            input1->addTarget(succ);
                        }
                        
                        --i;
                        modified = true;
                        continue;
                    }
                }
                
                // 规则2: 将 Transpose 合并到 Matmul
                if (op->getOpType() == OpType::MatMul)
                {
                    auto matmul = as<MatmulObj>(op);
                    auto inputs = matmul->getInputs();
                    
                    // 检查第一个输入是否为 Transpose
                    if (inputs[0] && inputs[0]->getSource())
                    {
                        auto source = inputs[0]->getSource();
                        if (source->getOpType() == OpType::Transpose)
                        {
                            auto transpose = as<TransposeObj>(source);
                            if (isSwapLastTwoDims(transpose.get()))
                            {
                                // 获取 Transpose 的输出张量
                                auto transposeOutput = transpose->getOutputs()[0];

                                // 将 Transpose 的效果融入到 Matmul 的 transA 中
                                bool newTransA = !matmul->getTransA();
                                matmul->setTransA(newTransA);
                                
                                // 更新 Matmul 的输入为 Transpose 的输入
                                auto transposeInput = transpose->getInputs()[0];
                                matmul->inputs[0] = transposeInput;
                                
                                // 清理predecessors和successors引用
                                for (auto &succ : transpose->getSuccessors())
                                {
                                    succ->removePredecessors(transpose);
                                }
                                for (auto &pred : transpose->getPredecessors())
                                {
                                    pred->removeSuccessors(transpose);
                                }
                                
                                // 移除 Transpose 算子
                                removeOperator(transpose);
                                
                                // 移除中间张量
                                removeTensor(transposeOutput);
                                
                                // 更新张量的连接关系
                                transposeInput->removeTarget(transpose);
                                transposeInput->addTarget(matmul);
                                
                                --i;
                                modified = true;
                                continue;
                            }
                        }
                    }
                    
                    // 检查第二个输入是否为 Transpose
                    if (inputs[1] && inputs[1]->getSource())
                    {
                        auto source = inputs[1]->getSource();
                        if (source->getOpType() == OpType::Transpose)
                        {
                            auto transpose = as<TransposeObj>(source);
                            if (isSwapLastTwoDims(transpose.get()))
                            {
                                // 获取 Transpose 的输出张量
                                auto transposeOutput = transpose->getOutputs()[0];
                                
                                // 将 Transpose 的效果融入到 Matmul 的 transB 中
                                bool newTransB = !matmul->getTransB();
                                matmul->setTransB(newTransB);
                                
                                // 更新 Matmul 的输入为 Transpose 的输入
                                auto transposeInput = transpose->getInputs()[0];
                                matmul->inputs[1] = transposeInput;
                                
                                // 清理predecessors和successors引用
                                for (auto &succ : transpose->getSuccessors())
                                {
                                    succ->removePredecessors(transpose);
                                }
                                for (auto &pred : transpose->getPredecessors())
                                {
                                    pred->removeSuccessors(transpose);
                                }
                                
                                // 移除 Transpose 算子
                                removeOperator(transpose);
                                
                                // 移除中间张量
                                removeTensor(transposeOutput);
                                
                                // 更新张量的连接关系
                                transposeInput->removeTarget(transpose);
                                transposeInput->addTarget(matmul);
                                
                                --i;
                                modified = true;
                                continue;
                            }
                        }
                    }
                }
            }
        }
        // =================================== 作业 ===================================
    }

    bool GraphObj::isSwapLastTwoDims(TransposeObj *transpose)
    {
        auto permute = transpose->getPermute();
        int rank = permute.size();
        
        if (rank < 2)
        {
            return false;
        }
        
        // 检查是否只交换了最后两个维度
        // 对于 rank 维度的张量，permute 应该是 [0, 1, ..., rank-2, rank-1]
        // 如果是交换最后两个维度，permute 应该是 [0, 1, ..., rank-1, rank-2]
        
        // 检查除最后两个维度外的所有维度是否保持不变
        for (int i = 0; i < rank - 2; ++i)
        {
            if (permute[i] != i)
            {
                return false;
            }
        }
        
        // 检查最后两个维度是否交换
        if (permute[rank - 2] != rank - 1 || permute[rank - 1] != rank - 2)
        {
            return false;
        }
        
        return true;
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        // 1. 为所有张量分配内存，记录偏移量
        std::map<Tensor, size_t> tensorOffsets;
        for (auto tensor : tensors)
        {
            size_t offset = allocator.alloc(tensor->getBytes());
            tensorOffsets[tensor] = offset;
        }

        // 2. 获取实际分配的内存指针
        void *basePtr = allocator.getPtr();

        // 3. 为每个张量创建 Blob 并绑定内存
        for (auto tensor : tensors)
        {
            size_t offset = tensorOffsets[tensor];
            void *tensorPtr = static_cast<char *>(basePtr) + offset;
            Blob blob = make_ref<BlobObj>(runtime, tensorPtr);
            tensor->setDataBlob(blob);
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini