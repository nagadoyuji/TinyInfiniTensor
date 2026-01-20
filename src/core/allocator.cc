#include "core/allocator.h"
#include <map>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;
        totalSize = 0;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = this->getAlignedSize(size);

        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
        {
            if (it->second >= size)
            {
                size_t offset = it->first;
                if (it->second > size)
                {
                    freeBlocks[it->first + size] = it->second - size;
                }
                freeBlocks.erase(it);
                used += size;
                return offset;
            }
        }

        size_t offset = used;
        used += size;
        if (used > peak)
        {
            peak = used;
        }
        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        used -= size;

        size_t start = addr;
        size_t end = addr + size;

        auto it = freeBlocks.lower_bound(start);
        
        if (it != freeBlocks.begin())
        {
            auto prev = std::prev(it);
            if (prev->first + prev->second == start)
            {
                start = prev->first;
                freeBlocks.erase(prev);
            }
        }

        if (it != freeBlocks.end() && it->first == end)
        {
            end = it->first + it->second;
            freeBlocks.erase(it);
        }

        freeBlocks[start] = end - start;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
