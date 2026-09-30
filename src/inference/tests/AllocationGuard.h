#pragma once

namespace alfa::inference::test
{

/// Counts heap allocations made by the current thread while an AllocationGuard is alive.
/// Covers operator new everywhere and, on glibc, malloc/calloc/realloc too, so allocations
/// inside anira, tanh-lib or ONNX Runtime are caught as well.
class AllocationGuard
{
public:
    AllocationGuard() noexcept;
    ~AllocationGuard();
    AllocationGuard(const AllocationGuard&) = delete;
    AllocationGuard& operator=(const AllocationGuard&) = delete;

    [[nodiscard]] long count() const noexcept;
};

} // namespace alfa::inference::test
