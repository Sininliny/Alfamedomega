// Replaces global operator new/delete for the test binary so AllocationGuard can count
// allocations on the audio path.

#include "TestHelpers.h"

#include <cstdlib>
#include <new>

namespace
{
thread_local int tGuardDepth = 0;
thread_local long tAllocations = 0;

void* allocate(std::size_t size)
{
    if (tGuardDepth > 0)
        ++tAllocations;
    if (void* p = std::malloc(size != 0 ? size : 1))
        return p;
    throw std::bad_alloc();
}

void* allocateAligned(std::size_t size, std::align_val_t align)
{
    if (tGuardDepth > 0)
        ++tAllocations;
    const auto a = static_cast<std::size_t>(align);
    const std::size_t rounded = (size + a - 1) / a * a;
    if (void* p = std::aligned_alloc(a, rounded != 0 ? rounded : a))
        return p;
    throw std::bad_alloc();
}
} // namespace

void* operator new(std::size_t size)
{
    return allocate(size);
}
void* operator new[](std::size_t size)
{
    return allocate(size);
}
void* operator new(std::size_t size, std::align_val_t a)
{
    return allocateAligned(size, a);
}
void* operator new[](std::size_t size, std::align_val_t a)
{
    return allocateAligned(size, a);
}
void operator delete(void* p) noexcept
{
    std::free(p);
}
void operator delete[](void* p) noexcept
{
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept
{
    std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept
{
    std::free(p);
}
void operator delete(void* p, std::align_val_t) noexcept
{
    std::free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept
{
    std::free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept
{
    std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept
{
    std::free(p);
}

namespace alfa::dsp::test
{
AllocationGuard::AllocationGuard() noexcept
{
    if (tGuardDepth++ == 0)
        tAllocations = 0;
}
AllocationGuard::~AllocationGuard()
{
    --tGuardDepth;
}
long AllocationGuard::count() const noexcept
{
    return tAllocations;
}
} // namespace alfa::dsp::test
