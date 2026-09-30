// Replaces the process's allocation functions for this test binary so AllocationGuard can count
// allocations made on the audio thread.

#include "AllocationGuard.h"

#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace
{
thread_local int tGuardDepth = 0;
thread_local long tAllocations = 0;

inline void noteAllocation() noexcept
{
    if (tGuardDepth > 0)
        ++tAllocations;
}

[[maybe_unused]] void* allocate(std::size_t size)
{
    noteAllocation();
    if (void* p = std::malloc(size != 0 ? size : 1))
        return p;
    throw std::bad_alloc();
}

void* allocateAligned(std::size_t size, std::align_val_t align)
{
    noteAllocation();
    const auto a = static_cast<std::size_t>(align);
#if defined(_WIN32)
    // MSVC has no std::aligned_alloc; its aligned blocks must be freed with _aligned_free.
    if (void* p = _aligned_malloc(size != 0 ? size : 1, a))
        return p;
#else
    const std::size_t rounded = (size + a - 1) / a * a;
    if (void* p = std::aligned_alloc(a, rounded != 0 ? rounded : a))
        return p;
#endif
    throw std::bad_alloc();
}

void freeAligned(void* p) noexcept
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}
} // namespace

#if defined(__GLIBC__)
// glibc exports its allocator under these names, so malloc can be wrapped without dlsym.
extern "C" void* __libc_malloc(std::size_t);
extern "C" void* __libc_calloc(std::size_t, std::size_t);
extern "C" void* __libc_realloc(void*, std::size_t);
extern "C" void __libc_free(void*);

extern "C" void* malloc(std::size_t size)
{
    // The guard counters are initial-exec TLS in the executable, so this does not recurse.
    if (tGuardDepth > 0)
        ++tAllocations;
    return __libc_malloc(size);
}
extern "C" void* calloc(std::size_t n, std::size_t size)
{
    if (tGuardDepth > 0)
        ++tAllocations;
    return __libc_calloc(n, size);
}
extern "C" void* realloc(void* p, std::size_t size)
{
    if (tGuardDepth > 0)
        ++tAllocations;
    return __libc_realloc(p, size);
}
extern "C" void free(void* p)
{
    __libc_free(p);
}

// operator new already goes through malloc; do not count it twice.
#define ALFA_NEW_COUNTS_VIA_MALLOC 1
#endif

#if ALFA_NEW_COUNTS_VIA_MALLOC
void* operator new(std::size_t size)
{
    if (void* p = std::malloc(size != 0 ? size : 1))
        return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size)
{
    return operator new(size);
}
#else
void* operator new(std::size_t size)
{
    return allocate(size);
}
void* operator new[](std::size_t size)
{
    return allocate(size);
}
#endif
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
    freeAligned(p);
}
void operator delete[](void* p, std::align_val_t) noexcept
{
    freeAligned(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept
{
    freeAligned(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept
{
    freeAligned(p);
}

namespace alfa::inference::test
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
} // namespace alfa::inference::test
