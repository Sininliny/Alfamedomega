#pragma once

// Flush denormals to zero for the lifetime of the object (audio thread only). Decaying
// filters and overlap-add tails otherwise fall into denormal range and cost 10-100x per op.

#include <cstdint>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define ALFA_DSP_HAS_SSE 1
#endif

namespace alfa::dsp::detail
{

class ScopedNoDenormals
{
public:
    ScopedNoDenormals() noexcept
    {
#if defined(ALFA_DSP_HAS_SSE)
        const unsigned int csr = _mm_getcsr();
        saved_ = csr;
        _mm_setcsr(csr | 0x8040u); // FTZ | DAZ
#elif defined(__aarch64__)
        std::uint64_t fpcr;
        asm volatile("mrs %0, fpcr" : "=r"(fpcr));
        saved_ = fpcr;
        asm volatile("msr fpcr, %0" : : "r"(fpcr | (1ull << 24))); // FZ
#endif
    }

    ~ScopedNoDenormals()
    {
#if defined(ALFA_DSP_HAS_SSE)
        _mm_setcsr(static_cast<unsigned int>(saved_));
#elif defined(__aarch64__)
        asm volatile("msr fpcr, %0" : : "r"(saved_));
#endif
    }

    ScopedNoDenormals(const ScopedNoDenormals&) = delete;
    ScopedNoDenormals& operator=(const ScopedNoDenormals&) = delete;

private:
    std::uint64_t saved_ = 0;
};

} // namespace alfa::dsp::detail
