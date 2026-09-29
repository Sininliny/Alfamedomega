#pragma once

// Real FFT on top of pffft, plus the aligned buffers it needs. Allocation happens only in
// prepare(); transforms are allocation-free.

#include <cstddef>
#include <memory>

namespace alfa::dsp::detail
{

/// 64-byte-aligned float array, zero-initialised. Move-only.
class AlignedBuffer
{
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t size) { resize(size); }

    void resize(std::size_t size);
    void clear() noexcept; ///< zero every element

    [[nodiscard]] float* data() noexcept { return data_.get(); }
    [[nodiscard]] const float* data() const noexcept { return data_.get(); }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    float& operator[](std::size_t i) noexcept { return data_[i]; }
    const float& operator[](std::size_t i) const noexcept { return data_[i]; }

private:
    struct Free
    {
        void operator()(float* p) const noexcept;
    };
    std::unique_ptr<float[], Free> data_;
    std::size_t size_ = 0;
};

/// Real FFT of size n (a power of two, at least 32).
///
/// Spectrum layout (pffft "ordered"): s[0] = DC, s[1] = Nyquist, then s[2k], s[2k+1] are the
/// real and imaginary parts of bin k for k = 1 .. n/2-1. Neither direction is scaled:
/// inverse(forward(x)) = n * x, and inverse() evaluates
///   x[t] = s[0] + s[1] (-1)^t + 2 sum_{k=1}^{n/2-1} Re((s[2k] + i s[2k+1]) e^{i 2 pi k t / n}).
/// All pointers passed in must be 16-byte aligned (AlignedBuffer is). Input and output may alias.
class RealFft
{
public:
    RealFft();
    ~RealFft();
    RealFft(RealFft&&) noexcept;
    RealFft& operator=(RealFft&&) noexcept;
    RealFft(const RealFft&) = delete;
    RealFft& operator=(const RealFft&) = delete;

    void prepare(int n);
    [[nodiscard]] int size() const noexcept { return n_; }

    void forward(const float* in, float* spectrum) noexcept;
    void inverse(const float* spectrum, float* out) noexcept;

private:
    struct Setup;
    Setup* setup_ = nullptr;
    AlignedBuffer work_;
    int n_ = 0;
};

[[nodiscard]] constexpr int nextPowerOfTwo(int x) noexcept
{
    int p = 1;
    while (p < x)
        p <<= 1;
    return p;
}

} // namespace alfa::dsp::detail
