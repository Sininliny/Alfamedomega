#pragma once

// Polyphase halfband FIR resamplers for NonlinearProcessor's 2x / 4x oversampling.
//
// FIR rather than IIR halfbands because the processor mixes the shaped signal with the dry
// input: linear phase plus an integer latency lets the dry path be delayed to match exactly,
// so a partial mix never combs. A halfband kernel of length N = 4K + 3 has its centre tap at
// c = 2K + 1, every other tap zero and centre tap 1/2, so each 2x stage costs only the M = 2K + 2
// "dense" taps per base sample in each direction. Up plus down adds c samples of base-rate
// latency per 2x round trip (NonlinearProcessor documents the 4x total).
//
// Everything is fixed-size; nothing here allocates.

#include "NonlinearHalfbandMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace alfa::dsp::detail
{

struct NonlinearHalfband
{
    static constexpr int kMaxDense = 24;

    std::array<float, kMaxDense> dense{}; ///< taps at even kernel indices 0, 2, .. 2(M-1)
    int m = 0;                            ///< dense tap count, 2K + 2
    int k = 0;                            ///< centre offset in dense steps; latency c = 2K + 1

    /// Kaiser-windowed halfband sinc. Not real-time (uses doubles and a Bessel series), call
    /// from prepare().
    void design(int halfK, double beta) noexcept
    {
        k = std::clamp(halfK, 0, kMaxDense / 2 - 1);
        m = 2 * k + 2;
        if (designNonlinearHalfbandDense<float>(dense, k, beta))
            return;
        // Preserve legacy handling of coefficients outside the tensor designer's domain.
        const int c = 2 * k + 1;
        const auto i0 = [](double x)
        {
            double sum = 1.0, term = 1.0;
            for (int n = 1; n < 40; ++n)
            {
                const double t = x / (2.0 * n);
                term *= t * t;
                sum += term;
            }
            return sum;
        };
        double total = 0.0;
        std::array<double, kMaxDense> h{};
        for (int j = 0; j < m; ++j)
        {
            const double d = static_cast<double>(2 * j - c); // odd offset from the centre
            const double pd = 3.14159265358979323846 * d / 2.0;
            const double sinc = std::sin(pd) / pd;
            const double r = d / static_cast<double>(c + 1);
            const double w = i0(beta * std::sqrt(1.0 - r * r)) / i0(beta);
            h[static_cast<std::size_t>(j)] = 0.5 * sinc * w;
            total += h[static_cast<std::size_t>(j)];
        }
        // Dense taps sum to exactly 1/2 so both polyphase branches have unity DC gain (no
        // residual at the image frequency for a constant input, e.g. the bias offset).
        for (int j = 0; j < m; ++j)
            dense[static_cast<std::size_t>(j)] =
                static_cast<float>(h[static_cast<std::size_t>(j)] * (0.5 / total));
        for (int j = m; j < kMaxDense; ++j)
            dense[static_cast<std::size_t>(j)] = 0.0f;
    }
};

/// Newest-first history of the last `size` values, stored twice so reads are contiguous.
template <int Capacity>
class NonlinearHistory
{
public:
    void setSize(int size) noexcept { size_ = std::clamp(size, 1, Capacity); }
    void clear() noexcept
    {
        buf_.fill(0.0f);
        pos_ = 0;
    }
    void push(float x) noexcept
    {
        pos_ = pos_ == 0 ? size_ - 1 : pos_ - 1;
        buf_[static_cast<std::size_t>(pos_)] = x;
        buf_[static_cast<std::size_t>(pos_ + size_)] = x;
    }
    /// Element j is the value pushed j pushes ago (0 = newest).
    [[nodiscard]] const float* data() const noexcept { return buf_.data() + pos_; }

private:
    std::array<float, static_cast<std::size_t>(2 * Capacity)> buf_{};
    int pos_ = 0;
    int size_ = Capacity;
};

/// Dot product of M taps with 4 independent accumulators, so it pipelines (and vectorises)
/// without relying on -ffast-math reassociation.
template <int M>
inline float nonlinearDot(const float* h, const float* x) noexcept
{
    static_assert(M % 4 == 0);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int j = 0; j < M; j += 4)
    {
        a0 += h[j] * x[j];
        a1 += h[j + 1] * x[j + 1];
        a2 += h[j + 2] * x[j + 2];
        a3 += h[j + 3] * x[j + 3];
    }
    return (a0 + a1) + (a2 + a3);
}

/// Base rate in, two samples out. y[2m] = 2 sum_j h[2j] x[m-j]; y[2m+1] = x[m-K] (centre tap).
/// K is fixed at compile time; the kernel must have been designed with the same K.
template <int K>
class NonlinearUpsampler2x
{
public:
    static constexpr int kM = 2 * K + 2;

    void setKernel(const NonlinearHalfband* kernel) noexcept
    {
        kernel_ = kernel;
        hist_.setSize(kM);
        hist_.clear();
    }
    void reset() noexcept { hist_.clear(); }

    void process(float x, float& y0, float& y1) noexcept
    {
        hist_.push(x);
        const float* hx = hist_.data();
        y0 = 2.0f * nonlinearDot<kM>(kernel_->dense.data(), hx);
        y1 = hx[K];
    }

private:
    const NonlinearHalfband* kernel_ = nullptr;
    NonlinearHistory<kM> hist_;
};

/// Two samples in, base rate out: y[m] = sum_k h[k] v[2m-k]. The dense taps see the even
/// samples, the centre tap the odd sample K + 1 pairs back.
template <int K>
class NonlinearDownsampler2x
{
public:
    static constexpr int kM = 2 * K + 2;

    void setKernel(const NonlinearHalfband* kernel) noexcept
    {
        kernel_ = kernel;
        even_.setSize(kM);
        odd_.setSize(K + 1);
        reset();
    }
    void reset() noexcept
    {
        even_.clear();
        odd_.clear();
    }

    float process(float v0, float v1) noexcept
    {
        even_.push(v0);
        const float acc = nonlinearDot<kM>(kernel_->dense.data(), even_.data()) + 0.5f * odd_.data()[K];
        odd_.push(v1);
        return acc;
    }

private:
    const NonlinearHalfband* kernel_ = nullptr;
    NonlinearHistory<kM> even_;
    NonlinearHistory<K + 1> odd_;
};

} // namespace alfa::dsp::detail
