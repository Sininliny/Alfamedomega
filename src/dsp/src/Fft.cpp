#include "alfa/dsp/detail/Fft.h"

#include <pffft.h>

#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace alfa::dsp::detail
{

void AlignedBuffer::Free::operator()(float* p) const noexcept
{
    pffft_aligned_free(p);
}

void AlignedBuffer::resize(std::size_t size)
{
    data_.reset();
    size_ = 0;
    if (size == 0)
        return;
    auto* p = static_cast<float*>(pffft_aligned_malloc(size * sizeof(float)));
    if (p == nullptr)
        throw std::bad_alloc();
    data_.reset(p);
    size_ = size;
    clear();
}

void AlignedBuffer::clear() noexcept
{
    if (size_ != 0)
        std::memset(data_.get(), 0, size_ * sizeof(float));
}

struct RealFft::Setup
{
    PFFFT_Setup* p = nullptr;
};

RealFft::RealFft() = default;

RealFft::~RealFft()
{
    if (setup_ != nullptr)
    {
        if (setup_->p != nullptr)
            pffft_destroy_setup(setup_->p);
        delete setup_;
    }
}

RealFft::RealFft(RealFft&& other) noexcept
    : setup_(std::exchange(other.setup_, nullptr)), work_(std::move(other.work_)), n_(std::exchange(other.n_, 0))
{
}

RealFft& RealFft::operator=(RealFft&& other) noexcept
{
    if (this != &other)
    {
        RealFft tmp(std::move(other));
        std::swap(setup_, tmp.setup_);
        std::swap(work_, tmp.work_);
        std::swap(n_, tmp.n_);
    }
    return *this;
}

void RealFft::prepare(int n)
{
    if (n == n_ && setup_ != nullptr)
        return;
    if (n < 32 || (n & (n - 1)) != 0)
        throw std::invalid_argument("RealFft size must be a power of two >= 32");
    auto* s = pffft_new_setup(n, PFFFT_REAL);
    if (s == nullptr)
        throw std::bad_alloc();
    if (setup_ == nullptr)
        setup_ = new Setup{};
    else if (setup_->p != nullptr)
        pffft_destroy_setup(setup_->p);
    setup_->p = s;
    work_.resize(static_cast<std::size_t>(n));
    n_ = n;
}

void RealFft::forward(const float* in, float* spectrum) noexcept
{
    pffft_transform_ordered(setup_->p, in, spectrum, work_.data(), PFFFT_FORWARD);
}

void RealFft::inverse(const float* spectrum, float* out) noexcept
{
    pffft_transform_ordered(setup_->p, spectrum, out, work_.data(), PFFFT_BACKWARD);
}

} // namespace alfa::dsp::detail
