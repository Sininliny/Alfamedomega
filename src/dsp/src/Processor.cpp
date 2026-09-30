#include "alfa/dsp/Processor.h"

#include <algorithm>
#include <cmath>

namespace alfa::dsp
{

float mapParam(const ParamInfo& info, float x) noexcept
{
    x = std::isfinite(x) ? std::clamp(x, 0.0f, 1.0f) : info.defaultNorm;
    switch (info.curve)
    {
    case Curve::Linear:
        return info.min + x * (info.max - info.min);
    case Curve::Log:
        return info.min * std::pow(info.max / info.min, x);
    case Curve::Choice:
        return std::round(x * info.max);
    case Curve::Toggle:
        return x >= 0.5f ? 1.0f : 0.0f;
    }
    return x;
}

} // namespace alfa::dsp
