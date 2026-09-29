#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/detail/FastMath.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using namespace alfa::dsp;

TEST_CASE("control-frame fields tile the frame without gaps", "[dsp][layout]")
{
    std::size_t expected = 0;
    for (const auto& f : kFields)
    {
        if (f.name == "osc.re")
            expected += 1;  // the reserved float after attack_tau
        REQUIRE(f.offset == expected);
        expected += f.count;
    }
    REQUIRE(expected == kFrameFloats);
}

TEST_CASE("hop length follows the 250 Hz frame rate", "[dsp][layout]")
{
    REQUIRE(hopSamples(48000.0) == 192);
    REQUIRE(hopSamples(44100.0) == 176);
    REQUIRE(hopSamples(96000.0) == 384);
}

TEST_CASE("tiers only grow", "[dsp][layout]")
{
    for (int t = 1; t < 4; ++t)
    {
        const auto a = limitsFor(static_cast<Tier>(t - 1));
        const auto b = limitsFor(static_cast<Tier>(t));
        REQUIRE(b.partials >= a.partials);
        REQUIRE(b.fmOperators >= a.fmOperators);
        REQUIRE(b.grainLayers >= a.grainLayers);
        REQUIRE(b.partials <= kMaxPartials);
        REQUIRE(b.fmOperators <= kMaxFmOperators);
        REQUIRE(b.grainLayers <= kMaxGrainLayers);
    }
}

TEST_CASE("sin2pi matches std::sin", "[dsp][math]")
{
    double worst = 0.0;
    for (int i = -40000; i <= 40000; ++i)
    {
        const float x = static_cast<float>(i) * 0.000123f;
        const double ref = std::sin(2.0 * 3.14159265358979323846 * static_cast<double>(x));
        worst = std::max(worst, std::abs(ref - detail::sin2pi(x)));
    }
    REQUIRE(worst < 2e-6);
}
