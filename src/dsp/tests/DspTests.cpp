#include "alfa/dsp/Dsp.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("dsp target links", "[dsp]")
{
    REQUIRE(alfa::dsp::componentName() == "dsp");
}
