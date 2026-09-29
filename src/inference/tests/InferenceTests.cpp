#include "alfa/inference/Inference.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("inference target links", "[inference]")
{
    REQUIRE(alfa::inference::componentName() == "inference");
}
