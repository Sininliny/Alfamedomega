#include "../src/detail/Finite.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <vector>

TEMPLATE_TEST_CASE("Finite check agrees with std::isfinite for every IEEE class", "[tensor][finite]", float,
                   double)
{
    using L = std::numeric_limits<TestType>;
    const std::vector<TestType> special{TestType(0),     -TestType(0),      TestType(1),    -TestType(1),
                                        L::max(),        L::lowest(),       L::min(),       L::denorm_min(),
                                        L::epsilon(),    L::infinity(),     -L::infinity(), L::quiet_NaN(),
                                        -L::quiet_NaN(), L::signaling_NaN()};
    for (const auto value : special)
    {
        // Odd lengths and positions exercise the vectorised body and the scalar tail.
        for (std::size_t length : {1u, 3u, 8u, 17u, 1025u})
            for (std::size_t at : {std::size_t{0}, length / 2, length - 1})
            {
                std::vector<TestType> values(length, TestType(0.25));
                values[at] = value;
                REQUIRE(alfa::tensor::detail::allFinite<TestType>(values) ==
                        static_cast<bool>(std::isfinite(value)));
            }
    }
    REQUIRE(alfa::tensor::detail::allFinite<TestType>({}));
}
