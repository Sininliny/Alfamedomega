#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace
{
std::string reference(const char* name)
{
    std::ifstream file(std::string(ALFA_SOURCE_DIR) + "/programs/" + name + ".aom.json");
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}
} // namespace

TEST_CASE("Tensor serialization resolves explicitly registered extension operations", "[tensor][compiler]")
{
    using namespace alfa::tensor;
    auto document = nlohmann::json::parse(reference("fx"));
    document["instances"][0]["primitive"] = "extension.test_recurrence";
    document["tensors"]["parameters"]["values"][0] = 0.25;
    Configuration configuration;
    std::string error;
    REQUIRE_FALSE(parseJSON(document.dump(), configuration, error));
    const PrimitiveResolver resolver =
        [](std::string_view id, std::size_t count, Primitive& result, std::string&)
    {
        if (id != "extension.test_recurrence" || count != 1)
            return false;
        result = kernels::feedbackCell();
        result.id = id;
        return true;
    };
    REQUIRE(parseJSON(document.dump(), configuration, error, resolver));
    Budget budget;
    REQUIRE(readBudgetJSON(document.dump(), budget, error));
    Runtime runtime;
    REQUIRE(runtime.prepare(configuration, budget, 48000, error));
    REQUIRE(runtime.tick({}));
    CHECK(serializeJSON(configuration).find("extension.test_recurrence") != std::string::npos);
}

TEST_CASE("Versioned tensor program round trips and factory descriptors retain ownership",
          "[tensor][compiler]")
{
    using namespace alfa::tensor;
    std::string error;
    Configuration config;
    REQUIRE(parseJSON(reference("hybrid"), config, error));
    CHECK(config.instances.size() == 2);
    REQUIRE(config.instances[0].descriptor != nullptr);
    auto copied = config;
    config = {};
    const auto serialized = serializeJSON(copied);
    Configuration reparsed;
    REQUIRE(parseJSON(serialized, reparsed, error));
    CHECK(reparsed.parameters.values == copied.parameters.values);
    CHECK(reparsed.routing.values == copied.routing.values);
    CHECK(reparsed.instances[0].primitive->id == "math.sine");
    Budget budget;
    REQUIRE(readBudgetJSON(reference("hybrid"), budget, error));
    Runtime runtime;
    REQUIRE(runtime.prepare(reparsed, budget, 48000, error));
}

TEST_CASE("Tensor compilation refuses incompatible versions and budgets atomically", "[tensor][compiler]")
{
    using namespace alfa::tensor;
    Runtime runtime;
    std::string error;
    REQUIRE(compileJSON(reference("fx"), 48000, runtime, error));
    auto document = nlohmann::json::parse(reference("fx"));
    SECTION("future ontology")
    {
        document["ontology"] = "aom.math/2";
    }
    SECTION("unknown primitive")
    {
        document["instances"][0]["primitive"] = "imaginary.placeholder";
    }
    SECTION("negative dimension")
    {
        document["instances"][0]["state_size"] = -1;
    }
    SECTION("insufficient budget")
    {
        document["budget"]["state_elements"] = 0;
    }
    SECTION("wrong clock")
    {
        document["instances"][0]["clock"]["period_samples"] = 0;
    }
    REQUIRE_FALSE(compileJSON(document.dump(), 48000, runtime, error));
    CHECK_FALSE(error.empty());
    CHECK(runtime.state().size() == 1);
    CHECK(runtime.schedule().size() == 1);
}
