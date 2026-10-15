#include "PresetStore.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace
{
using JSON = nlohmann::json;

struct TestDirectory
{
    juce::File file;
    TestDirectory()
    {
        const auto workspace = std::filesystem::weakly_canonical(ALFA_SOURCE_DIR);
        const auto parent = std::filesystem::weakly_canonical(workspace / "build");
        if (parent.parent_path() != workspace)
            throw std::runtime_error("Preset test directory must resolve inside the workspace");
        file =
            juce::File(juce::String(parent.string())).getChildFile("preset-tests-" + juce::Uuid().toString());
        if (file.exists())
            throw std::runtime_error("Refuse to reuse a preset test directory");
    }
    ~TestDirectory()
    {
        const auto workspace = std::filesystem::weakly_canonical(ALFA_SOURCE_DIR);
        const auto actual = std::filesystem::weakly_canonical(file.getFullPathName().toStdString());
        if (actual.parent_path() == workspace / "build" && file.getFileName().startsWith("preset-tests-"))
            file.deleteRecursively();
    }
};

juce::String document()
{
    return juce::File(juce::String(ALFA_SOURCE_DIR))
        .getChildFile("programs/instrument.amo.json")
        .loadFileAsString();
}
} // namespace

TEST_CASE("Double precision preset storage retains tier and full numerical JSON",
          "[tensor-host][presets][precision]")
{
    TestDirectory temporary;
    alfa::tensorhost::PresetStore store(temporary.file);
    auto original = JSON::parse(document().toStdString());
    original["tier"] = "Pro";
    original["scalar_type"] = "float64";
    original["budget"]["boundary_conversion_bytes"] = 16384;
    constexpr double frequency = 400.123456789011;
    original["tensors"]["parameters"]["values"][0] = frequency;
    juce::String error;
    REQUIRE(store.save("Double fixture", "Reference", juce::String(original.dump()), error));
    alfa::tensorhost::PresetStore reopened(temporary.file);
    REQUIRE(reopened.refresh(error));
    REQUIRE(reopened.entries().size() == 1);
    const auto loaded = JSON::parse(reopened.load(0, error).toStdString());
    for (const auto& [key, value] : original.items())
        CHECK(loaded.at(key) == value);
    CHECK(loaded["tensors"]["parameters"]["values"][0].get<double>() == frequency);
    CHECK(loaded["scalar_type"] == "float64");
    CHECK(loaded["tier"] == "Pro");
}

TEST_CASE("Preset browsing is read-only and saving preserves complete sound metadata",
          "[tensor-host][presets]")
{
    TestDirectory temporary;
    alfa::tensorhost::PresetStore store(temporary.file);
    juce::String error;
    REQUIRE_FALSE(temporary.file.exists());
    REQUIRE(store.refresh(error));
    CHECK(store.entries().empty());
    CHECK_FALSE(temporary.file.exists());
    auto original = JSON::parse(document().toStdString());
    original["metadata"] = {{"author", "Fixture"}, {"tags", {"custom", "preserved"}}};
    original["futureMetadata"] = {{"retained", true}};
    original["parameter_locks"] = {1, 0};
    REQUIRE(store.save("Alpha", "Category A", juce::String(original.dump()), error));
    REQUIRE(store.entries().size() == 1);
    CHECK(store.entries()[0].name == "Alpha");
    CHECK(store.entries()[0].category == "Category A");
    CHECK(store.entries()[0].file.getParentDirectory() == temporary.file);
    const auto loaded = JSON::parse(store.load(0, error).toStdString());
    for (const auto& [key, value] : original.items())
        CHECK(loaded.at(key) == value);
    CHECK(loaded.at("preset").at("schema") == "amo.preset/1");
    CHECK_FALSE(loaded.contains("mode"));
    CHECK_FALSE(loaded.contains("intentions"));
    CHECK_FALSE(loaded.contains("window"));
    alfa::tensorhost::PresetStore reopened(temporary.file);
    REQUIRE(reopened.refresh(error));
    CHECK(reopened.entries()[0].id == store.entries()[0].id);
    CHECK(reopened.load(0, error) == store.load(0, error));
}

TEST_CASE("Preset names cannot escape directory and search and neighbours retain stable indices",
          "[tensor-host][presets]")
{
    TestDirectory temporary;
    alfa::tensorhost::PresetStore store(temporary.file);
    juce::String error;
    REQUIRE(store.save("../Alpha", "Category A", document(), error));
    REQUIRE(store.save("Beta", "Category B", document(), error));
    REQUIRE(store.entries().size() == 2);
    for (const auto& entry : store.entries())
        CHECK(entry.file.getParentDirectory() == temporary.file);
    CHECK(store.search("alpha") == std::vector<std::size_t>{0});
    CHECK(store.search("", "category b") == std::vector<std::size_t>{1});
    CHECK(store.search("Beta", "Category A").empty());
    CHECK(store.next(1) == 0);
    CHECK(store.previous(0) == 1);
    CHECK_FALSE(store.next(2).has_value());
    CHECK(store.load(2, error).isEmpty());
    CHECK(error.isNotEmpty());
}

TEST_CASE("Explicit overwrite retains identity and rejects stale or malformed replacements",
          "[tensor-host][presets]")
{
    TestDirectory temporary;
    alfa::tensorhost::PresetStore store(temporary.file);
    juce::String error;
    REQUIRE(store.save("Alpha", "Category", document(), error));
    const auto id = store.entries()[0].id;
    const auto path = store.entries()[0].file;
    const auto original = path.loadFileAsString();
    CHECK_FALSE(store.save("alpha", "category", document(), error));
    CHECK(path.loadFileAsString() == original);
    CHECK_FALSE(store.save("Alpha", "Category", "{broken", error, id));
    CHECK(path.loadFileAsString() == original);
    CHECK_FALSE(store.save("Alpha", "Category", document(), error, "not-an-entry"));
    auto edited = JSON::parse(document().toStdString());
    edited["description"] = "Replacement metadata";
    REQUIRE(store.save("Renamed", "New Category", juce::String(edited.dump()), error, id));
    REQUIRE(store.entries().size() == 1);
    CHECK(store.entries()[0].id == id);
    CHECK(store.entries()[0].file == path);
    CHECK(store.entries()[0].name == "Renamed");
    auto external = JSON::parse(path.loadFileAsString().toStdString());
    external["description"] = "External edit";
    REQUIRE(path.replaceWithText(juce::String(external.dump())));
    CHECK_FALSE(store.save("Renamed", "New Category", document(), error, id));
    CHECK(error.containsIgnoreCase("changed"));
    CHECK(JSON::parse(path.loadFileAsString().toStdString()).at("description") == "External edit");
}

TEST_CASE("Malformed and unsupported presets never enter a usable library", "[tensor-host][presets]")
{
    TestDirectory temporary;
    alfa::tensorhost::PresetStore store(temporary.file);
    juce::String error;
    REQUIRE(store.save("Valid", "", document(), error));
    const auto badFile = temporary.file.getChildFile("broken.amo.json");
    REQUIRE(badFile.replaceWithText("{broken"));
    CHECK_FALSE(store.refresh(error));
    CHECK(error.contains("broken.amo.json"));
    REQUIRE(store.entries().size() == 1);
    CHECK(store.load(0, error).isNotEmpty());
    auto future = JSON::parse(document().toStdString());
    future["preset"] = {{"schema", "amo.preset/999"}, {"id", "future"}, {"name", "Future"}};
    CHECK_FALSE(store.save("Future", "", juce::String(future.dump()), error));
    CHECK_FALSE(store.save("Wrapper", "", "{\"document\":\"sound\",\"mode\":1}", error));
    const auto retained = store.entries()[0].file;
    REQUIRE(retained.replaceWithText("{}"));
    CHECK(store.load(0, error).isEmpty());
    CHECK(error.isNotEmpty());
}

TEST_CASE("Preset validation respects compiler and optional host capacity refusal", "[tensor-host][presets]")
{
    TestDirectory temporary;
    juce::String error;
    auto insufficient = JSON::parse(document().toStdString());
    insufficient["budget"]["active_primitives"] = 0;
    alfa::tensorhost::PresetStore store(temporary.file);
    CHECK_FALSE(store.save("Rejected", "", juce::String(insufficient.dump()), error));
    CHECK_FALSE(temporary.file.exists());
    auto incompatible = JSON::parse(document().toStdString());
    incompatible["runtimeModel"] = JSON::parse(juce::File(juce::String(ALFA_SOURCE_DIR))
                                                   .getChildFile("training/amo/testdata/runtime_gru.json")
                                                   .loadFileAsString()
                                                   .toStdString());
    incompatible["runtimeModel"]["budget"]["available_ram_bytes"] = 1;
    CHECK_FALSE(store.save("Rejected Model", "", juce::String(incompatible.dump()), error));
    alfa::tensorhost::PresetStore hostRefusal(temporary.file,
                                              [](const auto&, auto& diagnostic)
                                              {
                                                  diagnostic = "Host refuses this configuration";
                                                  return false;
                                              });
    CHECK_FALSE(hostRefusal.save("Host Rejected", "", document(), error));
    CHECK(error.contains("Host refuses"));
    CHECK_FALSE(temporary.file.exists());
}

TEST_CASE("Preset library rejects sounds that fail complete native lowering",
          "[tensor-host][presets][validation]")
{
    juce::String error;
    auto invalid = JSON::parse(document().toStdString());
    invalid["outputMix"][0][1] = 2;
    CHECK_FALSE(alfa::tensorhost::PresetStore::validateDocument(invalid.dump(), error));
    for (const auto& mix :
         {JSON::array({JSON::array({0.5, 0, 1.})}), JSON::array({JSON::array({0, 0.5, 1.})}),
          JSON::array({JSON::array({0, 0, 1., 2.})}), JSON::object()})
    {
        invalid["outputMix"] = mix;
        CHECK_FALSE(alfa::tensorhost::PresetStore::validateDocument(invalid.dump(), error));
    }
    invalid = JSON::parse(document().toStdString());
    invalid["control_layers"] = JSON::array(
        {{{"values", {0.}},
          {"ranges", {{-1., 1.}}},
          {"matrix", {{"axes", {"destination", "source"}}, {"shape", {1, 1}}, {"values", {1.}}}}}});
    CHECK_FALSE(alfa::tensorhost::PresetStore::validateDocument(invalid.dump(), error));
    invalid = JSON::parse(document().toStdString());
    invalid["runtimeModels"] = JSON::array({{{"id", "bad"}}});
    CHECK_FALSE(alfa::tensorhost::PresetStore::validateDocument(invalid.dump(), error));
    invalid = JSON::parse(document().toStdString());
    invalid["tier"] = "Pro";
    CHECK_FALSE(alfa::tensorhost::PresetStore::validateDocument(invalid.dump(), error));
}

TEST_CASE("Preset and execution validation share strict native performance policies",
          "[tensor-host][presets][validation]")
{
    const auto source = JSON::parse(document().toStdString());
    juce::String error;
    for (const auto& performance :
         {JSON(0), JSON{{"midi_layout", "unknown"}},
          JSON{{"midi_layout", "voice_indexed"}, {"voice_capacity", 1.5}, {"release_seconds", .1}},
          JSON{{"midi_layout", "voice_indexed"}, {"voice_capacity", 129}, {"release_seconds", .1}},
          JSON{{"midi_layout", "voice_indexed"}, {"voice_capacity", 8}, {"release_seconds", 601}}})
    {
        auto invalid = source;
        invalid["performance"] = performance;
        CHECK_FALSE(alfa::tensorhost::PresetStore::validateDocument(invalid.dump(), error));
    }
    auto invalid = source;
    invalid["tail_seconds"] = -1;
    CHECK_FALSE(alfa::tensorhost::PresetStore::validateDocument(invalid.dump(), error));
    invalid = source;
    invalid["preferred_mode"] = .5;
    CHECK_FALSE(alfa::tensorhost::PresetStore::validateDocument(invalid.dump(), error));
}
