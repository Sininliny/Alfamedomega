#include "TestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <fstream>

using namespace alfa::inference;

namespace
{

std::filesystem::path scratchDir(const std::string& name)
{
    auto dir = std::filesystem::temp_directory_path() / ("alfa-inference-" + name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void writeFile(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream(path, std::ios::binary) << text;
}

} // namespace

TEST_CASE("sha256 matches known vectors", "[inference][manifest]")
{
    const auto dir = scratchDir("sha");
    writeFile(dir / "abc", "abc");
    writeFile(dir / "empty", "");
    REQUIRE(sha256OfFile(dir / "abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    REQUIRE(sha256OfFile(dir / "empty") ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    // Crosses the 55/56/64-byte padding edges.
    writeFile(dir / "long", std::string(1000, 'a'));
    REQUIRE(sha256OfFile(dir / "long") == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    REQUIRE(sha256OfFile(dir / "missing").empty());
}

TEST_CASE("manifest picks the newest version and falls back to the nearest tier", "[inference][manifest]")
{
    const auto dir = scratchDir("manifest");
    writeFile(dir / "manifest.json", R"({
      "schema": 1,
      "models": [
        {"tier": "standard", "version": "1.9.0", "folder": "standard-1.9.0", "files": {}},
        {"tier": "standard", "version": "1.10.0", "folder": "standard-1.10.0", "files": {}},
        {"tier": "lite", "version": "1.0.0", "folder": "lite-1.0.0", "files": {}}
      ]})");
    ModelLibrary library;
    std::string error;
    REQUIRE(library.load(dir, error));
    REQUIRE(library.entries().size() == 3);
    REQUIRE(library.find(Tier::Standard)->version == "1.10.0");
    REQUIRE(library.find(Tier::Studio) == nullptr);
    REQUIRE(library.findNearest(Tier::Studio)->tier == Tier::Standard);
    REQUIRE(library.findNearest(Tier::Pro)->tier == Tier::Standard);
    REQUIRE(library.findNearest(Tier::Lite)->tier == Tier::Lite);
}

TEST_CASE("manifest errors are reported, not thrown", "[inference][manifest]")
{
    const auto dir = scratchDir("bad-manifest");
    ModelLibrary library;
    std::string error;
    REQUIRE_FALSE(library.load(dir, error));
    REQUIRE(error.find("no manifest") != std::string::npos);

    writeFile(dir / "manifest.json", R"({"schema": 1, "models": [{"tier": "huge"}]})");
    REQUIRE_FALSE(library.load(dir, error));
    REQUIRE(error.find("huge") != std::string::npos);

    writeFile(dir / "manifest.json", "{ not json");
    REQUIRE_FALSE(library.load(dir, error));
    REQUIRE(library.entries().empty());
}

TEST_CASE("verify catches a modified model file", "[inference][manifest]")
{
    const auto dir = scratchDir("verify");
    std::filesystem::create_directories(dir / "lite-1");
    writeFile(dir / "lite-1" / "a.onnx", "abc");
    ModelEntry entry;
    entry.folder = dir / "lite-1";
    // Upper case, as some tools print it, is accepted.
    entry.files["a.onnx"] = {"BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD", 3};
    std::string error;
    REQUIRE(ModelLibrary::verify(entry, error));

    writeFile(dir / "lite-1" / "a.onnx", "abd");
    REQUIRE_FALSE(ModelLibrary::verify(entry, error));
    REQUIRE(error.find("checksum mismatch") != std::string::npos);

    writeFile(dir / "lite-1" / "a.onnx", "abcd");
    REQUIRE_FALSE(ModelLibrary::verify(entry, error));
    REQUIRE(error.find("size mismatch") != std::string::npos);
}

TEST_CASE("manifest reads the control center's model-store schema", "[inference][manifest]")
{
    const auto dir = scratchDir("store");
    // As in docs/control-center/model-store.md.
    writeFile(dir / "manifest.json", R"({
      "schema": 1,
      "default_tier": "pro",
      "models": [
        {"id": "pro-2026-09-29", "tier": "pro", "version": "1.0.0", "dir": "pro-1.0.0",
         "files": [{"path": "voice_core.onnx", "sha256": "ab", "bytes": 12}],
         "bytes": 12, "sr": 48000, "frame_hz": 250, "hop": 192, "source": "trained", "installed_at": "x"},
        {"id": "half-installed", "tier": "lite", "version": "9.0.0", "dir": ".staging-lite", "files": []}
      ]})");
    ModelLibrary library;
    std::string error;
    REQUIRE(library.load(dir, error));
    REQUIRE(library.defaultTier() == Tier::Pro);
    REQUIRE(library.entries().size() == 1); // the dot folder is skipped
    const auto* pro = library.findById("pro-2026-09-29");
    REQUIRE(pro != nullptr);
    REQUIRE(pro->folder == dir / "pro-1.0.0");
    REQUIRE(pro->frameRateHz == 250.0);
    REQUIRE(pro->files.at("voice_core.onnx").sha256 == "ab");
    REQUIRE(pro->files.at("voice_core.onnx").size == 12u);
    REQUIRE(library.findById("half-installed") == nullptr);
}

TEST_CASE("manifest also reads the training export's first form", "[inference][manifest]")
{
    const auto dir = scratchDir("training-form");
    writeFile(dir / "manifest.json", R"({
      "schema": 1,
      "models": [{"tier": "studio", "version": "0.1.0", "folder": "studio-0.1.0", "frame_rate_hz": 250,
                  "files": {"voice_core.onnx": "cd"}}]})");
    ModelLibrary library;
    std::string error;
    REQUIRE(library.load(dir, error));
    REQUIRE(library.defaultTier() == Tier::Standard);
    const auto* studio = library.find(Tier::Studio);
    REQUIRE(studio != nullptr);
    REQUIRE(studio->id == "studio-0.1.0");
    REQUIRE(studio->folder == dir / "studio-0.1.0");
    REQUIRE(studio->files.at("voice_core.onnx").sha256 == "cd");
    REQUIRE_FALSE(studio->files.at("voice_core.onnx").size.has_value());
}

TEST_CASE("manifest refuses a newer schema and paths that leave the models folder", "[inference][manifest]")
{
    const auto dir = scratchDir("refuse");
    ModelLibrary library;
    std::string error;
    writeFile(dir / "manifest.json", R"({"schema": 2, "models": []})");
    REQUIRE_FALSE(library.load(dir, error));
    REQUIRE(error.find("schema 2") != std::string::npos);

    writeFile(
        dir / "manifest.json",
        R"({"schema": 1, "models": [{"tier": "lite", "version": "1", "dir": "../elsewhere", "files": []}]})");
    REQUIRE_FALSE(library.load(dir, error));
    REQUIRE(error.find("not allowed") != std::string::npos);

    writeFile(dir / "manifest.json",
              R"({"schema": 1, "models": [{"tier": "lite", "version": "1", "dir": "lite",
              "files": [{"path": "/etc/passwd", "sha256": "00"}]}]})");
    REQUIRE_FALSE(library.load(dir, error));
    REQUIRE(error.find("not allowed") != std::string::npos);
}

TEST_CASE("ALFA_MODELS_DIR overrides the default models folder", "[inference][manifest]")
{
#if !defined(_WIN32)
    ::unsetenv("ALFA_MODELS_DIR");
    REQUIRE(ModelLibrary::defaultModelsDir().filename() == "Models");
    ::setenv("ALFA_MODELS_DIR", "/tmp/alfa-models", 1);
    REQUIRE(ModelLibrary::defaultModelsDir() == std::filesystem::path("/tmp/alfa-models"));
    ::unsetenv("ALFA_MODELS_DIR");
#endif
}

TEST_CASE("the generated dummy models are listed and intact", "[inference][manifest]")
{
    const auto library = test::loadTestLibrary();
    for (const auto& s : kTiers)
    {
        const auto& entry = test::requireModel(library, s.tier);
        std::string error;
        INFO(error);
        REQUIRE(ModelLibrary::verify(entry, error));
        // Encoders, instance step, voice core, one expert per method, grain bank, export info.
        REQUIRE(entry.files.size() == 4 + kNumMethods + 2);
        REQUIRE(entry.files.at("grain_bank.npy").size.has_value());
    }
}
