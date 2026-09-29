#include "TestSupport.h"

#include <catch2/catch_test_macros.hpp>

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
    entry.files["a.onnx"] = "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD";
    std::string error;
    REQUIRE(ModelLibrary::verify(entry, error));

    writeFile(dir / "lite-1" / "a.onnx", "abd");
    REQUIRE_FALSE(ModelLibrary::verify(entry, error));
    REQUIRE(error.find("checksum mismatch") != std::string::npos);
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
        REQUIRE(entry.files.size() == 3 + kNumMethods);
    }
}
