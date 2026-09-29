#include "alfa/inference/ModelLibrary.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

namespace alfa::inference
{
namespace
{

// Orders "1.10.0" after "1.9.0". Pre-release suffixes ("0.0.0-dummy") sort by their text.
bool versionLess(const std::string& a, const std::string& b)
{
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size())
    {
        if (std::isdigit(static_cast<unsigned char>(a[i])) && std::isdigit(static_cast<unsigned char>(b[j])))
        {
            std::size_t ei = i, ej = j;
            while (ei < a.size() && std::isdigit(static_cast<unsigned char>(a[ei])))
                ++ei;
            while (ej < b.size() && std::isdigit(static_cast<unsigned char>(b[ej])))
                ++ej;
            const auto na = std::stoull(a.substr(i, ei - i));
            const auto nb = std::stoull(b.substr(j, ej - j));
            if (na != nb)
                return na < nb;
            i = ei;
            j = ej;
        }
        else
        {
            if (a[i] != b[j])
                return a[i] < b[j];
            ++i;
            ++j;
        }
    }
    return a.size() - i < b.size() - j;
}

} // namespace

std::filesystem::path ModelLibrary::defaultModelsDir()
{
    if (const char* overrideDir = std::getenv("ALFAMEDOMEGA_MODELS_DIR"); overrideDir && *overrideDir)
        return overrideDir;
#if defined(__APPLE__)
    return "/Library/Application Support/Alfamedomega/Models";
#elif defined(_WIN32)
    const char* programData = std::getenv("ProgramData");
    return std::filesystem::path(programData ? programData : "C:\\ProgramData") / "Alfamedomega" / "Models";
#else
    return "/usr/share/alfamedomega/models";
#endif
}

bool ModelLibrary::load(const std::filesystem::path& dir, std::string& error)
{
    entries_.clear();
    const auto manifestPath = dir / "manifest.json";
    std::ifstream in(manifestPath);
    if (!in)
    {
        error = "no manifest at " + manifestPath.string();
        return false;
    }

    try
    {
        const auto json = nlohmann::json::parse(in);
        if (json.value("schema", 0) != 1)
        {
            error = "unsupported manifest schema in " + manifestPath.string();
            return false;
        }
        for (const auto& m : json.at("models"))
        {
            const auto tierName = m.at("tier").get<std::string>();
            const auto tier = tierFromString(tierName);
            if (!tier)
            {
                error = "unknown tier '" + tierName + "' in manifest";
                return false;
            }
            ModelEntry entry;
            entry.tier = *tier;
            entry.version = m.at("version").get<std::string>();
            entry.folder = dir / m.at("folder").get<std::string>();
            entry.opset = m.value("opset", 0);
            entry.frameRateHz = m.value("frame_rate_hz", 250.0);
            for (const auto& [name, hash] : m.at("files").items())
                entry.files.emplace(name, hash.get<std::string>());
            entries_.push_back(std::move(entry));
        }
    }
    catch (const std::exception& e)
    {
        error = "malformed manifest " + manifestPath.string() + ": " + e.what();
        entries_.clear();
        return false;
    }
    return true;
}

const ModelEntry* ModelLibrary::find(Tier tier) const noexcept
{
    const ModelEntry* best = nullptr;
    for (const auto& e : entries_)
        if (e.tier == tier && (best == nullptr || versionLess(best->version, e.version)))
            best = &e;
    return best;
}

const ModelEntry* ModelLibrary::findNearest(Tier requested) const noexcept
{
    const int r = static_cast<int>(requested);
    // Distance 0 first, then one step smaller, one step larger, two smaller, and so on.
    for (int distance = 0; distance < static_cast<int>(kTiers.size()); ++distance)
    {
        for (const int candidate : {r - distance, r + distance})
        {
            if (candidate < 0 || candidate >= static_cast<int>(kTiers.size()))
                continue;
            if (const auto* e = find(static_cast<Tier>(candidate)))
                return e;
        }
    }
    return nullptr;
}

bool ModelLibrary::verify(const ModelEntry& entry, std::string& error)
{
    for (const auto& [name, expected] : entry.files)
    {
        const auto path = entry.folder / name;
        const auto actual = sha256OfFile(path);
        if (actual.empty())
        {
            error = "cannot read " + path.string();
            return false;
        }
        std::string expectedLower = expected;
        std::transform(expectedLower.begin(), expectedLower.end(), expectedLower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (actual != expectedLower)
        {
            error = "checksum mismatch for " + path.string();
            return false;
        }
    }
    return true;
}

} // namespace alfa::inference
