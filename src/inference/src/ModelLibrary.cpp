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
    if (const char* overrideDir = std::getenv("ALFA_MODELS_DIR"); overrideDir && *overrideDir)
        return overrideDir;
#if defined(__APPLE__)
    return "/Library/Application Support/Alfamedomega/Models";
#elif defined(_WIN32)
    const char* programData = std::getenv("ProgramData");
    return std::filesystem::path(programData ? programData : "C:\\ProgramData") / "Alfamedomega" / "Models";
#else
    if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && *xdg)
        return std::filesystem::path(xdg) / "Alfamedomega" / "Models";
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home ? home : ".") / ".local" / "share" / "Alfamedomega" / "Models";
#endif
}

namespace
{

// A folder or file name from the manifest must stay inside the models folder.
bool isSafeRelative(const std::string& name)
{
    const std::filesystem::path p(name);
    if (name.empty() || p.is_absolute() || p.has_root_name())
        return false;
    for (const auto& part : p)
        if (part == ".." || part.string().empty())
            return false;
    return true;
}

} // namespace

bool ModelLibrary::load(const std::filesystem::path& dir, std::string& error)
{
    entries_.clear();
    defaultTier_ = kDefaultTier;
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
        const int schema = json.value("schema", 0);
        if (schema < 1 || schema > kSchema)
        {
            error = "manifest schema " + std::to_string(schema) +
                    " is not supported (this build reads schema " + std::to_string(kSchema) +
                    "): " + manifestPath.string();
            return false;
        }
        if (json.contains("default_tier"))
            if (const auto t = tierFromString(json.at("default_tier").get<std::string>()))
                defaultTier_ = *t;

        for (const auto& m : json.at("models"))
        {
            const auto tierName = m.at("tier").get<std::string>();
            const auto tier = tierFromString(tierName);
            if (!tier)
            {
                error = "unknown tier '" + tierName + "' in manifest";
                entries_.clear();
                return false;
            }
            ModelEntry entry;
            entry.tier = *tier;
            entry.version = m.at("version").get<std::string>();
            entry.id = m.value("id", tierName + "-" + entry.version);
            // The control center writes "dir"; the training export's first version wrote "folder".
            const auto folder =
                m.contains("dir") ? m.at("dir").get<std::string>() : m.at("folder").get<std::string>();
            if (!isSafeRelative(folder))
            {
                error = "model folder '" + folder + "' is not allowed";
                entries_.clear();
                return false;
            }
            // Dot folders are the control center's staging and trash: never a model.
            if (folder.front() == '.')
                continue;
            entry.folder = dir / folder;
            entry.frameRateHz = m.value("frame_hz", m.value("frame_rate_hz", 250.0));

            // "files" is a list of {path, sha256, bytes}, or a map of path to sha256.
            const auto& files = m.at("files");
            if (files.is_array())
            {
                for (const auto& f : files)
                {
                    ModelFile file{f.at("sha256").get<std::string>(), std::nullopt};
                    if (f.contains("bytes"))
                        file.size = f.at("bytes").get<std::uintmax_t>();
                    entry.files.emplace(f.at("path").get<std::string>(), std::move(file));
                }
            }
            else
            {
                for (const auto& [name, hash] : files.items())
                    entry.files.emplace(name, ModelFile{hash.get<std::string>(), std::nullopt});
            }
            for (const auto& [name, file] : entry.files)
            {
                if (!isSafeRelative(name))
                {
                    error = "file name '" + name + "' in " + entry.id + " is not allowed";
                    entries_.clear();
                    return false;
                }
            }
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

const ModelEntry* ModelLibrary::findById(const std::string& id) const noexcept
{
    for (const auto& e : entries_)
        if (e.id == id)
            return &e;
    return nullptr;
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
    for (const auto& [name, file] : entry.files)
    {
        const auto path = entry.folder / name;
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec)
        {
            error = "cannot read " + path.string();
            return false;
        }
        if (file.size && *file.size != size)
        {
            error = "size mismatch for " + path.string();
            return false;
        }
        const auto actual = sha256OfFile(path);
        std::string expected = file.sha256;
        std::transform(expected.begin(), expected.end(), expected.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (actual.empty() || actual != expected)
        {
            error = "checksum mismatch for " + path.string();
            return false;
        }
    }
    return true;
}

} // namespace alfa::inference
