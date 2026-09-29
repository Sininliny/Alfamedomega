#pragma once

#include "alfa/inference/Tier.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace alfa::inference
{

struct ModelFile
{
    std::string sha256;                 ///< Lower- or upper-case hex.
    std::optional<std::uintmax_t> size; ///< Bytes, when the manifest gives them.
};

/// One installed model: a tier at a model version, in its own folder under the models folder.
struct ModelEntry
{
    std::string id; ///< "<tier>-<version>"; presets may pin it.
    Tier tier = kDefaultTier;
    std::string version;
    std::filesystem::path folder;           ///< Absolute path to the model's folder.
    std::map<std::string, ModelFile> files; ///< Relative path in the folder to its checksum.
    double frameRateHz = 250.0;
};

/// The shared models folder and its manifest.json, schema 1, as written by the control center
/// (docs/control-center/model-store.md) and by the training export.
///
/// Message thread only: reads files and allocates.
class ModelLibrary
{
public:
    static constexpr int kSchema = 1;

    /// Where the installer puts models. ALFA_MODELS_DIR overrides it.
    ///   macOS   /Library/Application Support/Alfamedomega/Models
    ///   Windows %ProgramData%\Alfamedomega\Models
    ///   Linux   $XDG_DATA_HOME/Alfamedomega/Models (default ~/.local/share/...)
    [[nodiscard]] static std::filesystem::path defaultModelsDir();

    /// Reads <dir>/manifest.json. Returns false and sets error if it is missing, malformed or
    /// from a newer schema.
    bool load(const std::filesystem::path& dir, std::string& error);

    [[nodiscard]] const std::vector<ModelEntry>& entries() const noexcept { return entries_; }

    /// The size a new plugin instance starts with.
    [[nodiscard]] Tier defaultTier() const noexcept { return defaultTier_; }

    /// Newest installed version of this tier, or nullptr.
    [[nodiscard]] const ModelEntry* find(Tier tier) const noexcept;

    /// The entry with this id (a preset that pins a model version), or nullptr.
    [[nodiscard]] const ModelEntry* findById(const std::string& id) const noexcept;

    /// The requested tier if installed, otherwise the nearest installed tier, preferring smaller
    /// ones (a project saved on Studio opens on Pro before Standard). nullptr if nothing is
    /// installed. The caller must tell the user when the result is not the requested tier.
    [[nodiscard]] const ModelEntry* findNearest(Tier requested) const noexcept;

    /// Checks every file of the entry against its size and SHA-256 in the manifest.
    [[nodiscard]] static bool verify(const ModelEntry& entry, std::string& error);

private:
    std::vector<ModelEntry> entries_;
    Tier defaultTier_ = kDefaultTier;
};

/// Lower-case hex SHA-256 of a file. Empty string if it cannot be read.
[[nodiscard]] std::string sha256OfFile(const std::filesystem::path& path);

} // namespace alfa::inference
