#pragma once

#include "alfa/inference/Tier.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace alfa::inference
{

/// One installed model: a tier at a model version, in its own folder under the models folder.
struct ModelEntry
{
    Tier tier = kDefaultTier;
    std::string version;
    std::filesystem::path folder;             ///< Absolute path to the tier's folder.
    std::map<std::string, std::string> files; ///< File name to expected SHA-256 (lower-case hex).
    int opset = 0;
    double frameRateHz = 250.0;
};

/// The shared models folder and its manifest.json (docs/packaging/packaging-plan.md section 3).
///
/// Message thread only: reads files and allocates.
class ModelLibrary
{
public:
    /// Where the installer puts models. ALFAMEDOMEGA_MODELS_DIR overrides it.
    ///   macOS   /Library/Application Support/Alfamedomega/Models
    ///   Windows %ProgramData%\Alfamedomega\Models
    ///   Linux   /usr/share/alfamedomega/models
    [[nodiscard]] static std::filesystem::path defaultModelsDir();

    /// Reads <dir>/manifest.json. Returns false and sets error if it is missing or malformed.
    bool load(const std::filesystem::path& dir, std::string& error);

    [[nodiscard]] const std::vector<ModelEntry>& entries() const noexcept { return entries_; }

    /// Newest installed version of this tier, or nullptr.
    [[nodiscard]] const ModelEntry* find(Tier tier) const noexcept;

    /// The requested tier if installed, otherwise the nearest installed tier, preferring smaller
    /// ones (a project saved on Studio opens on Pro before Standard). nullptr if nothing is
    /// installed. The caller must tell the user when the result is not the requested tier.
    [[nodiscard]] const ModelEntry* findNearest(Tier requested) const noexcept;

    /// Checks every file of the entry against its SHA-256 in the manifest.
    [[nodiscard]] static bool verify(const ModelEntry& entry, std::string& error);

private:
    std::vector<ModelEntry> entries_;
};

/// Lower-case hex SHA-256 of a file. Empty string if it cannot be read.
[[nodiscard]] std::string sha256OfFile(const std::filesystem::path& path);

} // namespace alfa::inference
