#pragma once

#include "alfa/tensor/Runtime.h"

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace alfa::tensor
{

struct LegacySnapshot
{
    // Opaque original serialization: retain host/project bytes and format identity exactly.
    // This is an in-memory migration boundary, not a new plugin file format.
    std::string format;
    std::vector<std::byte> bytes;
};

// Off-thread semantic translators may be supplied after the ontology is established.
// Referenced primitive descriptors must remain valid through Runtime::prepare.
using LegacyTranslator = bool (*)(const LegacySnapshot&, Configuration&, std::string& error);

enum class MigrationStatus
{
    LegacyOnly,
    TensorCandidate
};

class Migration
{
public:
    explicit Migration(LegacySnapshot original)
        : original_(std::move(original))
    {
    }

    [[nodiscard]] const LegacySnapshot& original() const noexcept { return original_; }
    [[nodiscard]] MigrationStatus status() const noexcept { return status_; }
    [[nodiscard]] const Runtime& candidate() const noexcept { return candidate_; }

    // Produces a reviewable candidate. This never switches the plugin's execution authority
    // or overwrites original bytes. No method-to-tensor conversion is invented implicitly.
    bool translate(LegacyTranslator translator, const Budget& budget, double sampleRate, std::string& error)
    {
        error.clear();
        if (translator == nullptr)
        {
            error = "A semantic legacy translator is required; retain the legacy entry";
            return false;
        }
        Configuration translated;
        if (!translator(original_, translated, error))
        {
            if (error.empty())
                error = "Legacy translation was refused";
            return false;
        }
        Runtime proposed;
        if (!proposed.prepare(translated, budget, sampleRate, error))
            return false;
        candidate_ = std::move(proposed);
        status_ = MigrationStatus::TensorCandidate;
        return true;
    }

private:
    LegacySnapshot original_;
    Runtime candidate_;
    MigrationStatus status_ = MigrationStatus::LegacyOnly;
};

} // namespace alfa::tensor
