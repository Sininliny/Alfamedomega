#pragma once

#include <juce_core/juce_core.h>

#include <functional>
#include <optional>
#include <vector>

namespace alfa::tensorhost
{
struct PresetEntry
{
    juce::String id, name, category;
    juce::File file;
    juce::String fingerprint;
};

// Filesystem-only sound library. Constructing/browsing never creates folders or
// writes presets. The caller passes sound JSON, not a host/UI-state wrapper.
// All operations belong off the audio thread; the editor serializes its calls.
class PresetStore
{
public:
    using Validator = std::function<bool(const juce::String&, juce::String&)>;
    explicit PresetStore(juce::File directory, Validator validator = {});
    static juce::File defaultDirectory();
    static juce::String presetName(const juce::String& document);
    static juce::String presetCategory(const juce::String& document);
    static bool validateDocument(const juce::String& document, juce::String& error);

    // Malformed files are excluded and reported; valid entries remain available.
    bool refresh(juce::String& error);
    const std::vector<PresetEntry>& entries() const noexcept { return entries_; }
    const juce::File& directory() const noexcept { return directory_; }
    std::vector<std::size_t> search(const juce::String& query, const juce::String& category = {}) const;
    std::optional<std::size_t> next(std::size_t index) const noexcept;
    std::optional<std::size_t> previous(std::size_t index) const noexcept;

    // Save As always creates a new UUID filename. Duplicate name/category is
    // refused unless the caller explicitly identifies the entry to overwrite.
    // Overwrite refuses on-disk changes since refresh and symbolic-link targets.
    bool save(const juce::String& name, const juce::String& category, const juce::String& document,
              juce::String& error, const juce::String& overwriteId = {});
    juce::String load(std::size_t index, juce::String& error) const;

private:
    bool validate(const juce::String& document, juce::String& error) const;
    juce::File directory_;
    Validator validator_;
    std::vector<PresetEntry> entries_;
};
} // namespace alfa::tensorhost
