#include "PresetStore.h"

#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Model.h"
#include "alfa/tensor/Precision.h"

#include <juce_cryptography/juce_cryptography.h>

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <set>

namespace alfa::tensorhost
{
namespace
{
using JSON = nlohmann::json;

juce::String stringField(const JSON& object, const char* key)
{
    if (object.is_object() && object.contains(key) && object.at(key).is_string())
        return juce::String(object.at(key).get<std::string>());
    return {};
}
juce::String displayField(const JSON& object, const char* key)
{
    if (object.contains("preset"))
        if (auto value = stringField(object.at("preset"), key); value.isNotEmpty())
            return value;
    if (auto value = stringField(object, key); value.isNotEmpty())
        return value;
    return object.contains("metadata") ? stringField(object.at("metadata"), key) : juce::String{};
}
juce::String fingerprint(const juce::String& text)
{
    return juce::SHA256(text.toRawUTF8(), static_cast<std::size_t>(text.getNumBytesAsUTF8())).toHexString();
}
PresetEntry entryFor(const juce::File& file, const juce::String& text)
{
    const auto object = JSON::parse(text.toStdString());
    const auto hash = fingerprint(text);
    auto name = displayField(object, "name");
    if (name.isEmpty())
    {
        name = file.getFileNameWithoutExtension();
        if (name.endsWithIgnoreCase(".aom"))
            name = name.dropLastCharacters(4);
    }
    auto id = object.contains("preset") ? stringField(object.at("preset"), "id") : juce::String{};
    if (id.isEmpty())
        id = hash;
    return {id, name, displayField(object, "category"), file, hash};
}
} // namespace

PresetStore::PresetStore(juce::File directory, Validator validator)
    : directory_(std::move(directory))
    , validator_(std::move(validator))
{
}

juce::File PresetStore::defaultDirectory()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Alfamedomega")
        .getChildFile("TensorPresets");
}
juce::String PresetStore::presetName(const juce::String& document)
{
    try
    {
        return displayField(JSON::parse(document.toStdString()), "name");
    }
    catch (const std::exception&)
    {
        return {};
    }
}
juce::String PresetStore::presetCategory(const juce::String& document)
{
    try
    {
        return displayField(JSON::parse(document.toStdString()), "category");
    }
    catch (const std::exception&)
    {
        return {};
    }
}

bool PresetStore::validateDocument(const juce::String& document, juce::String& error)
{
    try
    {
        const auto object = JSON::parse(document.toStdString());
        if (!object.is_object() || object.value("schema", "") != "aom.tensor/1")
            throw std::invalid_argument("Preset must contain a complete aom.tensor/1 sound document");
        if (object.contains("preset"))
        {
            const auto& metadata = object.at("preset");
            if (!metadata.is_object() || metadata.value("schema", "") != "aom.preset/1" ||
                stringField(metadata, "id").isEmpty() || stringField(metadata, "name").trim().isEmpty())
                throw std::invalid_argument("Unsupported or incomplete aom.preset/1 metadata");
            if (metadata.contains("category") && !metadata.at("category").is_string())
                throw std::invalid_argument("Preset category must be text");
        }
        std::string diagnostic;
        alfa::tensor::PrecisionProfile precision;
        if (!alfa::tensor::resolvePrecisionJSON(document.toStdString(), precision, diagnostic))
            throw std::invalid_argument(diagnostic);
        // Off-thread reference validation only. Loading into a host revalidates
        // against its actual sample rate and registered primitive implementations.
        const auto validateTyped = [&]<typename Scalar>()
        {
            alfa::tensor::ConfigurationT<Scalar> configuration;
            alfa::tensor::Budget budget;
            alfa::tensor::RuntimeT<Scalar> runtime;
            if (!alfa::tensor::parseJSON(document.toStdString(), configuration, diagnostic) ||
                !alfa::tensor::readBudgetJSON(document.toStdString(), budget, diagnostic) ||
                !runtime.prepare(configuration, budget, 48000, diagnostic))
                throw std::invalid_argument(diagnostic);
            if (object.contains("runtimeModel"))
            {
                alfa::tensor::TensorModelT<Scalar> model;
                if (!model.prepare(object.at("runtimeModel").dump(), configuration, diagnostic))
                    throw std::invalid_argument(diagnostic);
            }
            return runtime.output().size();
        };
        const auto outputs = precision.scalarType == alfa::tensor::ScalarType::Float64
                                 ? validateTyped.template operator()<double>()
                                 : validateTyped.template operator()<float>();
        if (!object.contains("outputMix") || !object.at("outputMix").is_array())
            throw std::invalid_argument("Preset requires explicit outputMix");
        for (const auto& row : object.at("outputMix"))
        {
            if (!row.is_array() || row.size() != 3 || !row.at(0).is_number_integer() ||
                !row.at(1).is_number_integer() || !row.at(2).is_number())
                throw std::invalid_argument("outputMix requires output/channel/gain triples");
            const auto output = row.at(0).get<std::int64_t>();
            if (output < 0 || static_cast<std::size_t>(output) >= outputs ||
                row.at(1).get<std::int64_t>() < 0 || !std::isfinite(row.at(2).get<double>()))
                throw std::invalid_argument("Invalid outputMix index, channel or gain");
        }
        error.clear();
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}

bool PresetStore::validate(const juce::String& document, juce::String& error) const
{
    return validateDocument(document, error) && (!validator_ || validator_(document, error));
}

bool PresetStore::refresh(juce::String& error)
{
    entries_.clear();
    error.clear();
    if (!directory_.exists())
        return true;
    if (!directory_.isDirectory())
    {
        error = "Preset library location is not a directory";
        return false;
    }
    juce::Array<juce::File> files;
    directory_.findChildFiles(files, juce::File::findFiles, false, "*.aom.json");
    std::set<std::string> ids;
    for (const auto& file : files)
    {
        const auto text = file.loadFileAsString();
        juce::String diagnostic;
        if (!validate(text, diagnostic))
        {
            error += file.getFileName() + ": " + diagnostic + "\n";
            continue;
        }
        auto entry = entryFor(file, text);
        if (!ids.insert(entry.id.toStdString()).second)
        {
            error += file.getFileName() + ": Duplicate preset identity\n";
            continue;
        }
        entries_.push_back(std::move(entry));
    }
    std::sort(entries_.begin(), entries_.end(),
              [](const auto& first, const auto& second)
              {
                  const auto name = first.name.compareIgnoreCase(second.name);
                  if (name != 0)
                      return name < 0;
                  const auto category = first.category.compareIgnoreCase(second.category);
                  return category != 0 ? category < 0 : first.id < second.id;
              });
    return error.isEmpty();
}

std::vector<std::size_t> PresetStore::search(const juce::String& query, const juce::String& category) const
{
    std::vector<std::size_t> results;
    for (std::size_t index = 0; index < entries_.size(); ++index)
    {
        const auto& entry = entries_[index];
        if ((category.isEmpty() || entry.category.equalsIgnoreCase(category)) &&
            (query.isEmpty() || entry.name.containsIgnoreCase(query) ||
             entry.category.containsIgnoreCase(query)))
            results.push_back(index);
    }
    return results;
}
std::optional<std::size_t> PresetStore::next(std::size_t index) const noexcept
{
    if (index >= entries_.size())
        return {};
    return (index + 1) % entries_.size();
}
std::optional<std::size_t> PresetStore::previous(std::size_t index) const noexcept
{
    if (index >= entries_.size())
        return {};
    return (index + entries_.size() - 1) % entries_.size();
}

bool PresetStore::save(const juce::String& name, const juce::String& category, const juce::String& document,
                       juce::String& error, const juce::String& overwriteId)
{
    if (name.trim().isEmpty())
    {
        error = "Enter a preset name";
        return false;
    }
    if (!validate(document, error))
        return false;
    if (overwriteId.isEmpty())
    {
        juce::String scanWarnings;
        refresh(scanWarnings); // Observe additions even when caller did not browse first.
        if (directory_.exists() && !directory_.isDirectory())
        {
            error = scanWarnings;
            return false;
        }
    }
    const PresetEntry* replacing = nullptr;
    for (const auto& entry : entries_)
    {
        if (entry.id == overwriteId && overwriteId.isNotEmpty())
            replacing = &entry;
        if (entry.name.equalsIgnoreCase(name.trim()) && entry.category.equalsIgnoreCase(category.trim()) &&
            entry.id != overwriteId)
        {
            error = "A preset with this name/category already exists; select it explicitly to overwrite";
            return false;
        }
    }
    if (overwriteId.isNotEmpty() && replacing == nullptr)
    {
        error = "Preset to overwrite is not in the refreshed library";
        return false;
    }
    if (replacing && (replacing->file.isSymbolicLink() || !replacing->file.existsAsFile() ||
                      fingerprint(replacing->file.loadFileAsString()) != replacing->fingerprint))
    {
        error = "Preset changed on disk or is a symbolic link; refresh before overwriting";
        return false;
    }
    try
    {
        auto object = JSON::parse(document.toStdString());
        auto metadata = object.value("preset", JSON::object());
        const auto id = replacing ? replacing->id : juce::Uuid().toString();
        metadata["schema"] = "aom.preset/1";
        metadata["id"] = id.toStdString();
        metadata["name"] = name.trim().toStdString();
        metadata["category"] = category.trim().toStdString();
        object["preset"] = std::move(metadata);
        const auto result = juce::String(object.dump(2));
        if (!validate(result, error))
            return false;
        if (!directory_.createDirectory())
        {
            error = "Cannot create preset library directory";
            return false;
        }
        const auto target = replacing ? replacing->file : directory_.getChildFile(id + ".aom.json");
        if (!replacing && target.exists())
        {
            error = "New preset identity unexpectedly already exists";
            return false;
        }
        juce::TemporaryFile temporary(target);
        if (!temporary.getFile().replaceWithText(result) || !temporary.overwriteTargetFileWithTemporary())
        {
            error = "Cannot atomically save preset; original file retained";
            return false;
        }
        juce::String refreshError;
        refresh(refreshError);
        error.clear();
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}

juce::String PresetStore::load(std::size_t index, juce::String& error) const
{
    if (index >= entries_.size())
    {
        error = "Preset selection is out of range";
        return {};
    }
    const auto& entry = entries_[index];
    if (!entry.file.existsAsFile())
    {
        error = "Preset file is missing; refresh the library";
        return {};
    }
    const auto text = entry.file.loadFileAsString();
    if (!validate(text, error))
        return {};
    const auto current = entryFor(entry.file, text);
    if (current.id != entry.id)
    {
        error = "Preset identity changed on disk; refresh the library";
        return {};
    }
    error.clear();
    return text;
}
} // namespace alfa::tensorhost
