#include "PrimitiveFragmentPanel.h"

#include "TensorExecution.h"
#include "alfa/tensor/Precision.h"
#include "alfa/tensor/Universe.h"

#include <juce_audio_formats/juce_audio_formats.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace alfa::tensorhost
{
namespace
{
using JSON = nlohmann::json;
std::size_t index(const juce::TextEditor& editor)
{
    const auto text = editor.getText().trim().toStdString();
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("State, channel and clock fields require explicit nonnegative integers");
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Integer exceeds addressable capacity");
    return static_cast<std::size_t>(value);
}
JSON array(const juce::TextEditor& editor, bool optional = true)
{
    const auto text = editor.getText().trim();
    if (optional && text.isEmpty())
        return JSON::array();
    const auto value = JSON::parse(text.toStdString());
    if (!value.is_array())
        throw std::invalid_argument(
            "Parameter, state, asset, weight, binding and mix inputs must be JSON arrays");
    return value;
}
JSON tensor(const JSON& values, const char* axis)
{
    return {{"axes", {axis}}, {"shape", {values.size()}}, {"values", values}};
}
} // namespace

PrimitiveFragmentPanel::PrimitiveFragmentPanel(double sampleRate, juce::String tier, Publish publish,
                                               const alfa::ui::Theme* theme)
    : sampleRate_(sampleRate)
    , tier_(std::move(tier))
    , publish_(std::move(publish))
{
    if (theme != nullptr)
        look_.setTheme(*theme);
    setLookAndFeel(&look_);
    setSize(980, 740);
    const auto fullCatalog = alfa::tensor::universe::catalog();
    catalog_.assign(fullCatalog.begin(), fullCatalog.end());
    for (const auto* base :
         {&alfa::tensor::kernels::sineMetadata(), &alfa::tensor::kernels::onePoleMetadata(),
          &alfa::tensor::kernels::feedbackCellMetadata(), &alfa::tensor::kernels::delayMetadata()})
        if (std::none_of(catalog_.begin(), catalog_.end(),
                         [&](const auto& entry) { return entry.id == base->id; }))
            catalog_.push_back(*base);
    for (std::size_t entry = 0; entry < catalog_.size(); ++entry)
        operation_.addItem(juce::String(std::string(catalog_[entry].id)), static_cast<int>(entry + 1));
    operation_.setTextWhenNothingSelected("Select a mathematical operation");
    operation_.onChange = [this]
    {
        selectionChanged();
    };
    between_.addItemList({"zero", "hold"}, 1);
    between_.setTextWhenNothingSelected("Choose output between updates");
    for (auto* component : std::initializer_list<juce::Component*>{
             &operation_, &between_, &state_,    &channels_,      &period_,     &phase_,     &parameters_,
             &initial_,   &assets_,  &bindings_, &routing_,       &modulation_, &outputMix_, &budget_,
             &metadata_,  &heading_, &status_,   &publishButton_, &wavButton_,  &wavChannel_})
        addAndMakeVisible(component);
    operation_.setComponentID("primitive.operation");
    between_.setComponentID("primitive.between");
    state_.setComponentID("primitive.state_size");
    channels_.setComponentID("primitive.input_channels");
    period_.setComponentID("primitive.period");
    phase_.setComponentID("primitive.phase");
    parameters_.setComponentID("primitive.parameters");
    initial_.setComponentID("primitive.initial_state");
    assets_.setComponentID("primitive.assets");
    bindings_.setComponentID("primitive.audio_bindings");
    routing_.setComponentID("primitive.routing");
    modulation_.setComponentID("primitive.modulation");
    outputMix_.setComponentID("primitive.output_mix");
    budget_.setComponentID("primitive.budget");
    metadata_.setComponentID("primitive.metadata");
    status_.setComponentID("primitive.status");
    wavChannel_.setComponentID("primitive.wav_channel");
    heading_.setText("Create an executable mathematical fragment | tier: " + tier_ +
                         ". Author state, parameters, clock, bindings and budget. WAV decoding uses a "
                         "float32 reader boundary.",
                     juce::dontSendNotification);
    heading_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13));
    for (auto* editor : {&parameters_, &initial_, &assets_, &bindings_, &routing_, &modulation_, &outputMix_,
                         &budget_, &metadata_})
    {
        editor->setMultiLine(true);
        editor->setReturnKeyStartsNewLine(true);
        editor->setFont(alfa::ui::AlfaLookAndFeel::monoFont(12));
    }
    metadata_.setReadOnly(true);
    metadata_.setText(
        "Select an operation to inspect representations, state, parameter domains and computational cost.");
    initial_.setTooltip("Optional initial state array. Blank explicitly uses zero state of the chosen size.");
    outputMix_.setTooltip(
        "Optional [raw output coordinate, host channel, gain] triples. Blank creates no direct output mix.");
    bindings_.setTooltip("Optional native audio operands using "
                         "source/source_element/destination_element/weights/weight_element/missing_source.");
    routing_.setTooltip(
        "Explicit routing coefficients indexed by authored audio bindings. Blank means no coefficients.");
    modulation_.setTooltip("Explicit modulation coefficients. Blank means no coefficients.");
    budget_.setTooltip("Explicit native capacity budget JSON, including boundary_conversion_bytes for "
                       "float64 host execution.");
    publishButton_.onClick = [this]
    {
        create();
    };
    wavChannel_.setTextToShowWhenEmpty("WAV channel index", look_.theme().text2);
    wavChannel_.setTooltip(
        "Explicit zero-based source channel; all other WAV channels are ignored without mixing.");
    wavButton_.onClick = [this]
    {
        try
        {
            const auto channel = index(wavChannel_);
            chooser_ =
                std::make_unique<juce::FileChooser>("Import one authored WAV channel", juce::File{}, "*.wav");
            auto safe = juce::Component::SafePointer<PrimitiveFragmentPanel>(this);
            chooser_->launchAsync(
                juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                [safe, channel](const juce::FileChooser& chooser)
                {
                    if (safe == nullptr || !chooser.getResult().existsAsFile())
                        return;
                    juce::String error;
                    if (safe->loadWavAsset(chooser.getResult(), channel, error))
                        safe->status_.setText(
                            "Selected WAV channel appended as an asset; parameters are unchanged",
                            juce::dontSendNotification);
                    else
                        safe->status_.setText(error, juce::dontSendNotification);
                });
        }
        catch (const std::exception& failure)
        {
            status_.setText(failure.what(), juce::dontSendNotification);
        }
    };
}
PrimitiveFragmentPanel::~PrimitiveFragmentPanel()
{
    setLookAndFeel(nullptr);
}
void PrimitiveFragmentPanel::selectionChanged()
{
    const auto selected = operation_.getSelectedId();
    if (selected <= 0 || static_cast<std::size_t>(selected) > catalog_.size())
        return;
    const auto& entry = catalog_[static_cast<std::size_t>(selected - 1)];
    juce::String text;
    const auto add = [&](const char* label, std::string_view value)
    {
        text += juce::String(label) + juce::String(std::string(value)) + "\n";
    };
    add("Input: ", entry.inputRepresentation);
    add("Output: ", entry.outputRepresentation);
    add("Persistent state: ", entry.stateRepresentation);
    add("Cost per update: ", entry.costPerUpdate);
    add("Rates: ", entry.supportedRates);
    add("Reuse: ", entry.reusableCount);
    text += "Parameters in local order (repeated families follow the described indexing):\n";
    for (const auto& parameter : entry.parameters)
        text += juce::String(std::string(parameter.name)) + " [" +
                juce::String(std::string(parameter.units)) + "] / " +
                juce::String(std::string(parameter.validRange)) + "\n";
    text += "Listed parameter count is a minimum; repeated layouts require all described groups.\n"
            "WAV imports preserve the chosen channel and source rate; the reader decodes float32 PCM before "
            "typed asset storage.";
    metadata_.setText(text, false);
}
void PrimitiveFragmentPanel::create()
{
    try
    {
        const auto selected = operation_.getSelectedId();
        if (selected <= 0 || static_cast<std::size_t>(selected) > catalog_.size())
            throw std::invalid_argument("Select an operation explicitly");
        if (between_.getSelectedId() == 0)
            throw std::invalid_argument("Select the output policy between updates explicitly");
        const auto budget = JSON::parse(budget_.getText().toStdString());
        if (!budget.is_object())
            throw std::invalid_argument("An explicit capacity budget object is required");
        const auto states = index(state_);
        if (!budget.at("state_elements").is_number_unsigned() &&
            (!budget.at("state_elements").is_number_integer() ||
             budget.at("state_elements").get<std::int64_t>() < 0))
            throw std::invalid_argument("State capacity must be a nonnegative integer");
        if (states > budget.at("state_elements").get<std::size_t>())
            throw std::invalid_argument(
                "Requested initial state exceeds explicit state capacity before allocation");
        auto initial = array(initial_);
        if (initial_.getText().trim().isEmpty())
            initial = std::vector<double>(states, 0.);
        const auto parameters = array(parameters_, false);
        const auto& selectedMetadata = catalog_[static_cast<std::size_t>(selected - 1)];
        if (parameters.size() < selectedMetadata.parameters.size())
            throw std::invalid_argument(
                "Parameter array is smaller than the selected operation's documented minimum");
        if (selectedMetadata.id == "time.fdn_hadamard8.v1" && parameters.size() != 24)
            throw std::invalid_argument(
                "Eight-line FDN requires the documented eight parameter triplets (24 values)");
        const auto routing = array(routing_);
        const auto modulation = array(modulation_);
        const auto assets = array(assets_);
        const auto bindings = array(bindings_);
        const auto channels = index(channels_);
        JSON sound{
            {"schema", "aom.tensor/1"},
            {"ontology", "aom.math/1"},
            {"tier", tier_.toStdString()},
            {"tensors",
             {{"activation", tensor(JSON::array({1}), "primitive")},
              {"parameters", tensor(parameters, "parameter")},
              {"routing", tensor(routing, "coefficient")},
              {"modulation", tensor(modulation, "relationship")}}},
            {"instances",
             JSON::array({{{"primitive", std::string(catalog_[static_cast<std::size_t>(selected - 1)].id)},
                           {"activation", 0},
                           {"parameters", {{"offset", 0}, {"count", parameters.size()}}},
                           {"state_size", states},
                           {"initial_state", initial},
                           {"assets", assets},
                           {"input_channels", channels},
                           {"audio_bindings", bindings},
                           {"clock", {{"period_samples", index(period_)}, {"phase_samples", index(phase_)}}},
                           {"output_between_updates", between_.getText().toStdString()}}})},
            {"projections", JSON::array()},
            {"shared_initial_state", JSON::array()},
            {"parameter_locks", JSON::array()},
            {"outputMix", array(outputMix_)},
            {"budget", budget},
            {"asset_sources", JSON::parse(assetSources_.toStdString())}};
        alfa::tensor::PrecisionProfile precision;
        std::string diagnostic;
        if (!alfa::tensor::resolvePrecisionJSON(sound.dump(), precision, diagnostic))
            throw std::invalid_argument(diagnostic);
        sound["scalar_type"] = alfa::tensor::scalarTypeName(precision.scalarType);
        const auto text = sound.dump(2);
        if (!prepareExecution(text, sampleRate_, diagnostic))
            throw std::invalid_argument(diagnostic);
        juce::String error;
        if (!publish_(text, error))
            throw std::invalid_argument(error.isEmpty() ? "Fragment publication was rejected"
                                                        : error.toStdString());
        status_.setColour(juce::Label::textColourId, look_.theme().text2);
        status_.setText("Validated fragment created with its authored state, clock and bindings",
                        juce::dontSendNotification);
    }
    catch (const std::exception& failure)
    {
        status_.setColour(juce::Label::textColourId, look_.theme().warn);
        status_.setText(failure.what(), juce::dontSendNotification);
    }
}
bool PrimitiveFragmentPanel::loadWavAsset(const juce::File& file, std::size_t channel, juce::String& error)
{
    try
    {
        if (!file.hasFileExtension("wav") || !file.existsAsFile())
            throw std::invalid_argument("Choose an existing WAV asset");
        const auto budget = JSON::parse(budget_.getText().toStdString());
        const auto& limit = budget.at("stored_tensor_elements");
        if (!limit.is_number_unsigned() && (!limit.is_number_integer() || limit.get<std::int64_t>() < 0))
            throw std::invalid_argument(
                "Asset capacity requires a nonnegative stored_tensor_elements budget");
        const auto capacity = limit.get<std::size_t>();
        auto assets = array(assets_);
        std::size_t stored = 1;
        const auto add = [&](std::size_t count)
        {
            if (stored > capacity || count > capacity - stored)
                throw std::invalid_argument(
                    "Assets exceed explicit stored tensor capacity before WAV decoding");
            stored += count;
        };
        add(array(parameters_, false).size());
        add(array(routing_).size());
        add(array(modulation_).size());
        add(index(channels_));
        for (const auto& asset : assets)
            add(asset.at("values").size());
        juce::AudioFormatManager formats;
        formats.registerFormat(new juce::WavAudioFormat, true);
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
        if (reader == nullptr || reader->lengthInSamples <= 0 ||
            reader->lengthInSamples > std::numeric_limits<int>::max() || channel >= reader->numChannels ||
            reader->numChannels > static_cast<unsigned>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Invalid WAV channel or unsupported sample count");
        const auto samples = static_cast<std::size_t>(reader->lengthInSamples);
        add(samples);
        std::vector<float> values(samples);
        std::vector<float*> destination(reader->numChannels, nullptr);
        destination[channel] = values.data();
        if (!reader->read(destination.data(), static_cast<int>(destination.size()), 0,
                          static_cast<int>(samples)))
            throw std::invalid_argument("WAV channel decoding failed");
        for (const auto value : values)
            if (!std::isfinite(value))
                throw std::invalid_argument("WAV asset contains nonfinite samples");
        const auto assetIndex = assets.size();
        assets.push_back(tensor(values, "sample"));
        auto sources = JSON::parse(assetSources_.toStdString());
        sources.push_back({{"asset_index", assetIndex},
                           {"path", file.getFullPathName().toStdString()},
                           {"sample_rate", reader->sampleRate},
                           {"source_channels", reader->numChannels},
                           {"source_channel", channel},
                           {"source_bits", reader->bitsPerSample},
                           {"decoded_scalar_type", "float32"},
                           {"frames", samples}});
        assets_.setText(assets.dump(2), false);
        assetSources_ = sources.dump();
        error.clear();
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
void PrimitiveFragmentPanel::paint(juce::Graphics& graphics)
{
    graphics.fillAll(look_.theme().bg);
    graphics.setColour(look_.theme().text2);
    graphics.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
    const std::array<const char*, 4> numeric{"State elements", "Audio input channels",
                                             "Clock period (samples)", "Clock phase (samples)"};
    for (std::size_t column = 0; column < numeric.size(); ++column)
        graphics.drawText(numeric[column], 20 + static_cast<int>(column) * 235, 94, 225, 20,
                          juce::Justification::centredLeft);
    const std::array<const char*, 8> labels{"Parameter values JSON",
                                            "Initial state JSON (blank = zeros)",
                                            "Asset tensors JSON (blank = [])",
                                            "Audio bindings JSON (blank = [])",
                                            "Routing weights JSON",
                                            "Modulation weights JSON",
                                            "Output mix JSON (blank = no direct output)",
                                            "Explicit capacity budget JSON"};
    for (std::size_t field = 0; field < labels.size(); ++field)
        graphics.drawText(labels[field], 20 + static_cast<int>(field % 2) * 475,
                          155 + static_cast<int>(field / 2) * 76, 460, 20, juce::Justification::centredLeft);
}
void PrimitiveFragmentPanel::resized()
{
    heading_.setBounds(20, 8, getWidth() - 40, 36);
    operation_.setBounds(20, 48, 535, 32);
    between_.setBounds(570, 48, getWidth() - 590, 32);
    const std::array<juce::Component*, 4> numeric{&state_, &channels_, &period_, &phase_};
    for (std::size_t column = 0; column < numeric.size(); ++column)
        numeric[column]->setBounds(20 + static_cast<int>(column) * 235, 116, 225, 30);
    const std::array<juce::Component*, 8> json{&parameters_, &initial_,    &assets_,    &bindings_,
                                               &routing_,    &modulation_, &outputMix_, &budget_};
    for (std::size_t field = 0; field < json.size(); ++field)
        json[field]->setBounds(20 + static_cast<int>(field % 2) * 475, 178 + static_cast<int>(field / 2) * 76,
                               460, 48);
    wavButton_.setBounds(20, 482, 185, 30);
    wavChannel_.setBounds(220, 482, 185, 30);
    metadata_.setBounds(20, 520, getWidth() - 40, getHeight() - 595);
    publishButton_.setBounds(20, getHeight() - 62, 210, 32);
    status_.setBounds(245, getHeight() - 62, getWidth() - 265, 42);
}
} // namespace alfa::tensorhost
