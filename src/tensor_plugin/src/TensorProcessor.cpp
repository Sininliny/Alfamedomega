#include "TensorProcessor.h"

#include "TensorEditor.h"
#include "TensorReference.h"
#include "alfa/tensor/Compiler.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace alfa::tensorhost
{
namespace
{
alfa::tensor::Tensor immutableInput(const nlohmann::json& data)
{
    alfa::tensor::Tensor tensor;
    tensor.axes = data.at("axes").get<std::vector<std::string>>();
    if (!data.at("shape").is_array())
        throw std::invalid_argument("latentSound shape must be an array");
    size_t elements = 1;
    for (const auto& value : data.at("shape"))
    {
        if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0))
            throw std::invalid_argument("latentSound shape must use nonnegative integers");
        const auto dimension = value.get<uint64_t>();
        if (dimension > std::numeric_limits<size_t>::max() ||
            (dimension != 0 && elements > std::numeric_limits<size_t>::max() / dimension))
            throw std::invalid_argument("latentSound shape overflows capacity");
        elements *= static_cast<size_t>(dimension);
        tensor.shape.push_back(static_cast<size_t>(dimension));
    }
    tensor.values = data.at("values").get<std::vector<float>>();
    if (tensor.axes.empty() && tensor.shape.empty() && tensor.values.empty())
        elements = 0;
    if (tensor.axes.size() != tensor.shape.size() || tensor.values.size() != elements)
        throw std::invalid_argument("latentSound shape and values differ");
    for (size_t axis = 0; axis < tensor.axes.size(); ++axis)
    {
        if (tensor.axes[axis].empty())
            throw std::invalid_argument("latentSound axes must be named");
        for (size_t previous = 0; previous < axis; ++previous)
            if (tensor.axes[axis] == tensor.axes[previous])
                throw std::invalid_argument("latentSound axes must be unique");
    }
    for (const auto value : tensor.values)
        if (!std::isfinite(value))
            throw std::invalid_argument("latentSound values must be finite");
    if (data.contains("presence"))
    {
        const auto& mask = data.at("presence");
        if (!mask.is_array() || (!mask.empty() && mask.size() != elements))
            throw std::invalid_argument("latentSound presence must match its values");
        for (size_t i = 0; i < mask.size(); ++i)
        {
            if (!mask[i].is_boolean() &&
                (!mask[i].is_number_integer() || mask[i].get<int64_t>() < 0 || mask[i].get<int64_t>() > 1))
                throw std::invalid_argument("latentSound presence must be binary");
            const auto bit = mask[i].is_boolean() ? (mask[i].get<bool>() ? 1 : 0) : mask[i].get<int>();
            if (bit == 0 && tensor.values[i] != 0)
                throw std::invalid_argument("latentSound padding must be zero");
            tensor.presence.push_back(static_cast<uint8_t>(bit));
        }
    }
    return tensor;
}
} // namespace

Processor::Processor()
    : AudioProcessor(BusesProperties()
                         .withInput("Audio input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Audio output", juce::AudioChannelSet::stereo(), true))
{
    keyboard.addListener(this);
    juce::String error;
    loadJSON(referenceDocument(0), error);
    startTimerHz(4);
}
juce::String Processor::referenceDocument(int index)
{
    const std::array<juce::String, 3> names{"instrument.aom.json", "fx.aom.json", "hybrid.aom.json"};
    const auto selected = names[static_cast<size_t>(juce::jlimit(0, 2, index))];
    for (int i = 0; i < TensorReference::namedResourceListSize; ++i)
    {
        const auto* resource = TensorReference::namedResourceList[i];
        const auto* filename = TensorReference::getNamedResourceOriginalFilename(resource);
        if (filename && juce::String(filename) == selected)
        {
            int length = 0;
            auto* data = TensorReference::getNamedResource(resource, length);
            return juce::String::fromUTF8(data, length);
        }
    }
    return {};
}
Processor::~Processor()
{
    stopTimer();
    keyboard.removeListener(this);
}
bool Processor::isBusesLayoutSupported(const BusesLayout& b) const
{
    const auto out = b.getMainOutputChannelSet();
    const auto in = b.getMainInputChannelSet();
    return (out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo()) &&
           (in.isDisabled() || in == out);
}
void Processor::prepareToPlay(double rate, int)
{
    mergedMidi_.ensureSize(65536);
    rate_ = rate;
    clock_ = 0;
    notes_.fill(0);
    noteCount_ = 0;
    gestures_.fill(0);
    context_.fill(0);
    const auto saved = document();
    if (saved.isNotEmpty())
    {
        juce::String error;
        loadJSON(saved, error);
    }
}
void Processor::queueKey(int channel, int note, float velocity, bool on) noexcept
{
    const auto write = keyWrite_.load(std::memory_order_relaxed);
    const auto next = (write + 1) % keyEvents_.size();
    if (next == keyRead_.load(std::memory_order_acquire))
    {
        failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    keyEvents_[write] = {channel, note, velocity, on};
    keyWrite_.store(next, std::memory_order_release);
}
void Processor::handleNoteOn(juce::MidiKeyboardState*, int channel, int note, float velocity)
{
    queueKey(channel, note, velocity, true);
}
void Processor::handleNoteOff(juce::MidiKeyboardState*, int channel, int note, float velocity)
{
    queueKey(channel, note, velocity, false);
}
void Processor::event(const juce::MidiMessage& m) noexcept
{
    const auto ch = static_cast<size_t>(juce::jlimit(1, 16, m.getChannel()) - 1);
    if (m.isNoteOnOrOff())
    {
        size_t row = 0;
        while (row < noteCount_ && (notes_[row * 10 + 8] != static_cast<float>(ch + 1) ||
                                    notes_[row * 10 + 9] != static_cast<float>(m.getNoteNumber())))
            ++row;
        if (m.isNoteOn())
        {
            if (row == noteCount_)
            {
                if (noteCount_ == 128)
                {
                    failures.fetch_add(1);
                    return;
                }
                ++noteCount_;
            }
            auto* n = notes_.data() + row * 10;
            n[0] = static_cast<float>(juce::MidiMessage::getMidiNoteInHertz(m.getNoteNumber()));
            n[1] = m.getFloatVelocity();
            n[2] = 1;
            n[3] = 0;
            n[4] = gestures_[ch * 3 + 1];
            n[5] = gestures_[ch * 3];
            n[6] = gestures_[ch * 3 + 2];
            n[7] = static_cast<float>(static_cast<double>(clock_) / rate_);
            n[8] = static_cast<float>(ch + 1);
            n[9] = static_cast<float>(m.getNoteNumber());
            context_[4] = n[7] - context_[5];
            context_[5] = n[7];
            context_[2] = n[9];
            context_[3] += 1;
        }
        else
        {
            context_[6] = static_cast<float>(m.getNoteNumber());
            context_[7] = static_cast<float>(static_cast<double>(clock_) / rate_);
            if (row < noteCount_)
            {
                --noteCount_;
                std::copy_n(notes_.data() + noteCount_ * 10, 10, notes_.data() + row * 10);
            }
        }
    }
    else if (m.isAftertouch())
    {
        for (size_t row = 0; row < noteCount_; ++row)
            if (notes_[row * 10 + 8] == static_cast<float>(ch + 1) &&
                notes_[row * 10 + 9] == static_cast<float>(m.getNoteNumber()))
                notes_[row * 10 + 4] = m.getAfterTouchValue() / 127.0f;
    }
    else if (m.isPitchWheel() || m.isChannelPressure() || (m.isController() && m.getControllerNumber() == 74))
    {
        const size_t field = m.isPitchWheel() ? 0 : m.isChannelPressure() ? 1 : 2;
        const auto value = m.isPitchWheel()        ? (m.getPitchWheelValue() - 8192) / 8192.0f
                           : m.isChannelPressure() ? m.getChannelPressureValue() / 127.0f
                                                   : m.getControllerValue() / 127.0f;
        gestures_[ch * 3 + field] = value;
        for (size_t row = 0; row < noteCount_; ++row)
            if (notes_[row * 10 + 8] == static_cast<float>(ch + 1))
                notes_[row * 10 + (field == 0 ? 5 : field == 1 ? 4 : 6)] = value;
    }
    else if (m.isAllNotesOff() || m.isAllSoundOff())
    {
        for (size_t row = noteCount_; row-- > 0;)
            if (notes_[row * 10 + 8] == static_cast<float>(ch + 1))
            {
                --noteCount_;
                std::copy_n(notes_.data() + noteCount_ * 10, 10, notes_.data() + row * 10);
            }
    }
}
void Processor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals denormals;
    const auto began = std::chrono::steady_clock::now();
    double modelTime = 0;
    const auto write = retireWrite_.load(std::memory_order_relaxed);
    const auto nextWrite = (write + 1) % retired_.size();
    if (nextWrite != retireRead_.load(std::memory_order_acquire))
    {
        if (auto* program = pending_.exchange(nullptr, std::memory_order_acq_rel))
        {
            if (active_)
            {
                retired_[write] = active_;
                retireWrite_.store(nextWrite, std::memory_order_release);
            }
            active_ = program;
        }
    }
    mergedMidi_.clear();
    if (midi.data.size() > 60000)
    {
        buffer.clear();
        failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    mergedMidi_.addEvents(midi, 0, buffer.getNumSamples(), 0);
    auto keyRead = keyRead_.load(std::memory_order_relaxed);
    while (keyRead != keyWrite_.load(std::memory_order_acquire))
    {
        const auto& key = keyEvents_[keyRead];
        mergedMidi_.addEvent(key.on ? juce::MidiMessage::noteOn(key.channel, key.note, key.velocity)
                                    : juce::MidiMessage::noteOff(key.channel, key.note),
                             0);
        keyRead = (keyRead + 1) % keyEvents_.size();
        keyRead_.store(keyRead, std::memory_order_release);
    }
    auto events = mergedMidi_.begin();
    for (size_t i = 0; i < preferences_.size(); ++i)
        preferences_[i] = intentions[i].load();
    const auto selectedMode = mode.load();
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
    {
        while (events != mergedMidi_.end() && (*events).samplePosition <= sample)
        {
            if ((*events).numBytes <= 3)
                event((*events).getMessage());
            ++events;
        }
        std::array<float, 2> audio{};
        for (int channel = 0; channel < juce::jmin(2, getTotalNumInputChannels()); ++channel)
            audio[static_cast<size_t>(channel)] = buffer.getSample(channel, sample);
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
            buffer.setSample(channel, sample, 0);
        alfa::tensor::Inputs input;
        if (selectedMode != 0)
            input.audio = {audioAxes_, stereoShape_, audio, {}};
        if (selectedMode != 1)
        {
            midiShape_[0] = noteCount_;
            input.midi = {midiAxes_, midiShape_, std::span<const float>(notes_.data(), noteCount_ * 10), {}};
            input.gesture = {gestureAxes_, gestureShape_, gestures_, {}};
        }
        context_[0] = static_cast<float>(static_cast<double>(clock_) / rate_);
        context_[1] = static_cast<float>(rate_);
        input.context = {contextAxes_, contextShape_, context_, {}};
        input.userPreference = {preferenceAxes_, preferenceShape_, preferences_, {}};
        if (active_)
        {
            const auto& latent = active_->latentSound;
            input.latentSound = {latent.axes, latent.shape, latent.values, latent.presence};
            if (active_->model && active_->runtime.sample() % active_->modelClock.periodSamples ==
                                      active_->modelClock.phaseSamples)
            {
                const auto modelBegan = std::chrono::steady_clock::now();
                if (!active_->model->apply(input, active_->runtime))
                    failures.fetch_add(1, std::memory_order_relaxed);
                modelTime +=
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - modelBegan)
                        .count();
            }
            if (active_->runtime.tick(input))
            {
                const auto output = active_->runtime.output();
                for (const auto& mix : active_->mix)
                    if (mix.channel < buffer.getNumChannels())
                        buffer.addSample(mix.channel, sample, mix.gain * output[mix.output]);
            }
            else
                failures.fetch_add(1, std::memory_order_relaxed);
        }
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
            if (!std::isfinite(buffer.getSample(channel, sample)))
            {
                buffer.setSample(channel, sample, 0);
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        for (size_t n = 0; n < noteCount_; ++n)
            notes_[n * 10 + 3] += static_cast<float>(1.0 / rate_);
        ++clock_;
    }
    samples.store(clock_, std::memory_order_relaxed);
    if (active_)
    {
        const auto state = active_->runtime.state();
        for (size_t i = 0; i < stateSnapshot.size(); ++i)
            stateSnapshot[i].store(i < state.size() ? state[i] : 0, std::memory_order_relaxed);
    }
    modelMilliseconds.store(modelTime, std::memory_order_relaxed);
    blockMilliseconds.store(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count(),
        std::memory_order_relaxed);
}
bool Processor::loadJSON(const juce::String& text, juce::String& error)
{
    std::lock_guard lock(configurationMutex_);
    try
    {
        const auto data = nlohmann::json::parse(text.toStdString());
        alfa::tensor::Configuration config;
        std::string diagnostic;
        if (!alfa::tensor::parseJSON(text.toStdString(), config, diagnostic))
        {
            error = diagnostic;
            return false;
        }
        alfa::tensor::Budget budget;
        if (!alfa::tensor::readBudgetJSON(text.toStdString(), budget, diagnostic))
        {
            error = diagnostic;
            return false;
        }
        auto proposed = std::make_unique<Program>();
        if (!proposed->runtime.prepare(config, budget, rate_, diagnostic))
        {
            error = diagnostic;
            return false;
        }
        if (data.contains("latentSound"))
            proposed->latentSound = immutableInput(data.at("latentSound"));
        if (data.contains("runtimeModel"))
        {
            proposed->model = std::make_unique<alfa::tensor::TensorModel>();
            if (!proposed->model->prepare(data.at("runtimeModel").dump(), config, diagnostic))
            {
                error = diagnostic;
                return false;
            }
            const auto& clock = data.at("runtimeModel").at("clock");
            proposed->modelClock = {clock.at("period_samples").get<uint64_t>(),
                                    clock.at("phase_samples").get<uint64_t>()};
        }
        for (const auto& item : data.at("outputMix"))
        {
            Mix mix{item.at(0).get<size_t>(), item.at(1).get<int>(), item.at(2).get<float>()};
            if (mix.output >= proposed->runtime.output().size() || mix.channel < 0 || mix.channel > 1 ||
                !std::isfinite(mix.gain))
            {
                error = "Invalid explicit outputMix";
                return false;
            }
            proposed->mix.push_back(mix);
        }
        inspection_ =
            "Compiled state elements: " + juce::String(static_cast<int>(proposed->runtime.state().size())) +
            " | output elements: " + juce::String(static_cast<int>(proposed->runtime.output().size())) +
            " | scheduled primitives: " +
            juce::String(static_cast<int>(proposed->runtime.schedule().size())) +
            "\nDeclared budget and routing, activation, allocation and clocks are inspectable in JSON.";
        inspection_ += proposed->model
                           ? "\nRuntime GRU compiler model loaded; inference follows explicit file clock: " +
                                 juce::String(data.at("runtimeModel").at("clock").dump())
                           : "\nNo trained model is loaded.";
        for (size_t i = 0; i < config.instances.size(); ++i)
        {
            const auto& instance = config.instances[i];
            const auto state = proposed->runtime.stateAllocation()[i];
            inspection_ += "\n" + juce::String(instance.primitive->id) +
                           " activation=" + juce::String(config.activation.values[instance.activationIndex]) +
                           " clock=" + juce::String(static_cast<juce::int64>(instance.clock.periodSamples)) +
                           " samples; state [" + juce::String(static_cast<juce::int64>(state.offset)) + "," +
                           juce::String(static_cast<juce::int64>(state.count)) + "]";
        }
        inspection_ += "\nRouting tensor: " + juce::String(data.at("tensors").at("routing").dump()) +
                       "\nProjection terms: " + juce::String(static_cast<int>(config.projections.size())) +
                       "\nParameter locks: " +
                       juce::String(data.value("parameter_locks", nlohmann::json::array()).dump());
        auto* publication = proposed.get();
        owned_.push_back(std::move(proposed));
        auto* superseded = pending_.exchange(publication, std::memory_order_acq_rel);
        if (superseded)
            std::erase_if(owned_, [superseded](const auto& item) { return item.get() == superseded; });
        json_ = text;
        error.clear();
        return true;
    }
    catch (const std::exception& e)
    {
        error = e.what();
        return false;
    }
}
void Processor::getStateInformation(juce::MemoryBlock& dest)
{
    auto wrapper = nlohmann::json{{"configuration", document().toStdString()}, {"mode", mode.load()}};
    std::vector<float> values;
    for (const auto& value : intentions)
        values.push_back(value.load());
    wrapper["intentions"] = values;
    const auto text = wrapper.dump();
    dest.replaceAll(text.data(), text.size());
}
void Processor::setStateInformation(const void* bytes, int length)
{
    try
    {
        auto data =
            nlohmann::json::parse(static_cast<const char*>(bytes), static_cast<const char*>(bytes) + length);
        juce::String error;
        if (!loadJSON(data.at("configuration").get<std::string>(), error))
            return;
        mode.store(juce::jlimit(0, 2, data.value("mode", 2)));
        const auto values = data.value("intentions", std::vector<float>{});
        for (size_t i = 0; i < juce::jmin(values.size(), intentions.size()); ++i)
            if (std::isfinite(values[i]))
                intentions[i].store(values[i]);
    }
    catch (const std::exception&)
    {
    }
}
void Processor::timerCallback()
{
    std::lock_guard lock(configurationMutex_);
    auto read = retireRead_.load(std::memory_order_relaxed);
    while (read != retireWrite_.load(std::memory_order_acquire))
    {
        auto* retired = retired_[read];
        std::erase_if(owned_, [retired](const auto& item) { return item.get() == retired; });
        read = (read + 1) % retired_.size();
        retireRead_.store(read, std::memory_order_release);
    }
}
juce::AudioProcessorEditor* Processor::createEditor()
{
    return new Editor(*this);
}
} // namespace alfa::tensorhost
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new alfa::tensorhost::Processor();
}
