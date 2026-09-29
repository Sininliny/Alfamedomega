#include "alfa/dsp/Synth.h"

#include "alfa/dsp/Voice.h"
#include "alfa/dsp/detail/Denormals.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace alfa::dsp
{

Synth::Synth() = default;
Synth::~Synth() = default;

void Synth::prepare(double sampleRate, int maxBlock, Tier tier)
{
    spec_.sampleRate = sampleRate;
    spec_.hop = hopSamples(sampleRate);
    spec_.limits = limitsFor(tier);
    maxBlock_ = maxBlock;

    voices_.clear();
    voices_.reserve(static_cast<std::size_t>(spec_.limits.polyphony));
    for (int v = 0; v < spec_.limits.polyphony; ++v)
    {
        auto voice = std::make_unique<Voice>();
        voice->prepare(spec_, 0x1234567u + 0x9E3779B9u * static_cast<std::uint32_t>(v + 1));
        voices_.push_back(std::move(voice));
    }
    startOrder_.assign(voices_.size(), 0);
    counter_ = 0;
}

void Synth::setGrainBank(const GrainBank* bank) noexcept
{
    for (auto& v : voices_)
        v->setGrainBank(bank);
}

void Synth::setSample(const SampleData* sample) noexcept
{
    for (auto& v : voices_)
        v->setSample(sample);
}

float Synth::f0For(int note) const noexcept
{
    return static_cast<float>(440.0 * std::pow(2.0, (note - 69 + static_cast<double>(bendSemitones_)) / 12.0));
}

int Synth::noteOn(int note, float velocity, const VoiceFrame* firstFrame) noexcept
{
    if (voices_.empty())
        return -1;
    int chosen = -1;
    // Free voice first, then the oldest released voice, then the oldest voice.
    for (std::size_t i = 0; i < voices_.size() && chosen < 0; ++i)
        if (!voices_[i]->active())
            chosen = static_cast<int>(i);
    for (int pass = 0; pass < 2 && chosen < 0; ++pass)
    {
        std::uint64_t oldest = UINT64_MAX;
        for (std::size_t i = 0; i < voices_.size(); ++i)
        {
            if (pass == 0 && !voices_[i]->releasing())
                continue;
            if (startOrder_[i] < oldest)
            {
                oldest = startOrder_[i];
                chosen = static_cast<int>(i);
            }
        }
    }
    const auto c = static_cast<std::size_t>(chosen);
    voices_[c]->start(note, velocity, f0For(note), firstFrame);
    startOrder_[c] = ++counter_;
    return chosen;
}

void Synth::noteOff(int note) noexcept
{
    for (auto& v : voices_)
        if (v->active() && !v->releasing() && v->note() == note)
            v->release();
}

void Synth::allNotesOff() noexcept
{
    for (auto& v : voices_)
        if (v->active())
            v->release();
}

void Synth::killAll() noexcept
{
    for (auto& v : voices_)
        v->kill();
}

void Synth::setPitchBend(float semitones) noexcept
{
    bendSemitones_ = semitones;
    for (auto& v : voices_)
        if (v->active())
            v->setF0(f0For(v->note()));
}

void Synth::setFrame(int voice, const VoiceFrame& frame) noexcept
{
    if (voice >= 0 && voice < voiceCount())
        voices_[static_cast<std::size_t>(voice)]->setFrame(frame);
}

void Synth::process(float* left, float* right, int n) noexcept
{
    detail::ScopedNoDenormals noDenormals;
    std::fill_n(left, n, 0.0f);
    for (auto& v : voices_)
        v->process(left, n);
    if (right != nullptr)
        std::memcpy(right, left, static_cast<std::size_t>(n) * sizeof(float));
}

VoiceInfo Synth::voiceInfo(int voice) const noexcept
{
    VoiceInfo info;
    if (voice < 0 || voice >= voiceCount())
        return info;
    const auto& v = *voices_[static_cast<std::size_t>(voice)];
    info.active = v.active();
    info.releasing = v.releasing();
    info.note = v.note();
    info.velocity = v.velocity();
    info.samplesSinceNoteOn = v.samplesSinceNoteOn();
    info.samplesSinceRelease = v.samplesSinceRelease();
    return info;
}

} // namespace alfa::dsp
