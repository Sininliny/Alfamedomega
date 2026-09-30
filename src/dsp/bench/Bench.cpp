// CPU cost of the DSP per tier, as a share of the audio block at 48 kHz.
// Build Release: cmake --build <dir> --target alfa_dsp_bench, then run <dir>/src/dsp/alfa_dsp_bench.
// Every voice of the tier's polyphony plays; frames are rewritten every hop as the network would.

#include "alfa/dsp/Convolution.h"
#include "alfa/dsp/Family.h"
#include "alfa/dsp/Synth.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace alfa::dsp;

namespace
{
struct Scenario
{
    const char* name;
    void (*fill)(VoiceFrame&);
};

void oscNoise(VoiceFrame& f) // subtractive, wavetable, additive, spectral
{
    for (int k = 0; k < kOscHarmonics; ++k)
        f.osc.re[static_cast<std::size_t>(k)] = 1.0f / static_cast<float>(k + 1);
    f.noise.band.fill(0.05f);
    f.setMix(Primitive::Osc, 0.5f);
    f.setMix(Primitive::Noise, 0.2f);
}

void fm(VoiceFrame& f)
{
    for (int i = 0; i < kMaxFmOperators; ++i)
    {
        f.fm.ratio[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        f.fm.level[static_cast<std::size_t>(i)] = 0.1f;
        for (int j = i + 1; j < kMaxFmOperators; ++j)
            f.fm.index[static_cast<std::size_t>(i * kMaxFmOperators + j)] = 0.7f;
    }
    f.setMix(Primitive::Fm, 0.5f);
}

void modal(VoiceFrame& f) // physical modelling
{
    for (int m = 0; m < kMaxPartials; ++m)
    {
        f.partials.ratio[static_cast<std::size_t>(m)] = 1.0f + 1.13f * static_cast<float>(m);
        f.partials.amp[static_cast<std::size_t>(m)] = 0.02f;
    }
    f.noise.band.fill(0.02f);
    f.setMix(Primitive::Partials, 0.5f);
    f.setMix(Primitive::Noise, 0.1f);
}

void grain(VoiceFrame& f)
{
    for (int l = 0; l < kMaxGrainLayers; ++l)
    {
        const auto i = static_cast<std::size_t>(l);
        f.grain.source[2 * i] = static_cast<float>(l % 4);
        f.grain.source[2 * i + 1] = static_cast<float>((l + 1) % 4);
        f.grain.weight[2 * i] = 0.6f;
        f.grain.weight[2 * i + 1] = 0.4f;
        f.grain.position[i] = 0.1f * static_cast<float>(l);
        f.grain.rate[i] = 1.0f + 0.1f * static_cast<float>(l);
        f.grain.gain[i] = 0.1f;
    }
    f.setMix(Primitive::Grain, 0.5f);
}

void sampler(VoiceFrame& f) // sampling: sampler with EQ plus a small OSC residual
{
    f.sampler.rate = 1.0f;
    f.sampler.gain = 1.0f;
    f.sampler.eq.fill(0.9f);
    for (int k = 0; k < 64; ++k)
        f.osc.re[static_cast<std::size_t>(k)] = 0.01f / static_cast<float>(k + 1);
    f.setMix(Primitive::Sampler, 0.5f);
    f.setMix(Primitive::Osc, 0.1f);
}

void everything(VoiceFrame& f) // hybrid worst case
{
    oscNoise(f);
    fm(f);
    modal(f);
    grain(f);
    sampler(f);
}

/// Slot setup for the family benchmarks: `family` in every voice slot (voice families) or every
/// instance slot (instance families), at its default params, fed by the OSC primitive or the bus.
struct SlotSetup
{
    Family family = Family::Off;
    bool instance = false;
    bool guard = false;            ///< the audio budget guard (off: raw cost)
    GuardReport* report = nullptr; ///< receives the guard's last report
};

void fillSlot(SlotControls& slot, Family family, int inputIndex)
{
    slot.family = static_cast<float>(family);
    slot.outGain = 0.5f;
    slot.input[static_cast<std::size_t>(inputIndex)] = 1.0f;
    const auto& table = paramTable(family);
    for (int i = 0; i < kSlotParams; ++i)
        slot.params[static_cast<std::size_t>(i)] = table[static_cast<std::size_t>(i)].defaultNorm;
}

double measure(Tier tier, const Scenario& s, double sampleRate, int block, SlotSetup slots = {})
{
    Synth synth;
    synth.prepare(sampleRate, block, tier);
    synth.setBudgetGuardEnabled(slots.guard);

    std::vector<float> bankData(4 * kGrainSourceLength);
    for (std::size_t i = 0; i < bankData.size(); ++i)
        bankData[i] = std::sin(0.01f * static_cast<float>(i));
    GrainBank bank{bankData.data(), 4, kGrainSourceLength};
    std::vector<float> sampleData(static_cast<std::size_t>(sampleRate * 4));
    for (std::size_t i = 0; i < sampleData.size(); ++i)
        sampleData[i] = std::sin(0.05f * static_cast<float>(i));
    SampleData sample{sampleData.data(),
                      static_cast<long long>(sampleData.size()),
                      sampleRate,
                      261.63,
                      1000,
                      static_cast<long long>(sampleData.size()) - 1000};
    synth.setGrainBank(&bank);
    synth.setSample(&sample);

    auto frame = std::make_unique<VoiceFrame>();
    frame->attackTau = 0.002f;
    s.fill(*frame);
    IrBank irs;
    if (slots.family != Family::Off)
    {
        if (slots.instance)
        {
            InstanceFrame inst;
            for (auto& slot : inst.slots)
                fillSlot(slot, slots.family, 0);
            synth.setInstanceFrame(inst);
            irs.setHostRate(sampleRate);
            SyntheticIrSpec ir;
            ir.sampleRate = sampleRate;
            ir.rt60 = 2.0;
            (void)irs.add(makeSyntheticIr(ir));
            synth.publishIrs(irs);
        }
        else
        {
            for (auto& slot : frame->slots)
                fillSlot(slot, slots.family, static_cast<int>(Primitive::Osc));
        }
    }
    for (int v = 0; v < synth.voiceCount(); ++v)
        synth.noteOn(40 + 2 * v, 0.8f, frame.get());

    std::vector<float> left(static_cast<std::size_t>(block)), right(static_cast<std::size_t>(block));
    const int blocks = static_cast<int>(3.0 * sampleRate / block); // 3 s of audio
    const int hop = synth.hop();
    int sinceFrame = 0;
    for (int i = 0; i < 50; ++i)
        synth.process(left.data(), right.data(), block); // warm up

    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < blocks; ++b)
    {
        sinceFrame += block;
        if (sinceFrame >= hop)
        {
            sinceFrame -= hop;
            for (int v = 0; v < synth.voiceCount(); ++v)
                synth.setFrame(v, *frame);
        }
        synth.process(left.data(), right.data(), block);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / blocks;
    if (slots.report != nullptr)
        *slots.report = synth.guardReport();
    return us;
}
} // namespace

int main()
{
    const Scenario scenarios[] = {
        {"osc+noise", oscNoise}, {"fm", fm},           {"modal", modal},
        {"grain", grain},        {"sampler", sampler}, {"everything", everything},
    };
    const char* tierNames[] = {"Lite", "Standard", "Pro", "Studio"};
    const int block = 128;
    for (const double sr : {48000.0, 96000.0})
    {
        const double blockUs = block / sr * 1e6;
        std::printf("\n%.0f Hz, %d-sample blocks (%.0f us), all voices of the tier playing: us per block (%% "
                    "of block)\n",
                    sr, block, blockUs);
        std::printf("%-12s", "scenario");
        for (const char* t : tierNames)
            std::printf("%18s", t);
        std::printf("\n");
        for (const auto& s : scenarios)
        {
            std::printf("%-12s", s.name);
            for (int t = 0; t < 4; ++t)
            {
                const double us = measure(static_cast<Tier>(t), s, sr, block);
                std::printf("%10.1f (%4.1f%%)", us, 100.0 * us / blockUs);
            }
            std::printf("\n");
        }
    }

    // Per family: osc+noise voices plus that family in every slot the tier allows. The cost of
    // the family alone is the row minus the osc+noise row above.
    const double sr = 48000.0;
    const double blockUs = block / sr * 1e6;
    std::printf("\nPer family at 48 kHz: osc+noise voices + family in every allowed slot (voice families in "
                "each voice's slots, others in the instance slots); us per block (%% of block)\n");
    std::printf("%-12s", "family");
    for (const char* t : tierNames)
        std::printf("%18s", t);
    std::printf("\n");
    for (int f = 1; f < kFamilyCount; ++f)
    {
        const auto family = static_cast<Family>(f);
        const auto name = familyName(family);
        std::printf("%-12.*s", static_cast<int>(name.size()), name.data());
        for (int t = 0; t < 4; ++t)
        {
            const double us =
                measure(static_cast<Tier>(t), scenarios[0], sr, block, {family, !allowedInVoice(family)});
            std::printf("%10.1f (%4.1f%%)", us, 100.0 * us / blockUs);
        }
        std::printf("\n");
    }

    // Worst case: every primitive plus NONLINEAR (the heaviest voice family) in every voice slot.
    std::printf("\nWorst case (everything + nonlinear in every voice slot), guard off vs on: us per block "
                "(%% of block), guard model estimate -> after, slots downgraded/bypassed\n");
    for (const double rate : {48000.0, 96000.0})
    {
        const double us0 = block / rate * 1e6;
        for (int t = 0; t < 4; ++t)
        {
            GuardReport rep;
            const double off =
                measure(static_cast<Tier>(t), scenarios[5], rate, block, {Family::Nonlinear, false});
            const double on = measure(static_cast<Tier>(t), scenarios[5], rate, block,
                                      {Family::Nonlinear, false, true, &rep});
            std::printf(
                "%3.0f kHz %-9s off %7.1f (%4.1f%%)  on %7.1f (%4.1f%%)  model %4.1f%% -> %4.1f%%  %d/%d\n",
                rate / 1000.0, tierNames[t], off, 100.0 * off / us0, on, 100.0 * on / us0,
                100.0 * rep.estimatedShare, 100.0 * rep.finalShare, rep.downgraded, rep.bypassed);
        }
    }
    return 0;
}
