// Measures the audio budget guard's cost table and prints CostTableGenerated.h.
// Build Release: cmake --build <dir> --target alfa_dsp_calibrate
// Run on an otherwise idle machine:
//   <dir>/src/dsp/alfa_dsp_calibrate > src/dsp/include/alfa/dsp/CostTableGenerated.h
//
// Every figure is ns of CPU per sample of audio at 48 and 96 kHz, the median of several runs:
// - primitives: a full tier's voices each playing one primitive at the tier's limits, minus the
//   empty synth, per voice;
// - slots: the processor alone at typical params, the median of several param sets (defaults, all 0.5, all 1,
//   random), params nudged every control block so coefficient updates are included; mono for
//   voice slots, stereo for instance slots; quality 0 and 1;
// - instance base: the synth with no notes (bus, instance chain with every slot off);
// - rateScale: per rate, sum(measured) / sum(modelled) over whole-synth patches (each tier at
//   full polyphony: each primitive alone, all of them, all with NONLINEAR in every voice slot,
//   OSC with FILTER in every voice slot, OSC with REVERB in every instance slot), so the model's
//   total matches reality; the guard's 1.2 safety margin applies on top.

#include "alfa/dsp/BudgetGuard.h"
#include "alfa/dsp/Convolution.h"
#include "alfa/dsp/Family.h"
#include "alfa/dsp/Synth.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using namespace alfa::dsp;

namespace
{

constexpr int kBlock = 128;
constexpr double kRates[2] = {48000.0, 96000.0};
constexpr int kRuns = 5;

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void fillPrimitive(VoiceFrame& f, Primitive p, const TierLimits& limits)
{
    switch (p)
    {
    case Primitive::Osc:
        for (int k = 0; k < kOscHarmonics; ++k)
            f.osc.re[static_cast<std::size_t>(k)] = 1.0f / static_cast<float>(k + 1);
        for (int r = 0; r + 1 < limits.oscUnison; ++r)
        {
            f.oscDetuneRatio[static_cast<std::size_t>(r)] = 1.0f + 0.003f * static_cast<float>(r + 1);
            f.oscDetuneLevel[static_cast<std::size_t>(r)] = 0.5f;
        }
        break;
    case Primitive::Partials:
        for (int m = 0; m < kMaxPartials; ++m)
        {
            f.partials.ratio[static_cast<std::size_t>(m)] = 1.0f + 1.13f * static_cast<float>(m);
            f.partials.amp[static_cast<std::size_t>(m)] = 0.02f;
        }
        break;
    case Primitive::Fm:
        for (int i = 0; i < kMaxFmOperators; ++i)
        {
            f.fm.ratio[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
            f.fm.level[static_cast<std::size_t>(i)] = 0.1f;
            for (int j = i + 1; j < kMaxFmOperators; ++j)
                f.fm.index[static_cast<std::size_t>(i * kMaxFmOperators + j)] = 0.7f;
        }
        break;
    case Primitive::Noise:
        f.noise.band.fill(0.05f);
        break;
    case Primitive::Grain:
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
        break;
    case Primitive::Sampler:
        f.sampler.rate = 1.0f;
        f.sampler.gain = 1.0f;
        f.sampler.eq.fill(0.9f);
        break;
    }
    f.setMix(p, 0.3f);
}

struct SynthRun
{
    double ns = 0.0;    ///< measured ns per sample
    float model = 0.0f; ///< the guard's modelled share (with the safety margin), if a table was given
};

/// The whole synth over ~1 s of audio, with `notes` voices playing `frame` and an optional
/// instance frame. With `table`, also reads the guard's estimate from that table.
SynthRun synthRun(Tier tier, double sr, const VoiceFrame* frame, int notes,
                  const InstanceFrame* inst = nullptr, const CostTable* table = nullptr)
{
    Synth synth;
    synth.prepare(sr, kBlock, tier);
    synth.setBudgetGuardEnabled(false);
    synth.setCostTable(table);
    if (inst != nullptr)
        synth.setInstanceFrame(*inst);
    std::vector<float> bankData(4 * kGrainSourceLength);
    for (std::size_t i = 0; i < bankData.size(); ++i)
        bankData[i] = std::sin(0.01f * static_cast<float>(i));
    GrainBank bank{bankData.data(), 4, kGrainSourceLength};
    std::vector<float> sampleData(static_cast<std::size_t>(sr * 4));
    for (std::size_t i = 0; i < sampleData.size(); ++i)
        sampleData[i] = std::sin(0.05f * static_cast<float>(i));
    SampleData sample{sampleData.data(),
                      static_cast<long long>(sampleData.size()),
                      sr,
                      261.63,
                      1000,
                      static_cast<long long>(sampleData.size()) - 1000};
    synth.setGrainBank(&bank);
    synth.setSample(&sample);
    for (int v = 0; v < notes; ++v)
        synth.noteOn(40 + 2 * v, 0.8f, frame);

    std::vector<float> l(kBlock), r(kBlock);
    for (int i = 0; i < 40; ++i)
        synth.process(l.data(), r.data(), kBlock);
    const int blocks = static_cast<int>(sr / kBlock);
    const int hop = synth.hop();
    int since = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < blocks; ++b)
    {
        since += kBlock;
        if (since >= hop && frame != nullptr)
        {
            since -= hop;
            for (int v = 0; v < synth.voiceCount(); ++v)
                synth.setFrame(v, *frame);
        }
        synth.process(l.data(), r.data(), kBlock);
    }
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    return {ns / (static_cast<double>(blocks) * kBlock), synth.guardReport().estimatedShare};
}

double synthNs(Tier tier, double sr, const VoiceFrame* frame, int notes)
{
    return synthRun(tier, sr, frame, notes).ns;
}

void fillSlot(SlotControls& slot, Family family, int inputIndex)
{
    slot.family = static_cast<float>(family);
    slot.outGain = 0.3f;
    slot.input[static_cast<std::size_t>(inputIndex)] = 1.0f;
    const auto& table = paramTable(family);
    for (int i = 0; i < kSlotParams; ++i)
        slot.params[static_cast<std::size_t>(i)] = table[static_cast<std::size_t>(i)].defaultNorm;
}

/// sum(measured) / sum(modelled, without the safety margin) over the fit patches at one rate.
float fitRateScale(const CostTable& table, int r)
{
    double measured = 0.0, modelled = 0.0;
    for (int t = 0; t < 4; ++t)
    {
        const auto tier = static_cast<Tier>(t);
        const auto limits = limitsFor(tier);
        std::vector<std::unique_ptr<VoiceFrame>> frames;
        std::vector<std::unique_ptr<InstanceFrame>> insts;
        const auto add = [&](std::unique_ptr<VoiceFrame> f, std::unique_ptr<InstanceFrame> i)
        {
            f->attackTau = 0.002f;
            frames.push_back(std::move(f));
            insts.push_back(std::move(i));
        };
        for (int p = 0; p < kPrimitiveCount; ++p)
        {
            auto f = std::make_unique<VoiceFrame>();
            fillPrimitive(*f, static_cast<Primitive>(p), limits);
            add(std::move(f), nullptr);
        }
        auto all = std::make_unique<VoiceFrame>();
        for (int p = 0; p < kPrimitiveCount; ++p)
            fillPrimitive(*all, static_cast<Primitive>(p), limits);
        auto allNl = std::make_unique<VoiceFrame>(*all);
        for (auto& slot : allNl->slots)
            fillSlot(slot, Family::Nonlinear, static_cast<int>(Primitive::Osc));
        add(std::move(all), nullptr);
        add(std::move(allNl), nullptr);
        auto osc = std::make_unique<VoiceFrame>();
        fillPrimitive(*osc, Primitive::Osc, limits);
        auto oscFilter = std::make_unique<VoiceFrame>(*osc);
        for (auto& slot : oscFilter->slots)
            fillSlot(slot, Family::Filter, static_cast<int>(Primitive::Osc));
        add(std::move(oscFilter), nullptr);
        auto reverb = std::make_unique<InstanceFrame>();
        for (auto& slot : reverb->slots)
            fillSlot(slot, Family::Reverb, 0);
        add(std::move(osc), std::move(reverb));

        for (std::size_t i = 0; i < frames.size(); ++i)
        {
            std::vector<double> ns;
            float model = 0.0f;
            for (int k = 0; k < kRuns; ++k)
            {
                const auto run =
                    synthRun(tier, kRates[r], frames[i].get(), limits.polyphony, insts[i].get(), &table);
                ns.push_back(run.ns);
                model = run.model;
            }
            measured += median(ns) * kRates[r] * 1.0e-9;
            modelled += static_cast<double>(model / BudgetGuard::kSafetyMargin);
        }
    }
    return modelled > 0.0 ? static_cast<float>(measured / modelled) : 1.0f;
}

/// ns per sample of one processor at typical params: the median over several param sets.
double processorNs(Family family, Tier tier, double sr, int channels, int quality, IrBank* irs)
{
    auto p = makeProcessor(family);
    ProcessSpec spec;
    spec.sampleRate = sr;
    spec.channels = channels;
    spec.limits = limitsFor(tier);
    p->prepare(spec);
    if (family == Family::Convolution && irs != nullptr)
        irs->publish(*static_cast<ConvolutionProcessor*>(p.get()));
    p->setQuality(quality);

    const auto& table = paramTable(family);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<std::array<float, kSlotParams>> sets(6);
    for (int i = 0; i < kSlotParams; ++i)
    {
        const auto k = static_cast<std::size_t>(i);
        sets[0][k] = table[k].defaultNorm;
        sets[1][k] = 0.5f;
        sets[2][k] = 1.0f;
        sets[3][k] = u(rng);
        sets[4][k] = u(rng);
        sets[5][k] = u(rng);
    }

    ControlContext ctx;
    ctx.f0 = channels == 1 ? 220.0f : 0.0f;
    ctx.velocity = 1.0f;
    std::vector<float> buf(static_cast<std::size_t>(2 * kBlock));
    float* ch[2] = {buf.data(), buf.data() + kBlock};
    std::mt19937 noise(3);
    std::uniform_real_distribution<float> n(-0.5f, 0.5f);

    std::vector<double> perSet;
    for (auto& params : sets)
    {
        p->reset();
        const int blocks = static_cast<int>(0.25 * sr / kBlock);
        double elapsed = 0.0;
        for (int b = 0; b < blocks + 20; ++b)
        {
            for (auto& x : buf)
                x = n(noise);
            params[0] = std::clamp(params[0] + (b % 2 == 0 ? 0.002f : -0.002f), 0.0f, 1.0f);
            const auto t0 = std::chrono::steady_clock::now();
            for (int c = 0; c < kBlock; c += kControlBlock)
            {
                float* sub[2] = {ch[0] + c, ch[1] + c};
                p->setParams(params.data(), ctx);
                p->process(sub, kControlBlock);
            }
            if (b >= 20) // warm-up
                elapsed +=
                    std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        }
        perSet.push_back(elapsed / (static_cast<double>(blocks) * kBlock));
    }
    return median(perSet);
}

void printPair(double a, double b)
{
    std::printf("{%.2ff, %.2ff}", a, b);
}

} // namespace

int main()
{
    const char* tierNames[] = {"Lite", "Standard", "Pro", "Studio"};
    IrBank irs[2];
    for (int r = 0; r < 2; ++r)
    {
        irs[r].setHostRate(kRates[r]);
        SyntheticIrSpec ir;
        ir.sampleRate = kRates[r];
        ir.rt60 = 6.0;
        ir.seconds = 6.0; // the processor's default maximum
        (void)irs[r].add(makeSyntheticIr(ir));
    }

    auto table = std::make_unique<CostTable>();
    const auto f = [](double v)
    {
        return static_cast<float>(v);
    };
    double base[2];
    for (int r = 0; r < 2; ++r)
    {
        std::vector<double> runs;
        for (int k = 0; k < kRuns; ++k)
            runs.push_back(synthNs(Tier::Standard, kRates[r], nullptr, 0));
        base[r] = median(runs);
        table->instanceBase[r] = f(base[r]);
    }

    // Each primitive alone costs base + own; all six together cost base + sum(own), so the
    // voice base is (sum(alone) - together) / 5.
    for (int t = 0; t < 4; ++t)
    {
        const auto tier = static_cast<Tier>(t);
        const auto limits = limitsFor(tier);
        for (int r = 0; r < 2; ++r)
        {
            const auto perVoice = [&](const VoiceFrame* frame)
            {
                std::vector<double> runs;
                for (int k = 0; k < kRuns; ++k)
                    runs.push_back(synthNs(tier, kRates[r], frame, limits.polyphony));
                return std::max(0.0, (median(runs) - base[r]) / limits.polyphony);
            };
            auto all = std::make_unique<VoiceFrame>();
            all->attackTau = 0.002f;
            double prim[kPrimitiveCount];
            double sum = 0.0;
            for (int p = 0; p < kPrimitiveCount; ++p)
            {
                auto frame = std::make_unique<VoiceFrame>();
                frame->attackTau = 0.002f;
                fillPrimitive(*frame, static_cast<Primitive>(p), limits);
                fillPrimitive(*all, static_cast<Primitive>(p), limits);
                prim[p] = perVoice(frame.get());
                sum += prim[p];
            }
            const double together = perVoice(all.get());
            const double voiceBase =
                std::clamp((sum - together) / (kPrimitiveCount - 1), 0.0, sum / kPrimitiveCount);
            table->voiceBase[t][r] = f(voiceBase);
            for (int p = 0; p < kPrimitiveCount; ++p)
                table->primitive[t][p][r] = f(std::max(0.0, prim[p] - voiceBase));
        }
    }

    for (int instance = 0; instance < 2; ++instance)
        for (int t = 0; t < 4; ++t)
            for (int fam = 1; fam < kFamilyCount; ++fam)
            {
                const auto family = static_cast<Family>(fam);
                if (instance == 0 && !allowedInVoice(family))
                    continue;
                const bool reduced = makeProcessor(family)->hasReducedQuality();
                auto& cell = instance ? table->instanceSlot[t][fam] : table->voiceSlot[t][fam];
                for (int q = 0; q < (reduced ? 2 : 1); ++q)
                    for (int r = 0; r < 2; ++r)
                    {
                        std::vector<double> m;
                        for (int k = 0; k < kRuns; ++k)
                            m.push_back(processorNs(family, static_cast<Tier>(t), kRates[r], instance ? 2 : 1,
                                                    q, &irs[r]));
                        cell[q][r] = f(median(m));
                    }
                if (!reduced) // same processing: the same cost, so the guard never "downgrades" it
                    for (int r = 0; r < 2; ++r)
                        cell[1][r] = cell[0][r];
            }

    table->rateScale[0] = 1.0f;
    table->rateScale[1] = 1.0f;
    const float fitted[2] = {fitRateScale(*table, 0), fitRateScale(*table, 1)};
    table->rateScale[0] = fitted[0];
    table->rateScale[1] = fitted[1];

    std::printf(
        "// Generated by alfa_dsp_calibrate (src/dsp/tools/Calibrate.cpp). Do not edit: re-run it on\n"
        "// an idle machine. ns of CPU per sample of audio, {48 kHz, 96 kHz}; see BudgetGuard.h.\n"
        "#pragma once\n\n// clang-format off\nnamespace alfa::dsp::generated\n{\n\n"
        "inline constexpr CostTable kCostTable{\n");
    std::printf("    // voiceBase[tier]\n    {");
    for (int t = 0; t < 4; ++t)
    {
        printPair(table->voiceBase[t][0], table->voiceBase[t][1]);
        std::printf(t < 3 ? ", " : "},\n");
    }
    std::printf("    // primitive[tier][osc, partials, fm, noise, grain, sampler]\n    {\n");
    for (int t = 0; t < 4; ++t)
    {
        std::printf("        {");
        for (int p = 0; p < kPrimitiveCount; ++p)
        {
            printPair(table->primitive[t][p][0], table->primitive[t][p][1]);
            std::printf(p + 1 < kPrimitiveCount ? ", " : "");
        }
        std::printf("}, // %s\n", tierNames[t]);
    }
    std::printf("    },\n");
    for (int instance = 0; instance < 2; ++instance)
    {
        std::printf("    // %s[tier][family][quality]\n    {\n", instance ? "instanceSlot" : "voiceSlot");
        for (int t = 0; t < 4; ++t)
        {
            std::printf("        { // %s\n", tierNames[t]);
            for (int fam = 0; fam < kFamilyCount; ++fam)
            {
                const auto& cell = instance ? table->instanceSlot[t][fam] : table->voiceSlot[t][fam];
                std::printf("            {");
                printPair(cell[0][0], cell[0][1]);
                std::printf(", ");
                printPair(cell[1][0], cell[1][1]);
                const auto name = familyName(static_cast<Family>(fam));
                std::printf("}, // %.*s\n", static_cast<int>(name.size()), name.data());
            }
            std::printf("        },\n");
        }
        std::printf("    },\n");
    }
    std::printf("    // instanceBase\n    ");
    printPair(table->instanceBase[0], table->instanceBase[1]);
    std::printf(",\n    // rateScale: measured / modelled total at each rate\n    {%.3ff, %.3ff},\n",
                static_cast<double>(fitted[0]), static_cast<double>(fitted[1]));
    std::printf("};\n\n} // namespace alfa::dsp::generated\n// clang-format on\n");
    return 0;
}
