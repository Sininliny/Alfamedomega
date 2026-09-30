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
// - instance base: the synth with no notes (bus, instance chain with every slot off).

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

/// ns per sample for the whole synth over ~1 s of audio, with `notes` voices playing `frame`.
double synthNs(Tier tier, double sr, const VoiceFrame* frame, int notes)
{
    Synth synth;
    synth.prepare(sr, kBlock, tier);
    synth.setBudgetGuardEnabled(false);
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
    return ns / (static_cast<double>(blocks) * kBlock);
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

    double base[2];
    for (int r = 0; r < 2; ++r)
    {
        std::vector<double> runs;
        for (int k = 0; k < kRuns; ++k)
            runs.push_back(synthNs(Tier::Standard, kRates[r], nullptr, 0));
        base[r] = median(runs);
    }

    std::printf(
        "// Generated by alfa_dsp_calibrate (src/dsp/tools/Calibrate.cpp). Do not edit: re-run it on\n"
        "// an idle machine. ns of CPU per sample of audio, {48 kHz, 96 kHz}; see BudgetGuard.h.\n"
        "#pragma once\n\n// clang-format off\nnamespace alfa::dsp::generated\n{\n\n"
        "inline constexpr CostTable kCostTable{\n");
    // Each primitive alone costs base + own; all six together cost base + sum(own), so the
    // voice base is (sum(alone) - together) / 5.
    double prim[4][kPrimitiveCount][2];
    double voiceBase[4][2];
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
            double sum = 0.0;
            for (int p = 0; p < kPrimitiveCount; ++p)
            {
                auto frame = std::make_unique<VoiceFrame>();
                frame->attackTau = 0.002f;
                fillPrimitive(*frame, static_cast<Primitive>(p), limits);
                fillPrimitive(*all, static_cast<Primitive>(p), limits);
                prim[t][p][r] = perVoice(frame.get());
                sum += prim[t][p][r];
            }
            const double together = perVoice(all.get());
            voiceBase[t][r] =
                std::clamp((sum - together) / (kPrimitiveCount - 1), 0.0, sum / kPrimitiveCount);
            for (int p = 0; p < kPrimitiveCount; ++p)
                prim[t][p][r] = std::max(0.0, prim[t][p][r] - voiceBase[t][r]);
        }
    }
    std::printf("    // voiceBase[tier]\n    {");
    for (int t = 0; t < 4; ++t)
    {
        printPair(voiceBase[t][0], voiceBase[t][1]);
        std::printf(t < 3 ? ", " : "},\n");
    }
    std::printf("    // primitive[tier][osc, partials, fm, noise, grain, sampler]\n    {\n");
    for (int t = 0; t < 4; ++t)
    {
        std::printf("        {");
        for (int p = 0; p < kPrimitiveCount; ++p)
        {
            printPair(prim[t][p][0], prim[t][p][1]);
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
            for (int f = 0; f < kFamilyCount; ++f)
            {
                const auto family = static_cast<Family>(f);
                const bool runs = f != 0 && (instance == 1 || allowedInVoice(family));
                const bool reduced = runs && makeProcessor(family)->hasReducedQuality();
                std::printf("            {");
                double full[2] = {0.0, 0.0};
                for (int q = 0; q < 2; ++q)
                {
                    double v[2] = {0.0, 0.0};
                    if (q == 1 && !reduced)
                    {
                        // Same processing: the same cost, so the guard never "downgrades" it.
                        printPair(full[0], full[1]);
                        continue;
                    }
                    for (int r = 0; r < 2 && runs; ++r)
                    {
                        std::vector<double> m;
                        for (int k = 0; k < kRuns; ++k)
                            m.push_back(processorNs(family, static_cast<Tier>(t), kRates[r], instance ? 2 : 1,
                                                    q, &irs[r]));
                        v[r] = median(m);
                    }
                    if (q == 0)
                    {
                        full[0] = v[0];
                        full[1] = v[1];
                    }
                    printPair(v[0], v[1]);
                    std::printf(q == 0 ? ", " : "");
                }
                const auto name = familyName(family);
                std::printf("}, // %.*s\n", static_cast<int>(name.size()), name.data());
            }
            std::printf("        },\n");
        }
        std::printf("    },\n");
    }
    std::printf("    // instanceBase\n    ");
    printPair(base[0], base[1]);
    std::printf(",\n};\n\n} // namespace alfa::dsp::generated\n// clang-format on\n");
    return 0;
}
