// Prints the control-frame layout as JSON, for the training script and the inference host.
// Usage: alfa_dsp_layout > src/dsp/layout.json

#include "alfa/dsp/ControlFrame.h"

#include <cstdio>

int main()
{
    using namespace alfa::dsp;
    std::printf("{\n");
    std::printf("  \"layout_version\": %d,\n", kLayoutVersion);
    std::printf("  \"frame_rate_hz\": %.1f,\n", kFrameRateHz);
    std::printf("  \"frame_floats\": %zu,\n", kFrameFloats);
    std::printf("  \"osc_table_size\": %d,\n", kOscTableSize);
    std::printf("  \"band_low_hz\": %.1f,\n", kBandLowHz);
    std::printf("  \"grain_source_length\": %d,\n", kGrainSourceLength);
    std::printf("  \"primitives\": [\"osc\", \"partials\", \"fm\", \"noise\", \"grain\", \"sampler\"],\n");
    std::printf("  \"tiers\": {\n");
    const char* names[] = {"lite", "standard", "pro", "studio"};
    for (int t = 0; t < 4; ++t)
    {
        const auto l = limitsFor(static_cast<Tier>(t));
        std::printf("    \"%s\": {\"partials\": %d, \"fm_operators\": %d, \"grain_layers\": %d, \"polyphony\": %d}%s\n",
                    names[t], l.partials, l.fmOperators, l.grainLayers, l.polyphony, t < 3 ? "," : "");
    }
    std::printf("  },\n");
    std::printf("  \"fields\": [\n");
    for (std::size_t i = 0; i < kFields.size(); ++i)
    {
        const auto& f = kFields[i];
        std::printf("    {\"name\": \"%.*s\", \"offset\": %zu, \"count\": %zu, \"unit\": \"%.*s\"}%s\n",
                    static_cast<int>(f.name.size()), f.name.data(), f.offset, f.count,
                    static_cast<int>(f.unit.size()), f.unit.data(), i + 1 < kFields.size() ? "," : "");
    }
    std::printf("  ]\n}\n");
    return 0;
}
