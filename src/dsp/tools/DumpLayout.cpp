// Prints the control-frame layout as JSON, for the training script and the inference host.
// Usage: alfa_dsp_layout > src/dsp/layout.json
//
// Contents: frame sizes and constants, the tier limits (generated from the tier spec), the flat
// field offsets of VoiceFrame and InstanceFrame, the modulation sources and destinations, and
// every processing family with its param table (param i of a slot is entry i; see mapParam).

#include "alfa/dsp/ControlFrame.h"
#include "alfa/dsp/Family.h"
#include "alfa/dsp/Processor.h"

#include <cstdio>
#include <span>
#include <string_view>

using namespace alfa::dsp;

namespace
{

void printString(std::string_view s)
{
    std::printf("\"%.*s\"", static_cast<int>(s.size()), s.data());
}

void printFields(const char* key, std::span<const FieldInfo> fields, bool last)
{
    std::printf("  \"%s\": [\n", key);
    for (std::size_t i = 0; i < fields.size(); ++i)
    {
        const auto& f = fields[i];
        std::printf("    {\"name\": ");
        printString(f.name);
        std::printf(", \"offset\": %zu, \"count\": %zu, \"unit\": ", f.offset, f.count);
        printString(f.unit);
        std::printf("}%s\n", i + 1 < fields.size() ? "," : "");
    }
    std::printf("  ]%s\n", last ? "" : ",");
}

const char* curveName(Curve c)
{
    switch (c)
    {
    case Curve::Linear:
        return "linear";
    case Curve::Log:
        return "log";
    case Curve::Choice:
        return "choice";
    case Curve::Toggle:
        return "toggle";
    }
    return "linear";
}

} // namespace

int main()
{
    std::printf("{\n");
    std::printf("  \"layout_version\": %d,\n", kLayoutVersion);
    std::printf("  \"frame_rate_hz\": %.1f,\n", kFrameRateHz);
    std::printf("  \"frame_floats\": %zu,\n", kFrameFloats);
    std::printf("  \"instance_frame_floats\": %zu,\n", kInstanceFrameFloats);
    std::printf("  \"osc_table_size\": %d,\n", kOscTableSize);
    std::printf("  \"band_low_hz\": %.1f,\n", kBandLowHz);
    std::printf("  \"grain_source_length\": %d,\n", kGrainSourceLength);
    std::printf("  \"control_block\": %d,\n", kControlBlock);
    std::printf("  \"slot_params\": %d,\n", kSlotParams);
    std::printf("  \"slot_inputs\": %d,\n", kSlotInputs);
    std::printf("  \"max_voice_slots\": %d,\n", kMaxVoiceSlots);
    std::printf("  \"max_instance_slots\": %d,\n", kMaxInstanceSlots);
    std::printf("  \"max_mod_routes\": %d,\n", kMaxModRoutes);
    std::printf("  \"primitives\": [\"osc\", \"partials\", \"fm\", \"noise\", \"grain\", \"sampler\"],\n");

    std::printf("  \"tiers\": {\n");
    const char* names[] = {"lite", "standard", "pro", "studio"};
    for (int t = 0; t < 4; ++t)
    {
        const auto l = limitsFor(static_cast<Tier>(t));
        std::printf(
            "    \"%s\": {\"partials\": %d, \"fm_operators\": %d, \"grain_layers\": %d, \"polyphony\": %d, "
            "\"voice_slots\": %d, \"instance_slots\": %d, \"mod_routes\": %d, \"resonator_modes\": %d, "
            "\"osc_unison\": %d}%s\n",
            names[t], l.partials, l.fmOperators, l.grainLayers, l.polyphony, l.voiceSlots, l.instanceSlots,
            l.modRoutes, l.resonatorModes, l.oscUnison, t < 3 ? "," : "");
    }
    std::printf("  },\n");

    printFields("fields", kFields, false);
    printFields("instance_fields", kInstanceFields, false);

    std::printf("  \"mod_sources\": [\"none\", \"env1\", \"env2\", \"lfo1\", \"lfo2\", \"sample_hold\", "
                "\"smooth_random\", \"follower\", \"velocity\", \"key\"],\n");
    std::printf(
        "  \"mod_destinations\": {\"slot_param\": %d, \"slot_out_gain\": %d, \"mix\": %d, \"pitch\": %d, "
        "\"pan\": %d, \"dry_gain\": %d, \"count\": %d},\n",
        moddest::slotParam, moddest::slotOutGain, moddest::mix, moddest::pitch, moddest::pan,
        moddest::dryGain, kModDestinations);

    std::printf("  \"families\": [\n");
    for (int f = 0; f < kFamilyCount; ++f)
    {
        const auto family = static_cast<Family>(f);
        std::printf("    {\"id\": %d, \"name\": ", f);
        printString(familyName(family));
        std::printf(", \"voice\": %s, \"params\": [", allowedInVoice(family) ? "true" : "false");
        const auto& table = paramTable(family);
        bool first = true;
        for (int i = 0; i < kSlotParams; ++i)
        {
            const auto& p = table[static_cast<std::size_t>(i)];
            if (p.name.empty())
                continue;
            std::printf("%s\n      {\"index\": %d, \"name\": ", first ? "" : ",", i);
            printString(p.name);
            std::printf(", \"min\": %g, \"max\": %g, \"curve\": \"%s\", \"default\": %g, \"unit\": ",
                        static_cast<double>(p.min), static_cast<double>(p.max), curveName(p.curve),
                        static_cast<double>(p.defaultNorm));
            printString(p.unit);
            std::printf("}");
            first = false;
        }
        std::printf("%s]}%s\n", first ? "" : "\n    ", f + 1 < kFamilyCount ? "," : "");
    }
    std::printf("  ]\n}\n");
    return 0;
}
