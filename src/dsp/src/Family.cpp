#include "alfa/dsp/Family.h"

#include "alfa/dsp/Convolution.h"
#include "alfa/dsp/Delay.h"
#include "alfa/dsp/Filters.h"
#include "alfa/dsp/Nonlinear.h"
#include "alfa/dsp/Physical.h"
#include "alfa/dsp/PitchTime.h"
#include "alfa/dsp/Spatial.h"
#include "alfa/dsp/Spectral.h"

namespace alfa::dsp
{

std::string_view familyName(Family f) noexcept
{
    switch (f)
    {
    case Family::Off:
        return "off";
    case Family::Filter:
        return "filter";
    case Family::Comb:
        return "comb";
    case Family::Formant:
        return "formant";
    case Family::Resonator:
        return "resonator";
    case Family::Nonlinear:
        return "nonlinear";
    case Family::Waveguide:
        return "waveguide";
    case Family::ModDelay:
        return "mod_delay";
    case Family::Phaser:
        return "phaser";
    case Family::Body:
        return "body";
    case Family::Delay:
        return "delay";
    case Family::Reverb:
        return "reverb";
    case Family::Convolution:
        return "convolution";
    case Family::Spectral:
        return "spectral";
    case Family::PitchShift:
        return "pitch_shift";
    case Family::Spatial:
        return "spatial";
    }
    return "unknown";
}

const ParamTable& paramTable(Family f) noexcept
{
    static const ParamTable kEmpty{};
    switch (f)
    {
    case Family::Off:
        return kEmpty;
    case Family::Filter:
        return FilterProcessor::params();
    case Family::Comb:
        return CombProcessor::params();
    case Family::Formant:
        return FormantProcessor::params();
    case Family::Resonator:
        return ResonatorProcessor::params();
    case Family::Nonlinear:
        return NonlinearProcessor::params();
    case Family::Waveguide:
        return WaveguideProcessor::params();
    case Family::ModDelay:
        return ModDelayProcessor::params();
    case Family::Phaser:
        return PhaserProcessor::params();
    case Family::Body:
        return BodyProcessor::params();
    case Family::Delay:
        return DelayProcessor::params();
    case Family::Reverb:
        return ReverbProcessor::params();
    case Family::Convolution:
        return ConvolutionProcessor::params();
    case Family::Spectral:
        return SpectralProcessor::params();
    case Family::PitchShift:
        return PitchShiftProcessor::params();
    case Family::Spatial:
        return SpatialProcessor::params();
    }
    return kEmpty;
}

std::unique_ptr<Processor> makeProcessor(Family f)
{
    switch (f)
    {
    case Family::Off:
        return nullptr;
    case Family::Filter:
        return std::make_unique<FilterProcessor>();
    case Family::Comb:
        return std::make_unique<CombProcessor>();
    case Family::Formant:
        return std::make_unique<FormantProcessor>();
    case Family::Resonator:
        return std::make_unique<ResonatorProcessor>();
    case Family::Nonlinear:
        return std::make_unique<NonlinearProcessor>();
    case Family::Waveguide:
        return std::make_unique<WaveguideProcessor>();
    case Family::ModDelay:
        return std::make_unique<ModDelayProcessor>();
    case Family::Phaser:
        return std::make_unique<PhaserProcessor>();
    case Family::Body:
        return std::make_unique<BodyProcessor>();
    case Family::Delay:
        return std::make_unique<DelayProcessor>();
    case Family::Reverb:
        return std::make_unique<ReverbProcessor>();
    case Family::Convolution:
        return std::make_unique<ConvolutionProcessor>();
    case Family::Spectral:
        return std::make_unique<SpectralProcessor>();
    case Family::PitchShift:
        return std::make_unique<PitchShiftProcessor>();
    case Family::Spatial:
        return std::make_unique<SpatialProcessor>();
    }
    return nullptr;
}

} // namespace alfa::dsp
