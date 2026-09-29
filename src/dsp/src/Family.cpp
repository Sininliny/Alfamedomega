// Families whose branch has not merged yet compile out (__has_include), and their slots stay
// silent. The guards go once every family is in.
#include "alfa/dsp/Family.h"

#if __has_include("alfa/dsp/Convolution.h")
#include "alfa/dsp/Convolution.h"
#define ALFA_HAS_CONVOLUTION 1
#endif
#if __has_include("alfa/dsp/Delay.h")
#include "alfa/dsp/Delay.h"
#define ALFA_HAS_DELAY 1
#endif
#if __has_include("alfa/dsp/Filters.h")
#include "alfa/dsp/Filters.h"
#define ALFA_HAS_FILTERS 1
#endif
#if __has_include("alfa/dsp/Nonlinear.h")
#include "alfa/dsp/Nonlinear.h"
#define ALFA_HAS_NONLINEAR 1
#endif
#if __has_include("alfa/dsp/Physical.h")
#include "alfa/dsp/Physical.h"
#define ALFA_HAS_PHYSICAL 1
#endif
#if __has_include("alfa/dsp/PitchTime.h")
#include "alfa/dsp/PitchTime.h"
#define ALFA_HAS_PITCHTIME 1
#endif
#if __has_include("alfa/dsp/Spatial.h")
#include "alfa/dsp/Spatial.h"
#define ALFA_HAS_SPATIAL 1
#endif
#if __has_include("alfa/dsp/Spectral.h")
#include "alfa/dsp/Spectral.h"
#define ALFA_HAS_SPECTRAL 1
#endif

namespace alfa::dsp
{

std::string_view familyName(Family f) noexcept
{
    switch (f)
    {
    case Family::Off: return "off";
    case Family::Filter: return "filter";
    case Family::Comb: return "comb";
    case Family::Formant: return "formant";
    case Family::Resonator: return "resonator";
    case Family::Nonlinear: return "nonlinear";
    case Family::Waveguide: return "waveguide";
    case Family::ModDelay: return "mod_delay";
    case Family::Phaser: return "phaser";
    case Family::Body: return "body";
    case Family::Delay: return "delay";
    case Family::Reverb: return "reverb";
    case Family::Convolution: return "convolution";
    case Family::Spectral: return "spectral";
    case Family::PitchShift: return "pitch_shift";
    case Family::Spatial: return "spatial";
    }
    return "unknown";
}

const ParamTable& paramTable(Family f) noexcept
{
    static const ParamTable kEmpty{};
    switch (f)
    {
    case Family::Off: return kEmpty;
    default: break;
#ifdef ALFA_HAS_FILTERS
    case Family::Filter: return FilterProcessor::params();
#endif
#ifdef ALFA_HAS_FILTERS
    case Family::Comb: return CombProcessor::params();
#endif
#ifdef ALFA_HAS_FILTERS
    case Family::Formant: return FormantProcessor::params();
#endif
#ifdef ALFA_HAS_PHYSICAL
    case Family::Resonator: return ResonatorProcessor::params();
#endif
#ifdef ALFA_HAS_NONLINEAR
    case Family::Nonlinear: return NonlinearProcessor::params();
#endif
#ifdef ALFA_HAS_PHYSICAL
    case Family::Waveguide: return WaveguideProcessor::params();
#endif
#ifdef ALFA_HAS_DELAY
    case Family::ModDelay: return ModDelayProcessor::params();
#endif
#ifdef ALFA_HAS_DELAY
    case Family::Phaser: return PhaserProcessor::params();
#endif
#ifdef ALFA_HAS_FILTERS
    case Family::Body: return BodyProcessor::params();
#endif
#ifdef ALFA_HAS_DELAY
    case Family::Delay: return DelayProcessor::params();
#endif
#ifdef ALFA_HAS_DELAY
    case Family::Reverb: return ReverbProcessor::params();
#endif
#ifdef ALFA_HAS_CONVOLUTION
    case Family::Convolution: return ConvolutionProcessor::params();
#endif
#ifdef ALFA_HAS_SPECTRAL
    case Family::Spectral: return SpectralProcessor::params();
#endif
#ifdef ALFA_HAS_PITCHTIME
    case Family::PitchShift: return PitchShiftProcessor::params();
#endif
#ifdef ALFA_HAS_SPATIAL
    case Family::Spatial: return SpatialProcessor::params();
#endif
    }
    return kEmpty;
}

std::unique_ptr<Processor> makeProcessor(Family f)
{
    switch (f)
    {
    case Family::Off: return nullptr;
    default: break;
#ifdef ALFA_HAS_FILTERS
    case Family::Filter: return std::make_unique<FilterProcessor>();
#endif
#ifdef ALFA_HAS_FILTERS
    case Family::Comb: return std::make_unique<CombProcessor>();
#endif
#ifdef ALFA_HAS_FILTERS
    case Family::Formant: return std::make_unique<FormantProcessor>();
#endif
#ifdef ALFA_HAS_PHYSICAL
    case Family::Resonator: return std::make_unique<ResonatorProcessor>();
#endif
#ifdef ALFA_HAS_NONLINEAR
    case Family::Nonlinear: return std::make_unique<NonlinearProcessor>();
#endif
#ifdef ALFA_HAS_PHYSICAL
    case Family::Waveguide: return std::make_unique<WaveguideProcessor>();
#endif
#ifdef ALFA_HAS_DELAY
    case Family::ModDelay: return std::make_unique<ModDelayProcessor>();
#endif
#ifdef ALFA_HAS_DELAY
    case Family::Phaser: return std::make_unique<PhaserProcessor>();
#endif
#ifdef ALFA_HAS_FILTERS
    case Family::Body: return std::make_unique<BodyProcessor>();
#endif
#ifdef ALFA_HAS_DELAY
    case Family::Delay: return std::make_unique<DelayProcessor>();
#endif
#ifdef ALFA_HAS_DELAY
    case Family::Reverb: return std::make_unique<ReverbProcessor>();
#endif
#ifdef ALFA_HAS_CONVOLUTION
    case Family::Convolution: return std::make_unique<ConvolutionProcessor>();
#endif
#ifdef ALFA_HAS_SPECTRAL
    case Family::Spectral: return std::make_unique<SpectralProcessor>();
#endif
#ifdef ALFA_HAS_PITCHTIME
    case Family::PitchShift: return std::make_unique<PitchShiftProcessor>();
#endif
#ifdef ALFA_HAS_SPATIAL
    case Family::Spatial: return std::make_unique<SpatialProcessor>();
#endif
    }
    return nullptr;
}

} // namespace alfa::dsp
