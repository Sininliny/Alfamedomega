// TEMPORARY until the family branches merge; replaced by Family.cpp.
#include "alfa/dsp/Family.h"
namespace alfa::dsp
{
std::string_view familyName(Family) noexcept { return "unknown"; }
const ParamTable& paramTable(Family) noexcept { static const ParamTable t{}; return t; }
std::unique_ptr<Processor> makeProcessor(Family) { return nullptr; }
} // namespace alfa::dsp
