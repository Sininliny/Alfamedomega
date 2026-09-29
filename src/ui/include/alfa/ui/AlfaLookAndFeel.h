#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace alfa::ui
{

/// Placeholder look-and-feel. The UI owner replaces this with the token-driven theme
/// described in docs/ui/ui-design.md.
class AlfaLookAndFeel : public juce::LookAndFeel_V4
{
public:
    AlfaLookAndFeel();
};

} // namespace alfa::ui
