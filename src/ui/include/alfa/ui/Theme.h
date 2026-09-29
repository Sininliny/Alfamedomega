#pragma once

#include <juce_graphics/juce_graphics.h>

namespace alfa::ui
{

/// The colour tokens from docs/ui/ui-design.md, "Visual system". Colour means
/// one thing: accent is a value you set, mod is modulation, warn is a warning.
struct Theme
{
    juce::String name;
    juce::Colour bg, panel, line, text, text2, accent, mod, warn;

    /// Parses a theme token file: {"name": "...", "bg": "#141518", ...}. Missing
    /// tokens fall back to Graphite.
    static Theme fromJson(const juce::String& json);
    juce::String toJson() const;

    static Theme graphite();
    static Theme paper();
    static Theme highContrast();

    /// Looks up a built-in theme by the ids UiState stores ("graphite", "paper",
    /// "contrast").
    static Theme builtIn(const juce::String& id);
    static juce::StringArray builtInIds();
    static juce::StringArray builtInNames();
};

} // namespace alfa::ui
