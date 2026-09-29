#pragma once

#include <juce_data_structures/juce_data_structures.h>

namespace alfa::ui
{

/// Per-instance interface settings: view, theme, size, which panels and signal
/// flow steps are shown. Saved with the plugin instance and, on request, as the
/// per-user default for new instances. Never saved in presets (ui-design.md
/// principle 5).
class UiState
{
public:
    static inline const juce::Identifier type{"UIState"};

    // Property names. Values are strings or numbers so the tree stays readable
    // when saved as XML.
    struct Ids
    {
        static inline const juce::Identifier view{"view"};               // perform, signal, sound, customize
        static inline const juce::Identifier theme{"theme"};             // graphite, paper, contrast
        static inline const juce::Identifier accent{"accent"};           // #RRGGBB, empty for theme default
        static inline const juce::Identifier scale{"scale"};             // 0.75 to 2.0
        static inline const juce::Identifier density{"density"};         // comfortable, compact
        static inline const juce::Identifier knobGesture{"knobGesture"}; // vertical, horizontal, circular
        static inline const juce::Identifier valueDisplay{"valueDisplay"}; // units, percent, hover
        static inline const juce::Identifier macrosPerPage{"macrosPerPage"};
        static inline const juce::Identifier macroPage{"macroPage"};
        static inline const juce::Identifier showBlend{"showBlend"};
        static inline const juce::Identifier showMacros{"showMacros"};
        static inline const juce::Identifier showTimbre{"showTimbre"};
        static inline const juce::Identifier showVoice{"showVoice"};
        static inline const juce::Identifier showKeyboard{"showKeyboard"};
        static inline const juce::Identifier showFlowStrip{"showFlowStrip"};
        static inline const juce::Identifier flowDetail{"flowDetail"}; // musician, student, developer
        static inline const juce::Identifier flowExplain{"flowExplain"};
        static inline const juce::Identifier flowHidden{"flowHidden"}; // comma-separated step ids
        static inline const juce::Identifier flowSelected{"flowSelected"};
        static inline const juce::Identifier soundLayer{"soundLayer"};
        static inline const juce::Identifier locks{"locks"}; // comma-separated parameter ids
    };

    UiState();

    juce::ValueTree& tree() noexcept { return state; }
    const juce::ValueTree& tree() const noexcept { return state; }

    /// Replaces the settings with a saved tree, filling any missing property with
    /// its default.
    void restore(const juce::ValueTree& saved);

    juce::var get(const juce::Identifier& id) const { return state.getProperty(id); }
    void set(const juce::Identifier& id, const juce::var& v) { state.setProperty(id, v, nullptr); }

    juce::String getString(const juce::Identifier& id) const { return state.getProperty(id).toString(); }
    bool getBool(const juce::Identifier& id) const { return static_cast<bool>(state.getProperty(id)); }
    int getInt(const juce::Identifier& id) const { return static_cast<int>(state.getProperty(id)); }
    double getDouble(const juce::Identifier& id) const { return static_cast<double>(state.getProperty(id)); }

    bool isStepHidden(const juce::String& stepId) const;
    void setStepHidden(const juce::String& stepId, bool hidden);

    juce::StringArray lockedParameters() const;
    bool isLocked(const juce::String& paramId) const;
    void setLocked(const juce::String& paramId, bool locked);

    /// Per-user defaults ("Use for new instances" in Customize).
    void saveAsUserDefault() const;
    void loadUserDefault();
    static juce::File userSettingsFile();

    static juce::ValueTree createDefault();

private:
    static juce::StringArray splitList(const juce::String&);
    juce::ValueTree state;
};

} // namespace alfa::ui
