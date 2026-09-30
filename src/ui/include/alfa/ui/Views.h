#pragma once

#include "alfa/ui/Controls.h"
#include "alfa/ui/LiveData.h"

namespace alfa::ui
{

/// Design size from ui-design.md; everything is laid out at this size and the
/// editor scales it uniformly (75% to 200%).
inline constexpr int kDesignWidth = 1280;
inline constexpr int kDesignHeight = 800;

/// Spacing between panels, from the density setting.
int gapFor(const UiState& ui);

// Header
// ----------------------------------------------------------------------------

class HeaderBar : public juce::Component, private LiveData::Listener, private juce::ValueTree::Listener
{
public:
    HeaderBar(EditorContext&, LiveData&);
    ~HeaderBar() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void liveDataUpdated() override;
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    void refreshPreset();
    void refreshRating();
    void showPresetMenu();

    EditorContext& ctx;
    LiveData& live;
    Segmented viewSwitch;
    juce::TextButton prevPreset{"<"}, nextPreset{">"}, presetName;
    juce::TextButton like{"+"}, dislike{"-"};
    juce::ComboBox tierBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> tierAttachment;
    juce::String statsText;
    bool statsWarn = false;
    juce::Rectangle<int> statsArea;
};

// Perform
// ----------------------------------------------------------------------------

/// Calls back on the message thread when the sound's method or hybrid
/// components change, so method-dependent layouts can rebuild.
class ShapeWatcher : private juce::AudioProcessorValueTreeState::Listener
{
public:
    ShapeWatcher(juce::AudioProcessorValueTreeState&, std::function<void()> onChange);
    ~ShapeWatcher() override;

private:
    void parameterChanged(const juce::String&, float) override;
    juce::AudioProcessorValueTreeState& params;
    std::function<void()> callback;
    std::shared_ptr<bool> alive = std::make_shared<bool>(true);
};

/// The nine methods as a pick list: capacity share, a live activity dot, and
/// the two components when the sound is a hybrid. Replaces the Method blend of
/// the first layout: a sound has one method, as the model was trained.
class MethodPicker : public juce::Component, public juce::SettableTooltipClient, private LiveData::Listener
{
public:
    MethodPicker(EditorContext&, LiveData&, juce::String title);
    ~MethodPicker() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void mouseUp(const juce::MouseEvent&) override;
    void mouseMove(const juce::MouseEvent&) override;
    void mouseExit(const juce::MouseEvent&) override;

    /// Rows show the method's one-line summary when there is room.
    void setShowSummaries(bool show);

private:
    void liveDataUpdated() override;
    void select(int method);
    int rowAt(juce::Point<int>) const;

    EditorContext& ctx;
    LiveData& live;
    juce::String title;
    bool summaries = false;
    int hover = -1;
    juce::ComboBox hybridA, hybridB;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> attachA, attachB;
    std::vector<juce::Rectangle<int>> rows;
    juce::Rectangle<int> hybridArea;
    ShapeWatcher watcher;
};

class TimbrePad : public juce::Component,
                  public juce::SettableTooltipClient,
                  private juce::AudioProcessorValueTreeState::Listener
{
public:
    explicit TimbrePad(EditorContext&);
    ~TimbrePad() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent&) override;
    void mouseDrag(const juce::MouseEvent&) override;
    void mouseUp(const juce::MouseEvent&) override;

private:
    void parameterChanged(const juce::String&, float) override;
    void bind();
    void setFromPosition(juce::Point<float>);
    juce::Rectangle<float> padArea() const;
    void storeSnapshot(int slot);
    void recallSnapshot(int slot);
    void randomizeNearby();

    EditorContext& ctx;
    int xKnob = 0, yKnob = 1;
    juce::RangedAudioParameter* xParam = nullptr;
    juce::RangedAudioParameter* yParam = nullptr;
    params::KnobInfo xInfo, yInfo;
    std::array<juce::Point<float>, 2> snapshots{juce::Point<float>{-1.0f, -1.0f}, {-1.0f, -1.0f}};
    juce::TextButton snapA{"A"}, snapB{"B"}, randomize{"Randomize nearby"};
    juce::ComboBox nearbyMode;
    juce::String nearbyNote; // why the last randomize used another mode
    ShapeWatcher watcher;
};

/// The Signal flow strip in compact form, for the Perform view (off by
/// default).
class FlowStrip : public juce::Component, public juce::SettableTooltipClient, private LiveData::Listener
{
public:
    FlowStrip(EditorContext&, LiveData&);
    ~FlowStrip() override;
    void paint(juce::Graphics&) override;
    void mouseUp(const juce::MouseEvent&) override;

private:
    void liveDataUpdated() override { repaint(); }
    EditorContext& ctx;
    LiveData& live;
};

class PerformView : public juce::Component, private LiveData::Listener, private juce::ValueTree::Listener
{
public:
    PerformView(EditorContext&, LiveData&);
    ~PerformView() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void liveDataUpdated() override;
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    void rebuildKnobs();
    void rebuildVoice();

    EditorContext& ctx;
    LiveData& live;

    MethodPicker methods;
    juce::OwnedArray<ParamControl> knobControls;
    std::unique_ptr<Segmented> knobPages;
    TimbrePad timbre;
    juce::OwnedArray<ParamControl> voiceControls;
    juce::MidiKeyboardComponent keyboard;
    FlowStrip flowStrip;
    ShapeWatcher watcher;
    juce::String knobsSubtitle;

    juce::Rectangle<int> methodArea, knobArea, timbreArea, voiceArea, envelopeArea, meterArea, keyboardArea,
        stripArea;
};

/// The knobs a method actually uses, in knob order.
std::vector<int> usedKnobs(const juce::AudioProcessorValueTreeState&);

/// The amp envelope's knobs for the current sound (physical has its own).
std::array<int, 4> envelopeKnobs(const juce::AudioProcessorValueTreeState&);

// Signal flow
// ------------------------------------------------------------------------

class SignalFlowView : public juce::Component, private LiveData::Listener, private juce::ValueTree::Listener
{
public:
    SignalFlowView(EditorContext&, LiveData&);
    ~SignalFlowView() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void mouseUp(const juce::MouseEvent&) override;

    /// One stage of the synthesis pipeline as this view explains it, written for
    /// three audiences: Musician (what it does), Student (what data goes in and
    /// out) and Developer (thread, rate, cost).
    struct Step
    {
        juce::String id;
        juce::String title;
        juce::String input;
        juce::String output;
        juce::String thread;
        juce::String note; // explanatory note, shown as a quotation
        std::vector<juce::String> paramIds;
    };

    static const std::vector<Step>& steps();

private:
    void liveDataUpdated() override;
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    void rebuildInspector();
    void paintCard(juce::Graphics&, const Step&, juce::Rectangle<int>, bool selected);
    void paintInspector(juce::Graphics&);
    juce::String detail() const;

    EditorContext& ctx;
    LiveData& live;
    Segmented detailLevel;
    juce::ToggleButton explain{"Explain each step"};
    juce::OwnedArray<juce::TextButton> stepChips;
    juce::OwnedArray<ParamControl> inspectorControls;

    std::vector<std::pair<int, juce::Rectangle<int>>> cardAreas;
    juce::Rectangle<int> toolbarArea, pipelineArea, inspectorArea, inspectorViz, inspectorParams,
        inspectorNotes;
    juce::String hiddenText;
};

// Sound design
// -----------------------------------------------------------------------

class SoundDesignView : public juce::Component, private LiveData::Listener, private juce::ValueTree::Listener
{
public:
    SoundDesignView(EditorContext&, LiveData&);
    ~SoundDesignView() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void liveDataUpdated() override;
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    void rebuild();

    EditorContext& ctx;
    LiveData& live;
    MethodPicker methods;
    juce::OwnedArray<ParamControl> knobControls;
    juce::ToggleButton showUnused{"Show unused knobs"};
    juce::TextButton addRoute{"Add route"};
    ShapeWatcher watcher;

    juce::Rectangle<int> methodArea, layerArea, displayArea, knobGrid, modArea;
};

// Customize
// --------------------------------------------------------------------------

class CustomizeView : public juce::Component, private juce::ValueTree::Listener
{
public:
    explicit CustomizeView(EditorContext&);
    ~CustomizeView() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    void syncFromState();
    void syncLearning();
    void applyConsent();
    void visibilityChanged() override { syncLearning(); }
    void exportTheme();
    void importTheme();

    EditorContext& ctx;
    Segmented themeChoice, density, gesture, valueDisplay, macrosPerPage;
    juce::ComboBox accent;
    juce::Slider scale;
    juce::OwnedArray<juce::ToggleButton> panelToggles;
    juce::ToggleButton learn{"Learn from how I play"};
    std::array<juce::ToggleButton, 4> learnFrom; // performance, parameters, suggestions, workflow
    juce::Slider personalization;
    juce::StringArray learned;
    juce::TextButton useForNew{"Use for new instances"}, exportButton{"Export theme file..."},
        importButton{"Import theme file..."};
    std::unique_ptr<juce::FileChooser> chooser;
    juce::String statusText;

    juce::Rectangle<int> settingsArea, previewArea, learningArea, personalizationLabel, learnedArea;
    std::vector<std::pair<juce::String, juce::Rectangle<int>>> rowLabels;
};

// Root
// ------------------------------------------------------------------------------

/// The whole interface at design size: header plus the four views. The editor
/// owns one and scales it.
class MainComponent : public juce::Component, private juce::ValueTree::Listener
{
public:
    explicit MainComponent(EditorContext&);
    ~MainComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    LiveData& liveData() noexcept { return live; }

    /// Applies theme and accent from UiState to the look-and-feel.
    void applyTheme();

private:
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    void showView();

    EditorContext& ctx;
    LiveData live;
    HeaderBar header;
    PerformView perform;
    SignalFlowView signalFlow;
    SoundDesignView soundDesign;
    CustomizeView customize;
    juce::TooltipWindow tooltips{this, 600};
};

} // namespace alfa::ui
