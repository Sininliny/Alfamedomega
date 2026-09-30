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
    void showPresetMenu();

    EditorContext& ctx;
    LiveData& live;
    Segmented viewSwitch;
    juce::TextButton prevPreset{"<"}, nextPreset{">"}, presetName;
    juce::ComboBox tierBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> tierAttachment;
    juce::String statsText;
    bool statsWarn = false;
    juce::Rectangle<int> statsArea;
};

// Perform
// ----------------------------------------------------------------------------

class TimbrePad : public juce::Component, private juce::AudioProcessorValueTreeState::Listener
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
    void setFromPosition(juce::Point<float>);
    juce::Rectangle<float> padArea() const;
    void storeSnapshot(int slot);
    void recallSnapshot(int slot);
    void randomizeNearby();

    EditorContext& ctx;
    juce::RangedAudioParameter* xParam;
    juce::RangedAudioParameter* yParam;
    std::array<juce::Point<float>, 2> snapshots{juce::Point<float>{-1.0f, -1.0f}, {-1.0f, -1.0f}};
    juce::TextButton snapA{"A"}, snapB{"B"}, randomize{"Randomize nearby"};
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
    void rebuildMacros();

    EditorContext& ctx;
    LiveData& live;

    juce::OwnedArray<ParamControl> blendControls;
    juce::OwnedArray<ParamControl> macroControls;
    Segmented macroPages;
    TimbrePad timbre;
    juce::OwnedArray<ParamControl> voiceControls;
    juce::MidiKeyboardComponent keyboard;
    FlowStrip flowStrip;

    juce::Rectangle<int> blendArea, macroArea, timbreArea, voiceArea, envelopeArea, meterArea, keyboardArea,
        stripArea;
};

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
    void mouseUp(const juce::MouseEvent&) override;

private:
    void liveDataUpdated() override;
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    void selectLayer(int);
    int layer() const;

    EditorContext& ctx;
    LiveData& live;
    std::unique_ptr<ParamControl> layerBlend;
    juce::OwnedArray<ParamControl> engineControls;
    juce::OwnedArray<ParamControl> neuralControls;
    juce::TextButton addRoute{"Add route"};

    juce::Rectangle<int> layersArea, layerArea, displayArea, modArea;
    std::vector<juce::Rectangle<int>> layerRows;
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
    void exportTheme();
    void importTheme();

    EditorContext& ctx;
    Segmented themeChoice, density, gesture, valueDisplay, macrosPerPage;
    juce::ComboBox accent;
    juce::Slider scale;
    juce::OwnedArray<juce::ToggleButton> panelToggles;
    juce::TextButton useForNew{"Use for new instances"}, exportButton{"Export theme file..."},
        importButton{"Import theme file..."};
    std::unique_ptr<juce::FileChooser> chooser;
    juce::String statusText;

    juce::Rectangle<int> settingsArea, previewArea;
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
