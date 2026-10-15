#pragma once
#include "ConsoleWidgets.h"
#include "ElementViews.h"
#include "PresetStore.h"
#include "TensorProcessor.h"

#include <nlohmann/json.hpp>

namespace alfa::tensorhost
{
class Editor final : public juce::AudioProcessorEditor, private juce::Timer, private juce::ListBoxModel
{
public:
    explicit Editor(Processor&);
    ~Editor() override;
    void paint(juce::Graphics&) override;
    void paintOverChildren(juce::Graphics&) override;
    void resized() override;
    void reportDeviceStatus(const juce::String& text, bool error) { showStatus(text, error); }
    void snapshotView(int index)
    {
        chooseView(index);
        timerCallback();
    }

private:
    struct Snapshot
    {
        juce::String document;
        std::array<float, 12> controls{};
        bool valid = false;
    };
    Processor& processor_;
    alfa::ui::AlfaLookAndFeel look_;
    KeyboardView keyboard_;
    LiveScope scope_;
    StereoMeter meters_;
    ExplorationPad exploration_;
    InfluenceView influence_;
    MatrixView matrix_;
    CapacityView capacity_;
    SoundSpace space_;
    HeadingView heading_;
    std::vector<std::unique_ptr<ElementView>> elements_;
    LayerStateView layerState_;
    juce::Label brand_, version_, status_, telemetry_, paramHeader_, routeHeader_, levelHeader_,
        matrixSelection_;
    // Views by index: 0 Play, 1 Instances (view.structure), 2 Tensor, 3 Settings, 4 Controls.
    std::array<juce::TextButton, 5> tabs_;
    std::array<juce::TextButton, 3> modes_;
    juce::TextButton library_{"Sound library"}, savePreset_{"Save as"}, openPreset_{"Open"}, panic_{"Panic"};
    juce::TextButton previousSound_{"<"}, nextSound_{">"}, newRoute_{"New relationship"};
    juce::TextButton order_{"Learned order"}, expert_{"Expert tools"};
    juce::ComboBox theme_, layerSelect_, xAxis_, yAxis_, tensorSelect_;
    juce::ComboBox accent_, gesture_;
    juce::Label precision_, resources_, hostPrecision_;
    juce::ToggleButton bypass_{"Bypass"}, showKeyboard_{"Keyboard"};
    juce::TextButton a_{"A"}, b_{"B"}, capture_{"Capture"}, pagePrevious_{"<"}, pageNext_{">"};
    juce::TextButton undo_{"Undo"}, redo_{"Redo"};
    std::unique_ptr<ParameterKnob> inputGain_, outputGain_, mix_;
    std::vector<std::unique_ptr<juce::Component>> highControls_;
    juce::Viewport parameterViewport_, routingViewport_, matrixViewport_;
    juce::Component parameterContent_, routingContent_;
    std::vector<std::unique_ptr<NumericRow>> parameterRows_, routeRows_;
    // Tensor parameter index of each parameter row (-1 for the activation row) and its live mark colour.
    std::vector<int> parameterRowIndex_;
    std::vector<juce::Colour> parameterLiveColour_;
    juce::ListBox primitives_{"Primitive activations", this};
    juce::TextEditor document_, inspection_, tensorValues_;
    juce::TextButton apply_{"Apply JSON"}, exportJSON_{"Export JSON"}, importModel_{"Import runtime model"},
        addRoute_{"Add relationship"}, compose_{"Compose fragments"};
    juce::Component::SafePointer<juce::DialogWindow> compositionDialog_;
    juce::Component::SafePointer<juce::DialogWindow> libraryDialog_;
    juce::TextButton refresh_{"Refresh library"}, folder_{"Library folder"}, exportTheme_{"Export theme"};
    juce::TextButton importTheme_{"Import theme"};
    juce::ComboBox sourceKind_, routeDestination_, routeWeight_, routeWeightsTensor_, mappingLayer_;
    juce::TextEditor sourceElement_, matrixValue_;
    juce::TextButton rawPrevious_{"Previous dimensions"}, rawNext_{"Next dimensions"};
    juce::TextButton setMatrix_{"Set coefficient"}, removeRoute_{"Remove last relationship"};
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment_;
    std::unique_ptr<juce::FileChooser> chooser_;
    std::unique_ptr<juce::AlertWindow> saveDialog_;
    std::unique_ptr<PresetStore> store_;
    nlohmann::json sound_;
    std::array<Snapshot, 2> snapshots_;
    uint64_t revision_ = 0;
    uint64_t telemetryMark_ = 0;
    bool telemetryFresh() const;
    int view_ = 0, selectedPrimitive_ = 0, highLayer_ = 0, highPage_ = 0, selectedSnapshot_ = 0, rawPage_ = 0;
    int focusedControl_ = 0, matrixSource_ = 0, matrixDestination_ = 0;
    bool refreshing_ = false, routeFormOpen_ = false, expertOpen_ = false, indexOrder_ = false,
         ranked_ = false;
    std::vector<std::pair<juce::Rectangle<int>, juce::String>> settingsLabels_;
    juce::Rectangle<int> workspace_, highArea_, influenceArea_, xyArea_, scopeArea_, primitiveArea_,
        parameterArea_, routingArea_, layerArea_, tensorArea_, selectionArea_, capacityArea_, settingsArea_,
        footer_, elementsArea_, spaceArea_, layerStateArea_, headingArea_, modesArea_, snapshotArea_,
        presetArea_, brandArea_;
    // Every primitive of the sound, grouped as the element views group them; the heading and the knob
    // fingerprints report per group.
    struct Group
    {
        juce::String title, primitive;
        std::vector<ElementView::Instance> members;
        std::vector<juce::String> names;
    };
    std::vector<Group> groups_;
    std::vector<std::array<double, 2>> parameterRanges_;
    // Control index of each knob on the current page.
    std::vector<size_t> highIndices_;
    // Recent movement of the projected layer in control units, for the heading.
    std::vector<double> lastU_, motion_;
    double motionTime_ = -1e9;
    void updateHeading(const std::vector<double>& parameters);
    void paintFingerprints(juce::Graphics&);
    void timerCallback() override;
    int getNumRows() override;
    void paintListBoxItem(int, juce::Graphics&, int, int, bool) override;
    void selectedRowsChanged(int) override;
    void chooseView(int);
    void refreshLibrary();
    void openLibrary();
    void savePreset();
    void openPreset();
    void refreshSound();
    void rebuildParameters();
    void rebuildRoutes();
    void rebuildLayers();
    void rebuildHighControls();
    std::vector<size_t> controlOrder() const;
    void focusControl(int index);
    void refreshInfluence();
    void selectMatrixCell(int source, int destination);
    void stepSound(int direction);
    void syncMode();
    void layoutSettings();
    void paintFlow(juce::Graphics&, juce::Rectangle<int>);
    void refreshTensor();
    void rebuildElements();
    void refreshSpace();
    void applySpace(const std::vector<double>&, bool final);
    std::vector<double> currentLayerValues() const;
    void openInstance(size_t instance);
    bool spaceDragging_ = false;
    std::vector<float> liveLayerValues() const;
    bool changeTensor(const juce::String&, size_t, double, juce::String&);
    bool changeDocument(nlohmann::json, juce::String&);
    void captureSnapshot(int);
    void recallSnapshot(int);
    void addRelationship();
    void setCoefficient();
    void showStatus(const juce::String&, bool error = false);
    juce::String parameterLabel(size_t) const;
    juce::String parameterName(size_t) const;
    juce::String controlName(size_t layer, size_t index) const;
    juce::String controlLabel(size_t layer, size_t index) const;
    void renameControl(size_t layer, size_t index, const juce::String& name);
    juce::String parameterUnit(size_t) const;
    std::optional<std::array<double, 3>> parameterRange(size_t) const;
    juce::String sourceLabel(const nlohmann::json&) const;
    void applyTheme();
    uint64_t warningRevision_ = 0; // last host warning shown in the status line
};
} // namespace alfa::tensorhost
