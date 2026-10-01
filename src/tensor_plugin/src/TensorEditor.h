#pragma once
#include "ConsoleWidgets.h"
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
    juce::MidiKeyboardComponent keyboard_;
    LiveScope scope_;
    StereoMeter meters_;
    ExplorationPad exploration_;
    InfluenceView influence_;
    MatrixView matrix_;
    CapacityView capacity_;
    juce::Label brand_, version_, status_, telemetry_, paramHeader_, routeHeader_, levelHeader_,
        matrixSelection_;
    std::array<juce::TextButton, 4> tabs_;
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
    int view_ = 0, selectedPrimitive_ = 0, highLayer_ = 0, highPage_ = 0, selectedSnapshot_ = 0, rawPage_ = 0;
    int focusedControl_ = 0, matrixSource_ = 0, matrixDestination_ = 0;
    bool refreshing_ = false, routeFormOpen_ = false, expertOpen_ = false, indexOrder_ = false,
         ranked_ = false;
    std::vector<std::pair<juce::Rectangle<int>, juce::String>> settingsLabels_;
    juce::Rectangle<int> workspace_, highArea_, influenceArea_, xyArea_, scopeArea_, primitiveArea_,
        parameterArea_, routingArea_, layerArea_, tensorArea_, selectionArea_, capacityArea_, settingsArea_,
        footer_;
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
    void focusControl(int index);
    void refreshInfluence();
    void selectMatrixCell(int source, int destination);
    void stepSound(int direction);
    void syncMode();
    void layoutSettings();
    void paintFlow(juce::Graphics&, juce::Rectangle<int>);
    void refreshTensor();
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
};
} // namespace alfa::tensorhost
