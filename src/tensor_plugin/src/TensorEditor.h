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
    struct BrowserItem
    {
        juce::String id, name, category;
        bool factory = true;
        size_t index = 0;
    };
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
    juce::Label brand_, version_, status_, telemetry_, presetName_, paramHeader_, routeHeader_, levelHeader_;
    std::array<juce::TextButton, 4> tabs_;
    juce::TextButton previous_{"<"}, next_{">"}, savePreset_{"Save as"}, openPreset_{"Open"}, panic_{"Panic"};
    juce::ComboBox presets_, categories_, mode_, theme_, layerSelect_, xAxis_, yAxis_, tensorSelect_;
    juce::ComboBox accent_, gesture_;
    juce::ComboBox tier_;
    juce::Label precision_, resources_, hostPrecision_;
    juce::TextEditor search_;
    juce::ToggleButton bypass_{"Bypass"}, showKeyboard_{"Keyboard"};
    juce::TextButton a_{"A"}, b_{"B"}, capture_{"Capture"}, pagePrevious_{"<"}, pageNext_{">"};
    juce::TextButton undo_{"Undo"}, redo_{"Redo"};
    std::unique_ptr<ParameterKnob> inputGain_, outputGain_, mix_;
    std::vector<std::unique_ptr<juce::Component>> highControls_;
    juce::Viewport parameterViewport_, routingViewport_, layerViewport_;
    juce::Component parameterContent_, routingContent_, layerContent_;
    std::vector<std::unique_ptr<NumericRow>> parameterRows_, routeRows_, layerRows_;
    juce::ListBox primitives_{"Primitive activations", this};
    juce::TextEditor document_, inspection_, tensorValues_;
    juce::TextButton apply_{"Apply JSON"}, exportJSON_{"Export JSON"}, importModel_{"Import runtime model"},
        addRoute_{"Add relationship"};
    juce::TextButton refresh_{"Refresh library"}, folder_{"Library folder"}, exportTheme_{"Export theme"};
    juce::TextButton importTheme_{"Import theme"};
    juce::ComboBox sourceKind_, routeDestination_, routeWeight_, routeWeightsTensor_, mappingLayer_,
        mappingSource_, mappingDestination_;
    juce::TextEditor sourceElement_, matrixValue_;
    juce::TextButton rawPrevious_{"Previous dimensions"}, rawNext_{"Next dimensions"};
    juce::TextButton setMatrix_{"Set coefficient"}, removeRoute_{"Remove last relationship"};
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> modeAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment_;
    std::unique_ptr<juce::FileChooser> chooser_;
    std::unique_ptr<juce::AlertWindow> saveDialog_;
    std::unique_ptr<PresetStore> store_;
    std::vector<BrowserItem> browser_;
    std::vector<size_t> filtered_;
    nlohmann::json sound_;
    std::array<Snapshot, 2> snapshots_;
    uint64_t revision_ = 0;
    int view_ = 0, selectedPrimitive_ = 0, highLayer_ = 0, highPage_ = 0, selectedSnapshot_ = 0, rawPage_ = 0;
    bool refreshing_ = false;
    juce::Rectangle<int> workspace_, highArea_, xyArea_, scopeArea_, primitiveArea_, parameterArea_,
        routingArea_, layerArea_, tensorArea_, settingsArea_;
    void timerCallback() override;
    int getNumRows() override;
    void paintListBoxItem(int, juce::Graphics&, int, int, bool) override;
    void selectedRowsChanged(int) override;
    void chooseView(int);
    void refreshLibrary();
    void filterLibrary();
    void choosePreset(size_t);
    void stepPreset(int);
    void savePreset();
    void openPreset();
    void refreshSound();
    void rebuildParameters();
    void rebuildRoutes();
    void rebuildLayers();
    void rebuildHighControls();
    void refreshTensor();
    bool changeTensor(const juce::String&, size_t, double, juce::String&);
    bool changeDocument(nlohmann::json, juce::String&);
    void captureSnapshot(int);
    void recallSnapshot(int);
    void addRelationship();
    void setCoefficient();
    void showStatus(const juce::String&, bool error = false);
    juce::String parameterLabel(size_t) const;
    juce::String parameterUnit(size_t) const;
    std::optional<std::array<double, 3>> parameterRange(size_t) const;
    juce::String sourceLabel(const nlohmann::json&) const;
    void applyTheme();
};
} // namespace alfa::tensorhost
