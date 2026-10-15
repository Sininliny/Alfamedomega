#include "TensorEditor.h"

#include "CompositionPanel.h"
#include "FactoryPrograms.h"
#include "SoundLibraryPanel.h"
#include "alfa/tensor/Universe.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

namespace alfa::tensorhost
{
namespace
{
const std::array<juce::String, 12> snapshotIds{"x0", "x1", "x2",         "x3",          "x4",  "x5",
                                               "x6", "x7", "input_gain", "output_gain", "mix", "mode"};
std::string string(const juce::String& s)
{
    return s.toStdString();
}
juce::String js(const std::string& s)
{
    return juce::String(s);
}
class LayerKnob final : public juce::Component
{
public:
    LayerKnob(Processor& p, size_t layer, size_t index, double value, double minimum, double maximum)
        : processor(p)
        , layer_(layer)
        , index_(index)
    {
        label.setText("X[" + juce::String(static_cast<int>(layer)) + "," +
                          juce::String(static_cast<int>(index)) + "]",
                      juce::dontSendNotification);
        label.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(label);
        alfa::ui::visuals::rotaryGesture(slider,
                                         p.uiSettings().getProperty("knob_gesture", "vertical").toString());
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 82, 16);
        slider.textFromValueFunction = [](double v)
        {
            return compactText(v);
        };
        slider.onDragStart = [this]
        {
            if (onTouched)
                onTouched();
        };
        slider.setRange(minimum, maximum, 0);
        slider.setValue(value, juce::dontSendNotification);
        addAndMakeVisible(slider);
        slider.onDragEnd = [this]
        {
            juce::String error;
            processor.editLayerValue(layer_, index_, slider.getValue(), error);
            slider.setTooltip(error);
        };
        slider.onValueChange = [this]
        {
            if (!slider.isMouseButtonDown())
            {
                juce::String error;
                processor.editLayerValue(layer_, index_, slider.getValue(), error);
                slider.setTooltip(error);
            }
        };
    }
    void resized() override
    {
        // Name under the knob, value as quiet text between them.
        auto r = getLocalBounds();
        label.setBounds(r.removeFromBottom(18));
        slider.setBounds(r);
    }
    // Built before it joins the editor, so it takes the editor's look (value text, fonts) only now.
    void parentHierarchyChanged() override { sendLookAndFeelChange(); }
    void mouseDown(const juce::MouseEvent&) override
    {
        if (onTouched)
            onTouched();
    }
    std::function<void()> onTouched;
    juce::Label& title() { return label; }

private:
    Processor& processor;
    size_t layer_, index_;
    juce::Label label;
    juce::Slider slider;
};
} // namespace
Editor::Editor(Processor& p)
    : AudioProcessorEditor(p)
    , processor_(p)
    , keyboard_(p.keyboard, juce::MidiKeyboardComponent::horizontalKeyboard)
    , scope_(p, look_)
    , meters_(p, look_)
    , exploration_(p, look_)
    , influence_(look_)
    , matrix_(look_)
    , capacity_(look_)
    , space_(look_)
    , layerState_(look_)
{
    setLookAndFeel(&look_);
    setResizable(true, true);
    setResizeLimits(1040, 700, 1920, 1200);
    const auto saved = p.uiSettings();
    view_ = juce::jlimit(0, 4, static_cast<int>(saved.getProperty("view", 0)));
    expertOpen_ = static_cast<bool>(saved.getProperty("expert_tools", false));
    setSize(static_cast<int>(saved.getProperty("width", 1280)),
            static_cast<int>(saved.getProperty("height", 800)));
    brand_.setText("ALFAMEDOMEGA", juce::dontSendNotification);
    brand_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(19, true));
    version_.setText("AUDIO TENSOR SPACE", juce::dontSendNotification);
    version_.setFont(alfa::ui::AlfaLookAndFeel::monoFont(10));
    for (auto* c : std::initializer_list<juce::Component*>{
             &brand_, &version_, &status_, &telemetry_, &library_, &previousSound_, &nextSound_, &savePreset_,
             &openPreset_, &panic_, &bypass_, &a_, &b_, &capture_, &undo_, &redo_, &keyboard_, &meters_})
        addAndMakeVisible(c);
    // Play is the calm surface; Controls, Instances and Tensor open the same sound deeper, level by level.
    // Settings sits apart at the right. Indices stay stable for saved state and snapshots.
    const std::array<juce::String, 5> views{"Play", "Instances", "Tensor", "Settings", "Controls"};
    const std::array<juce::String, 5> viewIds{"view.play", "view.structure", "view.tensor", "view.settings",
                                              "view.controls"};
    const std::array<juce::String, 5> viewTips{
        "Play the sound: every level drawn live, the first controls, XY and output",
        "Instances by primitive family, their parameters with live modulation, and their relationships",
        "Control matrices as heat maps, capacity, raw tensors and the compiled report",
        "Interface, library and the loaded sound's resources; never changes the sound",
        "Every X control layer: knobs, what the focused control reaches, and every layer's live values"};
    for (size_t i = 0; i < tabs_.size(); ++i)
    {
        addAndMakeVisible(tabs_[i]);
        tabs_[i].setButtonText(views[i]);
        tabs_[i].setComponentID(viewIds[i]);
        tabs_[i].setTooltip(viewTips[i]);
        tabs_[i].setClickingTogglesState(true);
        tabs_[i].onClick = [this, i]
        {
            chooseView(static_cast<int>(i));
        };
    }
    // Input mode changes which native inputs execute, so it stays visible on every view.
    const std::array<juce::String, 3> modeNames{"Instrument", "FX", "Hybrid"};
    const std::array<juce::String, 3> modeTips{"MIDI input only", "Audio input only",
                                               "Audio and MIDI inputs"};
    for (size_t i = 0; i < modes_.size(); ++i)
    {
        addAndMakeVisible(modes_[i]);
        modes_[i].setButtonText(modeNames[i]);
        modes_[i].setTooltip(modeTips[i] + ". The same tensor sound executes in every mode.");
        modes_[i].setComponentID("mode." + modeNames[i].toLowerCase());
        modes_[i].setClickingTogglesState(true);
        modes_[i].setRadioGroupId(4101);
        modes_[i].setConnectedEdges((i > 0 ? juce::Button::ConnectedOnLeft : 0) |
                                    (i + 1 < modes_.size() ? juce::Button::ConnectedOnRight : 0));
        modes_[i].onClick = [this, i]
        {
            if (modes_[i].getToggleState())
                processor_.setParameterValue("mode", static_cast<float>(i));
        };
    }
    library_.setComponentID("header.library");
    panic_.setComponentID("header.panic");
    previousSound_.setTooltip("Previous sound in the library");
    nextSound_.setTooltip("Next sound in the library");
    previousSound_.onClick = [this]
    {
        stepSound(-1);
    };
    nextSound_.onClick = [this]
    {
        stepSound(1);
    };
    bypassAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        p.parameters(), "bypass", bypass_);
    panic_.onClick = [this]
    {
        processor_.requestPanic();
        showStatus("All notes released");
    };
    undo_.onClick = [this]
    {
        if (processor_.undoManager().undo())
            refreshSound();
    };
    redo_.onClick = [this]
    {
        if (processor_.undoManager().redo())
            refreshSound();
    };
    library_.onClick = [this]
    {
        openLibrary();
    };
    library_.setTooltip("Current sound. Click to browse the library; browsing does not change the sound");
    openPreset_.onClick = [this]
    {
        openPreset();
    };
    savePreset_.onClick = [this]
    {
        savePreset();
    };
    for (int i = 0; i < 8; ++i)
    {
        const auto id = "x" + juce::String(i);
        xAxis_.addItem(id.toUpperCase(), i + 1);
        yAxis_.addItem(id.toUpperCase(), i + 1);
    }
    xAxis_.setSelectedId(static_cast<int>(saved.getProperty("xy_x", 0)) + 1);
    yAxis_.setSelectedId(static_cast<int>(saved.getProperty("xy_y", 1)) + 1);
    xAxis_.onChange = [this]
    {
        exploration_.xId = "x" + juce::String(xAxis_.getSelectedId() - 1);
        exploration_.xName = xAxis_.getText();
        processor_.uiSettings().setProperty("xy_x", xAxis_.getSelectedId() - 1, nullptr);
        exploration_.repaint();
    };
    yAxis_.onChange = [this]
    {
        exploration_.yId = "x" + juce::String(yAxis_.getSelectedId() - 1);
        exploration_.yName = yAxis_.getText();
        processor_.uiSettings().setProperty("xy_y", yAxis_.getSelectedId() - 1, nullptr);
        exploration_.repaint();
    };
    exploration_.xId = "x" + juce::String(xAxis_.getSelectedId() - 1);
    exploration_.xName = xAxis_.getText();
    exploration_.yId = "x" + juce::String(yAxis_.getSelectedId() - 1);
    exploration_.yName = yAxis_.getText();
    a_.onClick = [this]
    {
        if (juce::ModifierKeys::getCurrentModifiers().isShiftDown())
            captureSnapshot(0);
        else
            recallSnapshot(0);
    };
    b_.onClick = [this]
    {
        if (juce::ModifierKeys::getCurrentModifiers().isShiftDown())
            captureSnapshot(1);
        else
            recallSnapshot(1);
    };
    a_.setTooltip("Recall A; Shift-click to capture current sound");
    b_.setTooltip("Recall B; Shift-click to capture current sound");
    capture_.onClick = [this]
    {
        captureSnapshot(selectedSnapshot_);
    };
    inputGain_ = std::make_unique<ParameterKnob>(p, "input_gain", "Input");
    outputGain_ = std::make_unique<ParameterKnob>(p, "output_gain", "Output");
    mix_ = std::make_unique<ParameterKnob>(p, "mix", "Mix", true);
    for (auto* c : {inputGain_.get(), outputGain_.get(), mix_.get()})
        addAndMakeVisible(c);
    parameterViewport_.setViewedComponent(&parameterContent_, false);
    routingViewport_.setViewedComponent(&routingContent_, false);
    matrixViewport_.setViewedComponent(&matrix_, false);
    primitives_.setRowHeight(58);
    matrixViewport_.setScrollBarsShown(true, false);
    matrix_.onSelect = [this](int source, int destination)
    {
        selectMatrixCell(source, destination);
    };
    // The matrix is the Tensor view; raw JSON, model import and composition open on request.
    expert_.setComponentID("view.expert");
    expert_.setClickingTogglesState(true);
    expert_.setToggleState(expertOpen_, juce::dontSendNotification);
    expert_.setTooltip("Raw tensors as JSON, runtime model import, composition and the compiled report");
    expert_.onClick = [this]
    {
        expertOpen_ = expert_.getToggleState();
        processor_.uiSettings().setProperty("expert_tools", expertOpen_, nullptr);
        chooseView(view_);
    };
    matrixSelection_.setJustificationType(juce::Justification::topLeft);
    newRoute_.setClickingTogglesState(true);
    newRoute_.setTooltip("Add or remove an explicit routing or modulation relationship");
    newRoute_.onClick = [this]
    {
        routeFormOpen_ = newRoute_.getToggleState();
        chooseView(view_);
    };
    matrixSelection_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13.5f));
    space_.onMove = [this](const std::vector<double>& values, bool final)
    {
        applySpace(values, final);
    };
    layerState_.onSelect = [this](int layer, int index)
    {
        highLayer_ = layer;
        layerSelect_.setSelectedId(layer + 1, juce::dontSendNotification);
        const auto order = controlOrder();
        const auto position =
            std::find(order.begin(), order.end(), static_cast<size_t>(index)) - order.begin();
        highPage_ = static_cast<int>(position) / 8;
        rebuildHighControls();
        refreshSpace();
        focusControl(index);
    };
    for (auto* c : std::initializer_list<juce::Component*>{&scope_,
                                                           &space_,
                                                           &layerState_,
                                                           &exploration_,
                                                           &xAxis_,
                                                           &yAxis_,
                                                           &layerSelect_,
                                                           &order_,
                                                           &pagePrevious_,
                                                           &pageNext_,
                                                           &primitives_,
                                                           &parameterViewport_,
                                                           &routingViewport_,
                                                           &matrixViewport_,
                                                           &capacity_,
                                                           &expert_,
                                                           &influence_,
                                                           &newRoute_,
                                                           &matrixSelection_,
                                                           &paramHeader_,
                                                           &routeHeader_,
                                                           &levelHeader_,
                                                           &document_,
                                                           &inspection_,
                                                           &tensorValues_,
                                                           &tensorSelect_,
                                                           &apply_,
                                                           &exportJSON_,
                                                           &importModel_,
                                                           &compose_,
                                                           &addRoute_,
                                                           &removeRoute_,
                                                           &sourceKind_,
                                                           &sourceElement_,
                                                           &routeDestination_,
                                                           &routeWeight_,
                                                           &routeWeightsTensor_,
                                                           &mappingLayer_,
                                                           &matrixValue_,
                                                           &setMatrix_,
                                                           &rawPrevious_,
                                                           &rawNext_,
                                                           &theme_,
                                                           &refresh_,
                                                           &folder_,
                                                           &exportTheme_,
                                                           &showKeyboard_,
                                                           &accent_,
                                                           &gesture_,
                                                           &precision_,
                                                           &resources_,
                                                           &hostPrecision_,
                                                           &importTheme_})
        addChildComponent(c);
    order_.setClickingTogglesState(true);
    order_.setTooltip("Show controls by learned importance, most important first, or by index");
    order_.onClick = [this]
    {
        indexOrder_ = !order_.getToggleState();
        highPage_ = 0;
        rebuildHighControls();
    };
    layerSelect_.onChange = [this]
    {
        highLayer_ = juce::jmax(0, layerSelect_.getSelectedId() - 1);
        highPage_ = 0;
        rebuildHighControls();
        refreshSpace();
    };
    compose_.onClick = [this]
    {
        if (compositionDialog_)
        {
            compositionDialog_->toFront(true);
            return;
        }
        auto safe = juce::Component::SafePointer<Editor>(this);
        const auto startingDocument = processor_.document();
        juce::DialogWindow::LaunchOptions options;
        options.dialogTitle = "Tensor composition";
        options.dialogBackgroundColour = look_.theme().bg;
        options.escapeKeyTriggersCloseButton = true;
        options.useNativeTitleBar = true;
        options.resizable = false;
        options.componentToCentreAround = this;
        options.content.setOwned(new CompositionPanel(
            startingDocument, processor_.getSampleRate() > 0 ? processor_.getSampleRate() : 48000,
            [safe, expected = startingDocument](const juce::String& document, juce::String& error) mutable
            {
                if (!safe)
                {
                    error = "Editor closed";
                    return false;
                }
                if (safe->processor_.document() != expected)
                {
                    error = "Sound changed while composition was open. Close and reopen composition to "
                            "retain those edits.";
                    return false;
                }
                if (!safe->processor_.editConfigurationJSON(document, error))
                    return false;
                expected = safe->processor_.document();
                safe->refreshSound();
                return true;
            },
            &look_.theme()));
        compositionDialog_ = options.launchAsync();
    };
    pagePrevious_.onClick = [this]
    {
        highPage_ = juce::jmax(0, highPage_ - 1);
        rebuildHighControls();
    };
    pageNext_.onClick = [this]
    {
        ++highPage_;
        rebuildHighControls();
    };
    rawPrevious_.onClick = [this]
    {
        rawPage_ = juce::jmax(0, rawPage_ - 1);
        rebuildParameters();
    };
    rawNext_.onClick = [this]
    {
        ++rawPage_;
        rebuildParameters();
    };
    document_.setMultiLine(true);
    document_.setReturnKeyStartsNewLine(true);
    document_.setFont(alfa::ui::AlfaLookAndFeel::monoFont(12));
    inspection_.setMultiLine(true);
    inspection_.setReadOnly(true);
    inspection_.setFont(alfa::ui::AlfaLookAndFeel::monoFont(12));
    tensorValues_.setMultiLine(true);
    tensorValues_.setReadOnly(true);
    tensorValues_.setFont(alfa::ui::AlfaLookAndFeel::monoFont(12));
    tensorSelect_.addItemList({"Routing tensor", "Parameter tensor", "Activation tensor", "Modulation tensor",
                               "Assets and persistent state"},
                              1);
    tensorSelect_.setSelectedId(1);
    tensorSelect_.onChange = [this]
    {
        refreshTensor();
    };
    sourceKind_.addItemList(
        {"audio", "midi", "gesture", "user_preference", "latent_sound", "context", "state", "output_history"},
        1);
    sourceKind_.setSelectedId(1);
    sourceElement_.setText("0");
    sourceElement_.setInputRestrictions(12, "0123456789");
    routeWeightsTensor_.addItemList({"routing", "modulation"}, 1);
    routeWeightsTensor_.setSelectedId(1);
    routeWeightsTensor_.onChange = [this]
    {
        rebuildRoutes();
    };
    addRoute_.onClick = [this]
    {
        addRelationship();
    };
    removeRoute_.onClick = [this]
    {
        try
        {
            auto updated = sound_;
            auto& routes = updated["projections"];
            if (routes.is_array() && !routes.empty())
            {
                routes.erase(routes.end() - 1);
                juce::String error;
                changeDocument(std::move(updated), error);
                showStatus(error.isEmpty() ? "Relationship removed" : error, error.isNotEmpty());
            }
        }
        catch (const std::exception& e)
        {
            showStatus(e.what(), true);
        }
    };
    mappingLayer_.onChange = [this]
    {
        rebuildLayers();
    };
    setMatrix_.onClick = [this]
    {
        setCoefficient();
    };
    apply_.onClick = [this]
    {
        juce::String error;
        if (processor_.loadJSON(document_.getText(), error))
        {
            showStatus("Tensor configuration applied");
            refreshSound();
        }
        else
            showStatus(error, true);
    };
    exportJSON_.onClick = [this]
    {
        chooser_ = std::make_unique<juce::FileChooser>("Export sound tensor", juce::File{}, "*.json");
        auto safe = juce::Component::SafePointer<Editor>(this);
        chooser_->launchAsync(
            juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles |
                juce::FileBrowserComponent::warnAboutOverwriting,
            [safe](const juce::FileChooser& c)
            {
                if (safe && c.getResult() != juce::File{})
                    safe->showStatus(c.getResult().replaceWithText(safe->processor_.document())
                                         ? "Sound exported"
                                         : "Export failed");
            });
    };
    importModel_.onClick = [this]
    {
        chooser_ =
            std::make_unique<juce::FileChooser>("Import runtime compiler model", juce::File{}, "*.json");
        auto safe = juce::Component::SafePointer<Editor>(this);
        chooser_->launchAsync(
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [safe](const juce::FileChooser& c)
            {
                if (!safe || !c.getResult().existsAsFile())
                    return;
                try
                {
                    auto data = nlohmann::json::parse(safe->processor_.document().toStdString());
                    data["runtimeModel"] =
                        nlohmann::json::parse(c.getResult().loadFileAsString().toStdString());
                    juce::String error;
                    safe->showStatus(safe->changeDocument(std::move(data), error) ? "Runtime model imported"
                                                                                  : error,
                                     error.isNotEmpty());
                }
                catch (const std::exception& e)
                {
                    safe->showStatus(e.what(), true);
                }
            });
    };
    theme_.addItemList(alfa::ui::Theme::builtInNames(), 1);
    const auto ids = alfa::ui::Theme::builtInIds();
    theme_.setSelectedId(juce::jmax(0, ids.indexOf(saved.getProperty("theme", "graphite").toString())) + 1,
                         juce::dontSendNotification);
    theme_.onChange = [this]
    {
        processor_.uiSettings().removeProperty("custom_theme", nullptr);
        applyTheme();
    };
    accent_.addItemList({"Theme default", "Amber", "Teal", "Violet", "Lime", "Rose"}, 1);
    const juce::StringArray accents{"", "E9A23B", "3BB8A6", "A78BE8", "A6CE3F", "E8829E"};
    accent_.setSelectedId(juce::jmax(0, accents.indexOf(saved.getProperty("accent", "").toString())) + 1,
                          juce::dontSendNotification);
    accent_.onChange = [this, accents]
    {
        processor_.uiSettings().setProperty(
            "accent", accents[juce::jlimit(0, accents.size() - 1, accent_.getSelectedId() - 1)], nullptr);
        applyTheme();
    };
    precision_.setFont(alfa::ui::AlfaLookAndFeel::monoFont(12));
    resources_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
    hostPrecision_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
    resources_.setText("Precision and capacity come from the loaded sound.\nThe Tensor view's report shows "
                       "used / budget for each resource.",
                       juce::dontSendNotification);
    hostPrecision_.setText("Highest 8 host automation ports use float32.\nManual tensor and deeper-layer "
                           "editing preserve double values.",
                           juce::dontSendNotification);
    gesture_.addItemList({"Vertical drag", "Horizontal drag", "Circular drag"}, 1);
    const juce::StringArray gestures{"vertical", "horizontal", "circular"};
    gesture_.setSelectedId(
        juce::jmax(0, gestures.indexOf(saved.getProperty("knob_gesture", "vertical").toString())) + 1,
        juce::dontSendNotification);
    gesture_.onChange = [this, gestures]
    {
        const auto value = gestures[juce::jlimit(0, gestures.size() - 1, gesture_.getSelectedId() - 1)];
        processor_.uiSettings().setProperty("knob_gesture", value, nullptr);
        for (auto* knob : {inputGain_.get(), outputGain_.get(), mix_.get()})
            if (knob)
                alfa::ui::visuals::rotaryGesture(knob->control(), value);
        rebuildHighControls();
    };
    importTheme_.onClick = [this]
    {
        chooser_ = std::make_unique<juce::FileChooser>("Import interface theme", juce::File{}, "*.json");
        auto safe = juce::Component::SafePointer<Editor>(this);
        chooser_->launchAsync(
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [safe](const juce::FileChooser& c)
            {
                if (safe && c.getResult().existsAsFile())
                {
                    const auto text = c.getResult().loadFileAsString();
                    const auto parsed = juce::JSON::parse(text);
                    if (!parsed.isObject())
                    {
                        safe->showStatus("Theme must be a JSON object", true);
                        return;
                    }
                    safe->processor_.uiSettings().setProperty("custom_theme", text, nullptr);
                    safe->applyTheme();
                    safe->showStatus("Imported " + safe->look_.theme().name);
                }
            });
    };
    refresh_.onClick = [this]
    {
        refreshLibrary();
    };
    folder_.onClick = [this]
    {
        chooser_ = std::make_unique<juce::FileChooser>("Select local preset library", store_->directory());
        auto safe = juce::Component::SafePointer<Editor>(this);
        chooser_->launchAsync(
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
            [safe](const juce::FileChooser& c)
            {
                if (safe && c.getResult().isDirectory())
                {
                    safe->processor_.uiSettings().setProperty("preset_folder",
                                                              c.getResult().getFullPathName(), nullptr);
                    safe->store_ = std::make_unique<PresetStore>(c.getResult());
                    safe->refreshLibrary();
                }
            });
    };
    exportTheme_.onClick = [this]
    {
        chooser_ = std::make_unique<juce::FileChooser>("Export interface theme", juce::File{}, "*.json");
        auto safe = juce::Component::SafePointer<Editor>(this);
        chooser_->launchAsync(
            juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles |
                juce::FileBrowserComponent::warnAboutOverwriting,
            [safe](const juce::FileChooser& c)
            {
                if (safe && c.getResult() != juce::File{})
                    safe->showStatus(c.getResult().replaceWithText(safe->look_.theme().toJson())
                                         ? "Theme exported"
                                         : "Theme export failed");
            });
    };
    showKeyboard_.setToggleState(static_cast<bool>(saved.getProperty("keyboard", true)),
                                 juce::dontSendNotification);
    showKeyboard_.onClick = [this]
    {
        processor_.uiSettings().setProperty("keyboard", showKeyboard_.getToggleState(), nullptr);
        resized();
    };
    store_ = std::make_unique<PresetStore>(juce::File(
        saved.getProperty("preset_folder", PresetStore::defaultDirectory().getFullPathName()).toString()));
    applyTheme();
    refreshLibrary();
    refreshSound();
    captureSnapshot(0);
    chooseView(view_);
    startTimerHz(30);
}
Editor::~Editor()
{
    if (libraryDialog_)
        delete libraryDialog_.getComponent();
    if (compositionDialog_)
        delete compositionDialog_.getComponent();
    stopTimer();
    parameterViewport_.setViewedComponent(nullptr, false);
    routingViewport_.setViewedComponent(nullptr, false);
    matrixViewport_.setViewedComponent(nullptr, false);
    setLookAndFeel(nullptr);
}
void Editor::paint(juce::Graphics& g)
{
    const auto& t = look_.theme();
    g.fillAll(t.bg);
    g.setColour(t.line);
    g.drawHorizontalLine(63, 18.0f, static_cast<float>(getWidth() - 18));
    g.drawHorizontalLine(115, 18.0f, static_cast<float>(getWidth() - 18));
    if (view_ == 0)
    {
        surface(g, t, spaceArea_, "Sound space", "the two directions that change this sound most");
        surface(g, t, highArea_, "Controls");
    }
    if (view_ == 4)
    {
        surface(g, t, highArea_, "Controls", "X / many-to-many");
        surface(g, t, influenceArea_, "Influence", "what the focused control reaches");
        surface(g, t, layerStateArea_, "Layer state", "every layer, live; click a bar");
        surface(g, t, xyArea_, "XY", "two explicit axes");
    }
    if (view_ == 1)
    {
        surface(g, t, primitiveArea_, "Instances", "activation");
        surface(g, t, parameterArea_, "Parameters");
        surface(g, t, routingArea_, "Relationships", "routing / modulation");
    }
    if (view_ == 2)
    {
        surface(g, t, layerArea_, "Control matrix", "columns drive rows");
        if (expertOpen_)
            surface(g, t, tensorArea_, "Expert tools", "raw tensors and compiled report");
        else
        {
            surface(g, t, selectionArea_, "Selected coefficient");
            surface(g, t, capacityArea_, "Capacity", "used / budget");
        }
    }
    if (view_ == 3)
    {
        surface(g, t, settingsArea_, "Settings", "separate from the sound");
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12, true));
        for (const auto& [area, text] : settingsLabels_)
        {
            g.setColour(t.text2);
            g.drawText(text, area, juce::Justification::bottomLeft, true);
        }
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
        g.drawFittedText(
            store_ ? store_->directory().getFullPathName() : juce::String{},
            folder_.getBounds().translated(0, 36).withWidth(refresh_.getRight() - folder_.getX()),
            juce::Justification::topLeft, 2);
    }
    surface(g, t, footer_, "");
    paintFlow(g, footer_.withTrimmedLeft(320).withTrimmedRight(260).reduced(0, 14));
}
void Editor::paintOverChildren(juce::Graphics& g)
{
    // Each level keeps one colour, as a dot on its tab: Play the accent, Controls violet, Instances teal,
    // Tensor indigo. The same hues are used by the identity palette, so the tabs read as a legend.
    const auto& t = look_.theme();
    const std::array<juce::Colour, 5> colours{t.accent, identityColour(t, 0.47f), identityColour(t, 0.66f),
                                              t.text2, identityColour(t, 0.80f)};
    for (size_t i = 0; i < tabs_.size(); ++i)
    {
        if (i == 3)
            continue;
        const auto bounds = tabs_[i].getBounds().toFloat();
        const auto on = tabs_[i].getToggleState();
        g.setColour(on ? t.bg : colours[i]);
        g.fillEllipse(juce::Rectangle<float>(7, 7).withCentre({bounds.getX() + 11, bounds.getCentreY()}));
    }
}
void Editor::paintFlow(juce::Graphics& g, juce::Rectangle<int> area)
{
    // Live summary of the current sound: what comes in, through how much structure, and what goes out.
    const auto& t = look_.theme();
    const auto mode = processor_.mode.load();
    size_t controls = 0, layers = 0, instances = 0, state = 0;
    try
    {
        const auto all = sound_.value("control_layers", nlohmann::json::array());
        layers = all.size();
        for (const auto& layer : all)
            controls += layer.at("values").size();
        for (const auto& instance : sound_.at("instances"))
        {
            ++instances;
            state += instance.value("state_size", size_t(0));
        }
    }
    catch (const std::exception&)
    {
    }
    const juce::String input = mode == 0 ? "MIDI" : mode == 1 ? "Audio" : "Audio + MIDI";
    const juce::String structure =
        juce::String(static_cast<int>(controls)) + " X in " + juce::String(static_cast<int>(layers)) +
        (layers == 1 ? " layer, " : " layers, ") + juce::String(static_cast<int>(instances)) + " instances";
    auto row = area.removeFromTop(area.getHeight() / 2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12.5f, true));
    g.setColour(processor_.activeNotes.load() > 0 ? t.accent : t.text);
    g.drawText(input, row.removeFromLeft(row.getWidth() / 3), juce::Justification::centred, true);
    g.setColour(t.text2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12.5f));
    g.drawText(structure, row, juce::Justification::centredLeft, true);
    // Live readouts as label / value pairs, readable at a glance in a host window.
    const std::array<std::pair<juce::String, juce::String>, 4> stats{
        {{"State", juce::String(static_cast<int>(state))},
         {"Notes", juce::String(processor_.activeNotes.load())},
         {"Rate", compactText(processor_.getSampleRate() / 1000.0) + " kHz"},
         {"Block", juce::String(processor_.blockMilliseconds.load(), 2) + " ms"}}};
    const auto cell = area.getWidth() / static_cast<int>(stats.size());
    for (const auto& [label, value] : stats)
    {
        auto box = area.removeFromLeft(cell);
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
        g.drawText(label.toUpperCase(), box.removeFromLeft(box.getWidth() * 2 / 5),
                   juce::Justification::centredRight, false);
        g.setColour(t.text);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13.5f, true));
        g.drawText(value, box.withTrimmedLeft(8), juce::Justification::centredLeft, true);
    }
}
void Editor::resized()
{
    if (getWidth() < 1)
        return;
    // Header: brand, depths, input mode, bypass, panic and settings on one line.
    auto top = juce::Rectangle<int>(18, 10, getWidth() - 36, 44);
    auto brand = top.removeFromLeft(150);
    brand_.setBounds(brand.removeFromTop(25));
    version_.setBounds(brand);
    for (const auto i : {0, 4, 1, 2})
        tabs_[static_cast<size_t>(i)].setBounds(top.removeFromLeft(100).reduced(2, 5));
    tabs_[3].setBounds(top.removeFromRight(84).reduced(2, 5));
    panic_.setBounds(top.removeFromRight(64).reduced(2, 5));
    bypass_.setBounds(top.removeFromRight(80).reduced(2, 7));
    auto modes = top.removeFromRight(top.getWidth() < 360 ? 222 : 252).reduced(4, 5);
    const auto modeWidth = modes.getWidth() / static_cast<int>(modes_.size());
    for (auto& mode : modes_)
        mode.setBounds(modes.removeFromLeft(modeWidth));
    // Sound bar.
    auto browser = juce::Rectangle<int>(18, 72, getWidth() - 36, 34);
    redo_.setBounds(browser.removeFromRight(52).reduced(2, 0));
    undo_.setBounds(browser.removeFromRight(52).reduced(2, 0));
    browser.removeFromRight(10);
    capture_.setBounds(browser.removeFromRight(76).reduced(2, 0));
    b_.setBounds(browser.removeFromRight(34).reduced(2, 0));
    a_.setBounds(browser.removeFromRight(34).reduced(2, 0));
    browser.removeFromRight(10);
    openPreset_.setBounds(browser.removeFromRight(64).reduced(2, 0));
    savePreset_.setBounds(browser.removeFromRight(80).reduced(2, 0));
    browser.removeFromRight(10);
    previousSound_.setBounds(browser.removeFromLeft(34).reduced(2, 0));
    nextSound_.setBounds(browser.removeFromRight(34).reduced(2, 0));
    library_.setBounds(browser.reduced(2, 0));
    const auto keyboardHeight = showKeyboard_.getToggleState() ? 70 : 0;
    keyboard_.setVisible(keyboardHeight != 0);
    keyboard_.setBounds(18, getHeight() - keyboardHeight - 24, getWidth() - 36, keyboardHeight);
    // All 75 white keys of the MIDI range fill the width.
    keyboard_.setKeyWidth(static_cast<float>(keyboard_.getWidth() - 16) / 75.0f);
    status_.setBounds(20, getHeight() - 22, getWidth() - 40, 20);
    workspace_ = {18, 126, getWidth() - 36, getHeight() - 126 - keyboardHeight - 124};
    footer_ = {18, workspace_.getBottom() + 10, getWidth() - 36, 88};

    // Play: sound elements drawn as they sound (and the output), the sound space beside them, the first
    // controls along the bottom. Controls: the knobs on top, then influence, every layer live and XY.
    auto play = workspace_;
    if (view_ == 4)
    {
        highArea_ = play.removeFromTop(juce::jmin(230, play.getHeight() * 11 / 20));
        play.removeFromTop(12);
        xyArea_ = play.removeFromRight(juce::jlimit(240, 340, play.getWidth() / 4));
        play.removeFromRight(12);
        layerStateArea_ = play.removeFromRight(play.getWidth() * 9 / 20);
        play.removeFromRight(12);
        influenceArea_ = play;
    }
    else
    {
        highArea_ = play.removeFromBottom(juce::jlimit(124, 150, play.getHeight() / 3));
        play.removeFromBottom(12);
        spaceArea_ = play.removeFromRight(juce::jlimit(300, 420, play.getWidth() * 34 / 100));
        play.removeFromRight(12);
        elementsArea_ = play;
        // Elements and the output share a grid, two or three columns wide.
        const auto cells = static_cast<int>(elements_.size()) + 1;
        const auto columns = cells <= 1 ? 1 : cells <= 4 ? 2 : 3;
        const auto rows = (cells + columns - 1) / columns;
        const auto cellWidth = (elementsArea_.getWidth() - (columns - 1) * 12) / columns;
        const auto cellHeight = (elementsArea_.getHeight() - (rows - 1) * 12) / rows;
        for (int i = 0; i < cells; ++i)
        {
            auto cell = juce::Rectangle<int>(elementsArea_.getX() + (i % columns) * (cellWidth + 12),
                                             elementsArea_.getY() + (i / columns) * (cellHeight + 12),
                                             cellWidth, cellHeight);
            // The last cell takes what is left of its row, so a short last row leaves no hole.
            if (i == cells - 1)
                cell.setRight(elementsArea_.getRight());
            if (i < static_cast<int>(elements_.size()))
                elements_[static_cast<size_t>(i)]->setBounds(cell);
            else
                scopeArea_ = cell;
        }
    }
    space_.setBounds(spaceArea_.reduced(12, 10).withTrimmedTop(24));
    layerState_.setBounds(layerStateArea_.reduced(16, 12).withTrimmedTop(26));
    auto high = highArea_.reduced(14, view_ == 4 ? 14 : 8);
    high.removeFromTop(view_ == 4 ? 26 : 22);
    auto highToolbar = view_ == 4 ? high.removeFromTop(28) : highArea_.reduced(14, 6).withHeight(24);
    if (view_ == 4)
        layerSelect_.setBounds(highToolbar.removeFromLeft(160));
    else
        highToolbar.removeFromLeft(90);
    if (ranked_)
        order_.setBounds(highToolbar.removeFromLeft(130).withTrimmedLeft(8));
    pageNext_.setBounds(highToolbar.removeFromRight(30));
    pagePrevious_.setBounds(highToolbar.removeFromRight(30));
    if (view_ != 4)
        highToolbar.removeFromRight(juce::jmin(330, highToolbar.getWidth() / 2));
    levelHeader_.setBounds(highToolbar.reduced(8, 0));
    high.removeFromTop(view_ == 4 ? 6 : 2);
    const auto highWidth = high.getWidth() / 8;
    for (size_t i = 0; i < highControls_.size(); ++i)
        highControls_[i]->setBounds(high.removeFromLeft(highWidth).reduced(3, 0));
    influence_.setBounds(influenceArea_.reduced(16, 12).withTrimmedTop(24));
    auto xp = xyArea_.reduced(14);
    xp.removeFromTop(28);
    auto axes = xp.removeFromTop(28);
    xAxis_.setBounds(axes.removeFromLeft(axes.getWidth() / 2).reduced(2, 0));
    yAxis_.setBounds(axes.reduced(2, 0));
    exploration_.setBounds(xp.withTrimmedTop(6));
    scope_.setBounds(scopeArea_);

    // Structure: instances, the selected instance's parameters, and the relationships into it.
    primitiveArea_ = workspace_;
    parameterArea_ = primitiveArea_;
    primitiveArea_.setWidth(300);
    parameterArea_.removeFromLeft(312);
    routingArea_ = parameterArea_.removeFromRight(juce::jmax(330, parameterArea_.getWidth() * 2 / 5));
    parameterArea_.removeFromRight(12);
    primitives_.setBounds(primitiveArea_.reduced(10).withTrimmedTop(28));
    auto params = parameterArea_.reduced(10).withTrimmedTop(26);
    paramHeader_.setBounds(params.removeFromTop(25));
    auto rawButtons = params.removeFromBottom(32);
    rawPrevious_.setBounds(rawButtons.removeFromLeft(rawButtons.getWidth() / 2).reduced(2));
    rawNext_.setBounds(rawButtons.reduced(2));
    parameterViewport_.setBounds(params);
    parameterContent_.setSize(juce::jmax(1, parameterViewport_.getWidth() - 14),
                              static_cast<int>(parameterRows_.size()) * 46);
    for (size_t i = 0; i < parameterRows_.size(); ++i)
        parameterRows_[i]->setBounds(0, static_cast<int>(i) * 46, parameterContent_.getWidth(), 46);
    auto routes = routingArea_.reduced(10).withTrimmedTop(26);
    routeHeader_.setBounds(routes.removeFromTop(25));
    auto toggle = routes.removeFromBottom(34);
    newRoute_.setBounds(toggle.reduced(2));
    if (routeFormOpen_)
    {
        auto controls = routes.removeFromBottom(164);
        auto routeRow = controls.removeFromTop(30);
        sourceKind_.setBounds(routeRow.removeFromLeft(routeRow.getWidth() * 2 / 3).reduced(2));
        sourceElement_.setBounds(routeRow.reduced(2));
        routeDestination_.setBounds(controls.removeFromTop(30).reduced(2));
        routeRow = controls.removeFromTop(30);
        routeWeightsTensor_.setBounds(routeRow.removeFromLeft(routeRow.getWidth() / 2).reduced(2));
        routeWeight_.setBounds(routeRow.reduced(2));
        addRoute_.setBounds(controls.removeFromTop(34).reduced(2));
        removeRoute_.setBounds(controls.removeFromTop(34).reduced(2));
        routes.removeFromBottom(6);
    }
    routingViewport_.setBounds(routes);
    routingContent_.setSize(juce::jmax(1, routingViewport_.getWidth() - 14),
                            static_cast<int>(routeRows_.size()) * 50);
    for (size_t i = 0; i < routeRows_.size(); ++i)
        routeRows_[i]->setBounds(0, static_cast<int>(i) * 50, routingContent_.getWidth(), 50);

    // Tensor: the control matrix fills the view; the selected coefficient and capacity sit beside it.
    // Expert tools replace that column only when asked for.
    layerArea_ = workspace_;
    tensorArea_ = layerArea_.removeFromRight(expertOpen_ ? workspace_.getWidth() / 2
                                                         : juce::jlimit(300, 380, workspace_.getWidth() / 3));
    layerArea_.removeFromRight(12);
    selectionArea_ = tensorArea_.withHeight(juce::jmin(176, tensorArea_.getHeight() / 2));
    capacityArea_ = tensorArea_.withTrimmedTop(selectionArea_.getHeight() + 12);
    auto layers = layerArea_.reduced(12).withTrimmedTop(28);
    auto layerRow = layers.removeFromTop(30);
    mappingLayer_.setBounds(layerRow.removeFromLeft(juce::jmin(220, layerRow.getWidth() / 2)));
    expert_.setBounds(layerRow.removeFromRight(130));
    layers.removeFromTop(8);
    matrixViewport_.setBounds(layers);
    matrix_.setSize(juce::jmax(1, layers.getWidth() - 12), matrix_.preferredHeight(layers.getHeight()));
    auto selection = selectionArea_.reduced(14).withTrimmedTop(28);
    matrixSelection_.setBounds(selection.removeFromTop(juce::jmax(40, selection.getHeight() - 40)));
    auto edit = selection.removeFromBottom(32);
    setMatrix_.setBounds(edit.removeFromRight(130));
    matrixValue_.setBounds(edit.withTrimmedRight(8));
    capacity_.setBounds(capacityArea_.reduced(14).withTrimmedTop(28));
    auto tensor = tensorArea_.reduced(12).withTrimmedTop(28);
    auto tensorTools = tensor.removeFromTop(30);
    importModel_.setBounds(tensorTools.removeFromRight(150).reduced(2, 0));
    exportJSON_.setBounds(tensorTools.removeFromRight(96).reduced(2, 0));
    apply_.setBounds(tensorTools.removeFromRight(92).reduced(2, 0));
    tensorSelect_.setBounds(tensorTools.reduced(0, 0).withTrimmedRight(4));
    tensorValues_.setBounds(tensor.removeFromTop(tensor.getHeight() / 4).reduced(0, 6));
    compose_.setBounds(tensor.removeFromTop(30).reduced(0, 2));
    document_.setBounds(tensor.removeFromTop(tensor.getHeight() * 3 / 5).reduced(0, 5));
    inspection_.setBounds(tensor.reduced(0, 5));

    settingsArea_ = workspace_;
    layoutSettings();

    const auto bottomY = footer_.getY() + 12;
    if (inputGain_)
        inputGain_->setBounds(34, bottomY - 6, 90, 80);
    if (mix_)
        mix_->setBounds(132, bottomY - 6, 90, 80);
    if (outputGain_)
        outputGain_->setBounds(230, bottomY - 6, 90, 80);
    meters_.setBounds(footer_.getRight() - 132, bottomY - 4, 114, 72);
    telemetry_.setBounds(footer_.getRight() - 250, bottomY + 22, 110, 24);
    processor_.uiSettings().setProperty("width", getWidth(), nullptr);
    processor_.uiSettings().setProperty("height", getHeight(), nullptr);
}
void Editor::layoutSettings()
{
    // Two columns of labelled rows; labels are painted from settingsLabels_ so nothing overlaps.
    settingsLabels_.clear();
    auto area = settingsArea_.reduced(24).withTrimmedTop(30);
    auto left = area.removeFromLeft(juce::jmin(560, area.getWidth() / 2));
    area.removeFromLeft(24);
    auto right = area;
    const auto label = [this](juce::Rectangle<int>& column, const juce::String& text)
    {
        settingsLabels_.push_back({column.removeFromTop(22), text});
        column.removeFromTop(4);
    };
    label(left, "Theme");
    auto row = left.removeFromTop(32);
    theme_.setBounds(row.removeFromLeft(200));
    row.removeFromLeft(10);
    exportTheme_.setBounds(row.removeFromLeft(130));
    row.removeFromLeft(10);
    importTheme_.setBounds(row.removeFromLeft(130));
    left.removeFromTop(16);
    auto pair = left.removeFromTop(22);
    settingsLabels_.push_back({pair.removeFromLeft(210), "Accent colour"});
    settingsLabels_.push_back({pair, "Knob drag"});
    left.removeFromTop(4);
    row = left.removeFromTop(32);
    accent_.setBounds(row.removeFromLeft(200));
    row.removeFromLeft(10);
    gesture_.setBounds(row.removeFromLeft(200));
    left.removeFromTop(16);
    label(left, "On-screen keyboard");
    showKeyboard_.setBounds(left.removeFromTop(32).withWidth(160));
    left.removeFromTop(16);
    label(left, "Sound library folder");
    row = left.removeFromTop(32);
    folder_.setBounds(row.removeFromLeft(200));
    row.removeFromLeft(10);
    refresh_.setBounds(row.removeFromLeft(150));

    label(right, "Sound resources and precision");
    precision_.setBounds(right.removeFromTop(24));
    resources_.setBounds(right.removeFromTop(84));
    hostPrecision_.setBounds(right.removeFromTop(44));
}
void Editor::chooseView(int index)
{
    view_ = juce::jlimit(0, 4, index);
    for (size_t i = 0; i < tabs_.size(); ++i)
        tabs_[i].setToggleState(static_cast<int>(i) == view_, juce::dontSendNotification);
    const auto controls = view_ == 0 || view_ == 4;
    for (auto* c : std::initializer_list<juce::Component*>{&scope_, &space_})
        c->setVisible(view_ == 0);
    for (auto& element : elements_)
        element->setVisible(view_ == 0);
    for (auto* c : std::initializer_list<juce::Component*>{&layerSelect_, &influence_, &layerState_,
                                                           &exploration_, &xAxis_, &yAxis_})
        c->setVisible(view_ == 4);
    for (auto* c : std::initializer_list<juce::Component*>{&pagePrevious_, &pageNext_, &levelHeader_})
        c->setVisible(controls);
    order_.setVisible(controls && ranked_);
    for (auto& knob : highControls_)
        knob->setVisible(controls);
    for (auto* c : std::initializer_list<juce::Component*>{&primitives_, &parameterViewport_,
                                                           &routingViewport_, &paramHeader_, &routeHeader_,
                                                           &rawPrevious_, &rawNext_, &newRoute_})
        c->setVisible(view_ == 1);
    for (auto* c :
         std::initializer_list<juce::Component*>{&addRoute_, &removeRoute_, &sourceKind_, &sourceElement_,
                                                 &routeDestination_, &routeWeight_, &routeWeightsTensor_})
        c->setVisible(view_ == 1 && routeFormOpen_);
    for (auto* c : std::initializer_list<juce::Component*>{&matrixViewport_, &mappingLayer_, &expert_})
        c->setVisible(view_ == 2);
    for (auto* c :
         std::initializer_list<juce::Component*>{&matrixSelection_, &matrixValue_, &setMatrix_, &capacity_})
        c->setVisible(view_ == 2 && !expertOpen_);
    for (auto* c :
         std::initializer_list<juce::Component*>{&document_, &inspection_, &tensorValues_, &tensorSelect_,
                                                 &apply_, &exportJSON_, &importModel_, &compose_})
        c->setVisible(view_ == 2 && expertOpen_);
    for (auto* c : std::initializer_list<juce::Component*>{&theme_, &refresh_, &folder_, &exportTheme_,
                                                           &importTheme_, &accent_, &gesture_, &precision_,
                                                           &resources_, &hostPrecision_, &showKeyboard_})
        c->setVisible(view_ == 3);
    processor_.uiSettings().setProperty("view", view_, nullptr);
    resized();
    repaint();
}
void Editor::showStatus(const juce::String& text, bool error)
{
    status_.setColour(juce::Label::textColourId, error ? look_.theme().warn : look_.theme().text2);
    status_.setText(text, juce::dontSendNotification);
}
void Editor::applyTheme()
{
    const auto ids = alfa::ui::Theme::builtInIds();
    const auto index = juce::jlimit(0, ids.size() - 1, theme_.getSelectedId() - 1);
    auto chosen = alfa::ui::Theme::builtIn(ids[index]);
    const auto custom = processor_.uiSettings().getProperty("custom_theme", "").toString();
    if (custom.isNotEmpty())
        chosen = alfa::ui::Theme::fromJson(custom);
    const auto accent = processor_.uiSettings().getProperty("accent", "").toString();
    if (accent.isNotEmpty())
        chosen.accent = juce::Colour::fromString("ff" + accent.removeCharacters("#"));
    look_.setTheme(chosen);
    processor_.uiSettings().setProperty("theme", ids[index], nullptr);
    // Labels inside combo boxes and value text copy their colours when built; rebuild them in the new theme.
    sendLookAndFeelChange();
}
void Editor::refreshLibrary()
{
    juce::String error;
    store_->refresh(error);
    library_.setButtonText(PresetStore::presetName(processor_.document()));
    if (error.isNotEmpty())
        showStatus(error, true);
}
void Editor::openLibrary()
{
    if (libraryDialog_)
    {
        libraryDialog_->toFront(true);
        return;
    }
    refreshLibrary();
    std::vector<SoundLibraryPanel::Entry> entries;
    for (const auto& program : factoryPrograms())
        entries.push_back({js(program.id), js(program.name), js(program.category), {}, js(program.document)});
    for (const auto& entry : store_->entries())
        entries.push_back({entry.id, entry.name, entry.category, entry.file, {}});
    auto safe = juce::Component::SafePointer<Editor>(this);
    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = "Sound library";
    options.dialogBackgroundColour = look_.theme().bg;
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = false;
    options.componentToCentreAround = this;
    options.content.setOwned(new SoundLibraryPanel(
        std::move(entries),
        [safe](const juce::String& document, juce::String& error)
        {
            if (!safe)
            {
                error = "Editor closed";
                return false;
            }
            if (!safe->processor_.loadPresetJSON(document, error))
                return false;
            safe->refreshSound();
            safe->showStatus("Loaded configuration: " + PresetStore::presetName(safe->processor_.document()));
            return true;
        },
        &look_.theme()));
    libraryDialog_ = options.launchAsync();
}
void Editor::savePreset()
{
    saveDialog_ = std::make_unique<juce::AlertWindow>(
        "Save sound", "Store the current tensor sound and all control mappings.",
        juce::MessageBoxIconType::NoIcon);
    saveDialog_->addTextEditor("name", PresetStore::presetName(processor_.document()), "Name");
    saveDialog_->addTextEditor("category", PresetStore::presetCategory(processor_.document()), "Category");
    saveDialog_->addButton("Save new", 1, juce::KeyPress(juce::KeyPress::returnKey));
    saveDialog_->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));
    auto safe = juce::Component::SafePointer<Editor>(this);
    saveDialog_->enterModalState(
        true,
        juce::ModalCallbackFunction::create(
            [safe](int result)
            {
                if (!safe || result != 1)
                    return;
                juce::String error;
                const auto name = safe->saveDialog_->getTextEditorContents("name");
                const auto category = safe->saveDialog_->getTextEditorContents("category");
                if (safe->store_->save(name, category, safe->processor_.document(), error))
                {
                    safe->refreshLibrary();
                    safe->showStatus("Saved " + name);
                }
                else
                    safe->showStatus(error, true);
            }),
        false);
}
void Editor::openPreset()
{
    chooser_ = std::make_unique<juce::FileChooser>("Open sound tensor preset", store_->directory(), "*.json");
    auto safe = juce::Component::SafePointer<Editor>(this);
    chooser_->launchAsync(
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [safe](const juce::FileChooser& c)
        {
            if (safe && c.getResult().existsAsFile())
            {
                juce::String error;
                if (safe->processor_.loadPresetJSON(c.getResult().loadFileAsString(), error))
                {
                    safe->refreshSound();
                    safe->showStatus("Loaded " + c.getResult().getFileNameWithoutExtension());
                }
                else
                    safe->showStatus(error, true);
            }
        });
}
void Editor::captureSnapshot(int index)
{
    selectedSnapshot_ = juce::jlimit(0, 1, index);
    auto& snapshot = snapshots_[static_cast<size_t>(selectedSnapshot_)];
    snapshot.document = processor_.document();
    for (size_t i = 0; i < snapshotIds.size(); ++i)
        if (auto* p = processor_.parameters().getRawParameterValue(snapshotIds[i]))
            snapshot.controls[i] = p->load();
    snapshot.valid = true;
    a_.setToggleState(selectedSnapshot_ == 0, juce::dontSendNotification);
    b_.setToggleState(selectedSnapshot_ == 1, juce::dontSendNotification);
    showStatus("Captured snapshot " + juce::String(selectedSnapshot_ == 0 ? "A" : "B"));
}
void Editor::recallSnapshot(int index)
{
    selectedSnapshot_ = juce::jlimit(0, 1, index);
    const auto& snapshot = snapshots_[static_cast<size_t>(selectedSnapshot_)];
    if (!snapshot.valid)
    {
        captureSnapshot(index);
        return;
    }
    juce::String error;
    if (!processor_.loadJSON(snapshot.document, error))
    {
        showStatus(error, true);
        return;
    }
    for (size_t i = 0; i < snapshotIds.size(); ++i)
        processor_.setParameterValue(snapshotIds[i], snapshot.controls[i]);
    a_.setToggleState(index == 0, juce::dontSendNotification);
    b_.setToggleState(index == 1, juce::dontSendNotification);
    refreshSound();
    showStatus("Snapshot " + juce::String(index == 0 ? "A" : "B"));
}
void Editor::refreshSound()
{
    try
    {
        sound_ = nlohmann::json::parse(processor_.document().toStdString());
        document_.setText(js(sound_.dump(2)), false);
        precision_.setText(processor_.scalarBytes() == 8 ? "Uniform DSP precision: float64"
                                                         : "Uniform DSP precision: float32",
                           juce::dontSendNotification);
        revision_ = processor_.programRevision.load();
        telemetryMark_ = processor_.telemetryBlocks.load(std::memory_order_acquire);
        selectedPrimitive_ =
            juce::jlimit(0, static_cast<int>(sound_.at("instances").size()), selectedPrimitive_);
        primitives_.updateContent();
        primitives_.selectRow(selectedPrimitive_, true, false);
        refreshing_ = true;
        const auto level = layerSelect_.getSelectedId(), matrixLayer = mappingLayer_.getSelectedId();
        layerSelect_.clear(juce::dontSendNotification);
        mappingLayer_.clear(juce::dontSendNotification);
        const auto layers = sound_.value("control_layers", nlohmann::json::array());
        for (size_t i = 0; i < layers.size(); ++i)
        {
            layerSelect_.addItem("Layer " + juce::String(static_cast<int>(i)), static_cast<int>(i) + 1);
            mappingLayer_.addItem("Layer " + juce::String(static_cast<int>(i)) + " matrix",
                                  static_cast<int>(i) + 1);
        }
        layerSelect_.setSelectedId(juce::jlimit(1, juce::jmax(1, static_cast<int>(layers.size())), level),
                                   juce::dontSendNotification);
        mappingLayer_.setSelectedId(
            layers.empty() ? 0 : juce::jlimit(1, static_cast<int>(layers.size()), matrixLayer),
            juce::dontSendNotification);
        refreshing_ = false;
        highLayer_ = juce::jmax(0, layerSelect_.getSelectedId() - 1);
        const auto highest = layers.empty() ? size_t(0) : layers.at(0).at("values").size();
        refreshing_ = true;
        const auto previousX = xAxis_.getSelectedId(), previousY = yAxis_.getSelectedId();
        xAxis_.clear(juce::dontSendNotification);
        yAxis_.clear(juce::dontSendNotification);
        for (size_t i = 0; i < juce::jmin(size_t(8), highest); ++i)
        {
            const auto title = controlName(0, i);
            xAxis_.addItem(title, static_cast<int>(i) + 1);
            yAxis_.addItem(title, static_cast<int>(i) + 1);
        }
        const auto capacity = static_cast<int>(juce::jmin(size_t(8), highest));
        xAxis_.setSelectedId(capacity ? juce::jlimit(1, capacity, previousX) : 0, juce::dontSendNotification);
        yAxis_.setSelectedId(capacity ? juce::jlimit(1, capacity, previousY) : 0, juce::dontSendNotification);
        exploration_.xId = "x" + juce::String(juce::jmax(0, xAxis_.getSelectedId() - 1));
        exploration_.xName = xAxis_.getText();
        exploration_.yId = "x" + juce::String(juce::jmax(0, yAxis_.getSelectedId() - 1));
        exploration_.yName = yAxis_.getText();
        refreshing_ = false;
        rebuildParameters();
        rebuildRoutes();
        rebuildLayers();
        rebuildHighControls();
        refreshTensor();
        rebuildElements();
        refreshSpace();
        std::vector<LayerStateView::Layer> stateLayers;
        for (const auto& layer : sound_.value("control_layers", nlohmann::json::array()))
        {
            auto& state = stateLayers.emplace_back();
            for (const auto& range : layer.at("ranges"))
                state.ranges.push_back({range.at(0).get<double>(), range.at(1).get<double>()});
            for (const auto& value : layer.at("values"))
                state.values.push_back(value.get<double>());
        }
        layerState_.setLayers(std::move(stateLayers));
        library_.setButtonText(PresetStore::presetName(processor_.document()));
        chooseView(view_);
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
}
juce::String Editor::parameterLabel(size_t index) const
{
    if (sound_.contains("parameter_metadata"))
        for (const auto& meta : sound_["parameter_metadata"])
            if (meta.value("index", size_t(-1)) == index)
                return "X[raw," + juce::String(static_cast<int>(index)) + "]  " +
                       js(meta.value("name", std::string{}));
    return "X[raw," + juce::String(static_cast<int>(index)) + "]";
}
juce::String Editor::parameterName(size_t index) const
{
    if (sound_.contains("parameter_metadata"))
        for (const auto& meta : sound_["parameter_metadata"])
            if (meta.value("index", size_t(-1)) == index)
                if (const auto name = meta.value("name", std::string{}); !name.empty())
                    return js(name);
    return "X[raw," + juce::String(static_cast<int>(index)) + "]";
}
juce::String Editor::controlName(size_t layer, size_t index) const
{
    const auto layers = sound_.value("control_layers", nlohmann::json::array());
    if (layer < layers.size())
        if (const auto labels = layers[layer].value("labels", nlohmann::json::array()); index < labels.size())
            if (const auto name = labels[index].get<std::string>(); !name.empty())
                return js(name);
    return "X[" + juce::String(static_cast<int>(layer)) + "," + juce::String(static_cast<int>(index)) + "]";
}
juce::String Editor::controlLabel(size_t layer, size_t index) const
{
    // A user name with its coordinate, where both help (selections, tooltips).
    const auto coordinate =
        "X[" + juce::String(static_cast<int>(layer)) + "," + juce::String(static_cast<int>(index)) + "]";
    const auto name = controlName(layer, index);
    return name == coordinate ? coordinate : name + "  " + coordinate;
}
void Editor::renameControl(size_t layer, size_t index, const juce::String& name)
{
    try
    {
        if (name.getNumBytesAsUTF8() > 64)
            throw std::invalid_argument("Control names are limited to 64 bytes");
        auto data = nlohmann::json::parse(processor_.document().toStdString());
        auto& entry = data.at("control_layers").at(layer);
        const auto count = entry.at("values").size();
        auto labels = entry.value("labels", nlohmann::json::array());
        labels.get_ref<nlohmann::json::array_t&>().resize(count, "");
        // The default X[layer,index] is not stored; clearing every name removes the field.
        const auto coordinate =
            "X[" + juce::String(static_cast<int>(layer)) + "," + juce::String(static_cast<int>(index)) + "]";
        labels.at(index) = name == coordinate ? std::string{} : string(name);
        if (std::all_of(labels.begin(), labels.end(),
                        [](const nlohmann::json& l) { return l.get<std::string>().empty(); }))
            entry.erase("labels");
        else
            entry["labels"] = std::move(labels);
        juce::String error;
        if (changeDocument(std::move(data), error))
            showStatus(name.isEmpty() ? coordinate + " uses its default name" : coordinate + " renamed");
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
}
void Editor::focusControl(int index)
{
    if (focusedControl_ == index)
        return;
    focusedControl_ = index;
    refreshInfluence();
}
void Editor::refreshInfluence()
{
    // Compose the authored matrices from the focused control down to the bottom parameters.
    // This is the linear sensitivity; clamps between layers can only reduce it.
    try
    {
        const auto layers = sound_.value("control_layers", nlohmann::json::array());
        const auto layer = static_cast<size_t>(highLayer_);
        if (layer >= layers.size() || layers[layer].at("values").empty())
        {
            influence_.setEntries("No control layer", {}, 0);
            return;
        }
        const auto sources = layers[layer].at("values").size();
        focusedControl_ = juce::jlimit(0, static_cast<int>(sources) - 1, focusedControl_);
        std::vector<double> vector(sources, 0.0);
        vector[static_cast<size_t>(focusedControl_)] = 1;
        for (size_t i = layer; i < layers.size(); ++i)
        {
            const auto& matrix = layers[i].at("matrix");
            const auto rows = matrix.at("shape").at(0).get<size_t>();
            const auto columns = matrix.at("shape").at(1).get<size_t>();
            const auto& values = matrix.at("values");
            std::vector<double> next(rows, 0.0);
            for (size_t d = 0; d < rows; ++d)
                for (size_t s = 0; s < columns && s < vector.size(); ++s)
                    if (std::abs(vector[s]) > 0.0)
                        next[d] += values.at(d * columns + s).get<double>() * vector[s];
            vector = std::move(next);
        }
        const auto locks = sound_.value("parameter_locks", nlohmann::json::array());
        std::vector<InfluenceView::Entry> entries;
        for (size_t i = 0; i < vector.size(); ++i)
            if (std::abs(vector[i]) > 1e-12)
                entries.push_back(
                    {parameterName(i), vector[i], i < locks.size() && locks[i].get<int>() != 0});
        const auto reached = static_cast<int>(entries.size());
        std::stable_sort(entries.begin(), entries.end(), [](const auto& a, const auto& b)
                         { return std::abs(a.weight) > std::abs(b.weight); });
        influence_.setEntries(controlName(layer, static_cast<size_t>(focusedControl_)), std::move(entries),
                              reached);
    }
    catch (const std::exception& e)
    {
        influence_.setEntries("Influence unavailable", {}, 0);
        showStatus(e.what(), true);
    }
}
void Editor::stepSound(int direction)
{
    // Factory programs first, then the user's library, in the same order as the library window.
    std::vector<std::pair<juce::String, juce::String>> sounds;
    for (const auto& program : factoryPrograms())
        sounds.push_back({js(program.name), js(program.document)});
    for (const auto& entry : store_->entries())
        sounds.push_back({entry.name, {}});
    if (sounds.empty())
        return;
    const auto current = PresetStore::presetName(processor_.document());
    int index = -1;
    for (size_t i = 0; i < sounds.size(); ++i)
        if (sounds[i].first == current)
            index = static_cast<int>(i);
    const auto count = static_cast<int>(sounds.size());
    const auto target = static_cast<size_t>(index < 0 ? (direction > 0 ? 0 : count - 1)
                                                      : (index + direction + count) % count);
    auto document = sounds[target].second;
    if (document.isEmpty())
        document = store_->entries()[target - factoryPrograms().size()].file.loadFileAsString();
    juce::String error;
    if (processor_.loadPresetJSON(document, error))
    {
        refreshSound();
        showStatus("Loaded " + sounds[target].first);
    }
    else
        showStatus(error, true);
}
void Editor::syncMode()
{
    const auto mode = juce::jlimit(0, 2, processor_.mode.load());
    for (size_t i = 0; i < modes_.size(); ++i)
        modes_[i].setToggleState(static_cast<int>(i) == mode, juce::dontSendNotification);
}
juce::String Editor::parameterUnit(size_t index) const
{
    if (sound_.contains("parameter_metadata"))
        for (const auto& meta : sound_["parameter_metadata"])
            if (meta.value("index", size_t(-1)) == index)
                return js(meta.value("unit", std::string{}));
    return "value";
}
std::optional<std::array<double, 3>> Editor::parameterRange(size_t index) const
{
    if (sound_.contains("parameter_metadata"))
        for (const auto& meta : sound_["parameter_metadata"])
            if (meta.value("index", size_t(-1)) == index && meta.contains("min") && meta.contains("max"))
            {
                const auto low = meta["min"].get<double>(), high = meta["max"].get<double>(),
                           step = meta.value("step", 0.0);
                if (std::isfinite(low) && std::isfinite(high) && low < high && std::isfinite(step) &&
                    step >= 0)
                    return std::array<double, 3>{low, high, step};
            }
    return std::nullopt;
}
int Editor::getNumRows()
{
    try
    {
        return static_cast<int>(sound_.at("instances").size()) + 1;
    }
    catch (const std::exception&)
    {
        return 0;
    }
}
void Editor::paintListBoxItem(int row, juce::Graphics& g, int width, int height, bool selected)
{
    // One card per instance: family colour, title, primitive, activation and the live output level.
    const auto& t = look_.theme();
    auto card = juce::Rectangle<float>(3, 3, static_cast<float>(width - 6), static_cast<float>(height - 6));
    g.setColour(selected ? t.accent.withAlpha(0.10f) : t.bg.withAlpha(0.55f));
    g.fillRoundedRectangle(card, 6);
    if (selected)
    {
        g.setColour(t.accent.withAlpha(0.7f));
        g.drawRoundedRectangle(card, 6, 1);
    }
    if (row == 0)
    {
        g.setColour(selected ? t.accent : t.text);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13, true));
        g.drawText("All raw parameters", card.reduced(12, 0).toNearestInt(),
                   juce::Justification::centredLeft);
        return;
    }
    try
    {
        const auto index = static_cast<size_t>(row - 1);
        const auto& instance = sound_.at("instances").at(index);
        const auto id = js(instance.at("primitive").get<std::string>());
        const auto activation = sound_.at("tensors")
                                    .at("activation")
                                    .at("values")
                                    .at(instance.at("activation").get<size_t>())
                                    .get<double>();
        const auto colour = familyColour(t, id);
        const auto active = activation != 0.0;
        g.setColour(active ? colour : colour.withAlpha(0.3f));
        g.fillRoundedRectangle(card.withWidth(4).translated(1, 0).reduced(0, 4), 2);
        auto text = card.reduced(14, 6);
        const auto parameters = instance.at("parameters").at("count").get<int>();
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13, true));
        g.setColour(active ? (selected ? t.accent : t.text) : t.text2);
        g.drawText(instanceTitle(sound_, index), text.removeFromTop(18).withTrimmedRight(70).toNearestInt(),
                   juce::Justification::centredLeft, true);
        g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(11));
        g.setColour(active ? t.text : t.text2);
        g.drawText(active ? "act " + compactText(activation) : juce::String("off"),
                   juce::Rectangle<float>(card.getRight() - 84, card.getY() + 6, 72, 18).toNearestInt(),
                   juce::Justification::centredRight);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
        g.setColour(colour.interpolatedWith(t.text2, 0.35f));
        g.drawText(juce::String(static_cast<int>(index)) + "  " + id +
                       (parameters ? "  /  " + juce::String(parameters) + " parameters" : juce::String()),
                   text.removeFromTop(15).toNearestInt(), juce::Justification::centredLeft, true);
        // Live output level of this instance in the last block, on a 60 dB scale.
        auto meter = text.removeFromBottom(5);
        g.setColour(t.line);
        g.fillRoundedRectangle(meter, 2.5f);
        if (telemetryFresh() && index < processor_.instanceLevels.size())
        {
            const auto level =
                levelFraction(processor_.instanceLevels[index].load(std::memory_order_relaxed));
            g.setColour(colour);
            g.fillRoundedRectangle(meter.withWidth(meter.getWidth() * level), 2.5f);
        }
    }
    catch (const std::exception&)
    {
    }
}
void Editor::selectedRowsChanged(int row)
{
    selectedPrimitive_ = juce::jmax(0, row);
    rawPage_ = 0;
    rebuildParameters();
    rebuildRoutes();
}
void Editor::rebuildParameters()
{
    parameterRows_.clear();
    parameterRowIndex_.clear();
    parameterLiveColour_.clear();
    try
    {
        // The live mark takes the colour of the strongest relationship into the parameter; parameters moved
        // only by control layers use the modulation colour.
        std::map<size_t, std::pair<double, juce::String>> strongest;
        for (const auto& route : sound_.value("projections", nlohmann::json::array()))
        {
            const auto target = route.at("parameter_element").get<size_t>();
            const auto weight = std::abs(sound_.at("tensors")
                                             .at(route.at("weights").get<std::string>())
                                             .at("values")
                                             .at(route.at("weight_element").get<size_t>())
                                             .get<double>());
            if (!strongest.count(target) || weight > strongest[target].first)
                strongest[target] = {weight, js(route.at("source").get<std::string>())};
        }
        const auto& values = sound_.at("tensors").at("parameters").at("values");
        size_t start = 0, count = values.size();
        if (selectedPrimitive_ > 0)
        {
            const auto& instance = sound_.at("instances").at(static_cast<size_t>(selectedPrimitive_ - 1));
            start = instance.at("parameters").at("offset").get<size_t>();
            count = instance.at("parameters").at("count").get<size_t>();
            const auto ai = instance.at("activation").get<size_t>();
            auto row = std::make_unique<NumericRow>(
                "Primitive activation", "weight", sound_["tensors"]["activation"]["values"][ai].get<double>(),
                [this, ai](double v, juce::String& e) { return changeTensor("activation", ai, v, e); });
            parameterContent_.addAndMakeVisible(*row);
            parameterRows_.push_back(std::move(row));
            parameterRowIndex_.push_back(-1);
            parameterLiveColour_.push_back(look_.theme().mod);
        }
        rawPage_ = juce::jlimit(0, juce::jmax(0, static_cast<int>((count + 63) / 64) - 1), rawPage_);
        const auto begin = start + static_cast<size_t>(rawPage_) * 64;
        const auto end = juce::jmin(start + count, begin + 64);
        const auto locks = sound_.value("parameter_locks", nlohmann::json::array());
        for (size_t index = begin; index < end && index < values.size(); ++index)
        {
            const auto locked = index < locks.size() && locks[index].get<int>() != 0;
            auto row = std::make_unique<NumericRow>(
                parameterName(index) + "  [" + juce::String(static_cast<int>(index)) + "]",
                parameterUnit(index), values[index].get<double>(),
                [this, index](double v, juce::String& e)
                {
                    const auto ok = processor_.editTensorParameter(index, v, e);
                    showStatus(ok ? "Updated " + parameterLabel(index) : e, !ok);
                    return ok;
                },
                [this, index](bool lock)
                {
                    juce::String error;
                    if (!processor_.setParameterLock(index, lock, error))
                        showStatus(error, true);
                },
                locked, parameterRange(index));
            parameterContent_.addAndMakeVisible(*row);
            parameterRows_.push_back(std::move(row));
            parameterRowIndex_.push_back(static_cast<int>(index));
            parameterLiveColour_.push_back(strongest.count(index)
                                               ? sourceColour(look_.theme(), strongest[index].second)
                                               : look_.theme().mod);
        }
        paramHeader_.setText(juce::String(static_cast<int>(begin)) + "-" +
                                 juce::String(static_cast<int>(end)) + " / " +
                                 juce::String(static_cast<int>(count)) + " dimensions",
                             juce::dontSendNotification);
        rawPrevious_.setEnabled(rawPage_ > 0);
        rawNext_.setEnabled(end < start + count);
        resized();
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
}
bool Editor::changeTensor(const juce::String& category, size_t index, double value, juce::String& error)
{
    try
    {
        auto data = nlohmann::json::parse(processor_.document().toStdString());
        data.at("tensors").at(string(category)).at("values").at(index) = value;
        return changeDocument(std::move(data), error);
    }
    catch (const std::exception& e)
    {
        error = e.what();
        showStatus(error, true);
        return false;
    }
}
bool Editor::changeDocument(nlohmann::json data, juce::String& error)
{
    const auto result = processor_.editConfigurationJSON(js(data.dump(2)), error);
    if (!result)
        showStatus(error, true);
    return result;
}
juce::String Editor::sourceLabel(const nlohmann::json& route) const
{
    const auto kind = js(route.at("source").get<std::string>());
    const auto source = route.at("source_element").get<size_t>();
    if (kind == "midi")
    {
        static const std::array<juce::String, 10> names{"Pitch Hz", "Velocity", "Gate", "Duration",
                                                        "Pressure", "Bend",     "CC74", "Onset",
                                                        "Channel",  "Key"};
        return "Voice " + juce::String(static_cast<int>(source / 10)) + " / " + names[source % 10];
    }
    return kind + "[" + juce::String(static_cast<int>(source)) + "]";
}
void Editor::rebuildRoutes()
{
    if (refreshing_)
        return;
    routeRows_.clear();
    try
    {
        const auto routes = sound_.value("projections", nlohmann::json::array());
        // Show the relationships that drive the selected instance; "All raw parameters" shows every one.
        size_t first = 0, last = std::numeric_limits<size_t>::max();
        if (selectedPrimitive_ > 0)
        {
            const auto& instance = sound_.at("instances").at(static_cast<size_t>(selectedPrimitive_ - 1));
            first = instance.at("parameters").at("offset").get<size_t>();
            last = first + instance.at("parameters").at("count").get<size_t>();
        }
        std::vector<size_t> shown;
        double scale = 1;
        for (size_t i = 0; i < routes.size(); ++i)
        {
            const auto target = routes[i].at("parameter_element").get<size_t>();
            if (target < first || target >= last)
                continue;
            shown.push_back(i);
            const auto tensorName = routes[i].at("weights").get<std::string>();
            const auto weight = routes[i].at("weight_element").get<size_t>();
            scale = juce::jmax(
                scale, std::abs(sound_.at("tensors").at(tensorName).at("values").at(weight).get<double>()));
        }
        routeHeader_.setText(selectedPrimitive_ > 0
                                 ? juce::String(static_cast<int>(shown.size())) + " of " +
                                       juce::String(static_cast<int>(routes.size())) +
                                       " relationships drive this instance"
                                 : juce::String(static_cast<int>(routes.size())) + " relationships",
                             juce::dontSendNotification);
        for (const auto i : shown)
        {
            const auto& route = routes[i];
            const auto tensorName = js(route.at("weights").get<std::string>());
            const auto weight = route.at("weight_element").get<size_t>();
            const auto target = route.at("parameter_element").get<size_t>();
            auto row = std::make_unique<NumericRow>(
                sourceLabel(route) + "  >  " + parameterName(target), tensorName,
                sound_.at("tensors").at(string(tensorName)).at("values").at(weight).get<double>(),
                [this, tensorName, weight](double v, juce::String& e)
                { return changeTensor(tensorName, weight, v, e); });
            row->showWeightBar(look_.theme(), scale);
            row->setIdentity(sourceColour(look_.theme(), js(route.at("source").get<std::string>())));
            routingContent_.addAndMakeVisible(*row);
            routeRows_.push_back(std::move(row));
        }
        const auto destination = routeDestination_.getSelectedId(), weight = routeWeight_.getSelectedId();
        routeDestination_.clear(juce::dontSendNotification);
        const auto& params = sound_.at("tensors").at("parameters").at("values");
        for (size_t i = 0; i < params.size(); ++i)
            routeDestination_.addItem(parameterLabel(i), static_cast<int>(i) + 1);
        routeDestination_.setSelectedId(
            juce::jlimit(1, juce::jmax(1, static_cast<int>(params.size())), destination),
            juce::dontSendNotification);
        routeWeight_.clear(juce::dontSendNotification);
        const auto field = routeWeightsTensor_.getSelectedId() == 2 ? "modulation" : "routing";
        const auto& weights = sound_.at("tensors").at(field).at("values");
        for (size_t i = 0; i < weights.size(); ++i)
            routeWeight_.addItem("Coefficient " + juce::String(static_cast<int>(i)) + " = " +
                                     juce::String(weights[i].get<double>(), 3),
                                 static_cast<int>(i) + 1);
        routeWeight_.setSelectedId(juce::jlimit(1, juce::jmax(1, static_cast<int>(weights.size())), weight),
                                   juce::dontSendNotification);
        addRoute_.setEnabled(!params.empty() && !weights.empty());
        removeRoute_.setEnabled(!routes.empty());
        resized();
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
}
void Editor::addRelationship()
{
    try
    {
        auto data = nlohmann::json::parse(processor_.document().toStdString());
        if (!data.contains("projections"))
            data["projections"] = nlohmann::json::array();
        data["projections"].push_back(
            {{"source", string(sourceKind_.getText())},
             {"source_element", static_cast<size_t>(sourceElement_.getText().getLargeIntValue())},
             {"parameter_element", routeDestination_.getSelectedId() - 1},
             {"weights", routeWeightsTensor_.getSelectedId() == 2 ? "modulation" : "routing"},
             {"weight_element", routeWeight_.getSelectedId() - 1},
             {"missing_source", "zero"}});
        juce::String error;
        showStatus(changeDocument(std::move(data), error) ? "Relationship added" : error, error.isNotEmpty());
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
}
void Editor::rebuildLayers()
{
    if (refreshing_)
        return;
    try
    {
        const auto layers = sound_.value("control_layers", nlohmann::json::array());
        const auto layer = static_cast<size_t>(juce::jmax(0, mappingLayer_.getSelectedId() - 1));
        if (layer < layers.size())
        {
            const auto& current = layers[layer];
            const auto sources = current.at("values").size();
            const auto destinations = current.at("matrix").at("shape").at(0).get<size_t>();
            std::vector<double> values;
            for (const auto& v : current.at("matrix").at("values"))
                values.push_back(v.get<double>());
            juce::StringArray rows, columns;
            for (size_t i = 0; i < destinations; ++i)
                rows.add(layer + 1 < layers.size() ? controlName(layer + 1, i) : parameterName(i));
            for (size_t i = 0; i < sources; ++i)
                columns.add(controlName(layer, i));
            matrix_.setMatrix(std::move(values), rows, columns);
            matrixSource_ = juce::jlimit(0, juce::jmax(0, static_cast<int>(sources) - 1), matrixSource_);
            matrixDestination_ =
                juce::jlimit(0, juce::jmax(0, static_cast<int>(destinations) - 1), matrixDestination_);
            matrix_.select(matrixSource_, matrixDestination_);
            setMatrix_.setEnabled(sources > 0 && destinations > 0);
        }
        else
        {
            matrix_.setMatrix({}, {}, {});
            setMatrix_.setEnabled(false);
        }
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
    refreshTensor();
    resized();
}
void Editor::selectMatrixCell(int source, int destination)
{
    matrixSource_ = source;
    matrixDestination_ = destination;
    matrix_.select(source, destination);
    refreshTensor();
}
std::vector<size_t> Editor::controlOrder() const
{
    // A learned ranking written by AOM Centre orders the controls, most important first.
    // Without one the order is the authored index order and nothing is called learned.
    const auto layers = sound_.value("control_layers", nlohmann::json::array());
    const auto layer = static_cast<size_t>(highLayer_);
    const auto count = layer < layers.size() ? layers[layer].at("values").size() : size_t(0);
    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), size_t{0});
    if (indexOrder_ || layer >= layers.size() || !layers[layer].contains("ranking"))
        return order;
    const auto importance = layers[layer].at("ranking").at("importance").get<std::vector<double>>();
    if (importance.size() == count)
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return importance[a] > importance[b]; });
    return order;
}
void Editor::rebuildHighControls()
{
    highControls_.clear();
    try
    {
        const auto layers = sound_.value("control_layers", nlohmann::json::array());
        const auto layer = static_cast<size_t>(highLayer_);
        size_t count = 0;
        if (layer < layers.size())
            count = layers[layer].at("values").size();
        // A learned ranking written by AOM Centre orders the controls, most important first.
        // Without one the order is the authored index order and nothing is called learned.
        std::vector<size_t> order(count);
        std::iota(order.begin(), order.end(), size_t{0});
        std::vector<double> importance;
        juce::String model;
        if (layer < layers.size() && layers[layer].contains("ranking"))
        {
            const auto& ranking = layers[layer].at("ranking");
            importance = ranking.at("importance").get<std::vector<double>>();
            model = js(ranking.value("name", std::string{}));
            if (model.isEmpty())
                model = js(ranking.at("model").get<std::string>()).substring(0, 8);
        }
        ranked_ = importance.size() == count && count > 0;
        order_.setToggleState(ranked_ && !indexOrder_, juce::dontSendNotification);
        const bool learned = ranked_ && !indexOrder_;
        if (learned)
            std::stable_sort(order.begin(), order.end(),
                             [&](size_t a, size_t b) { return importance[a] > importance[b]; });
        highPage_ = juce::jlimit(0, juce::jmax(0, static_cast<int>((count + 7) / 8) - 1), highPage_);
        const auto begin = static_cast<size_t>(highPage_) * 8, end = juce::jmin(count, begin + 8);
        for (size_t position = begin; position < end; ++position)
        {
            const auto index = order[position];
            std::unique_ptr<juce::Component> knob;
            const auto name = controlName(layer, index);
            const auto coordinate =
                "X[" + juce::String(highLayer_) + "," + juce::String(static_cast<int>(index)) + "]";
            const auto title = learned ? juce::String(static_cast<int>(position) + 1) + "  " + name : name;
            const auto raw = name == coordinate ? juce::String() : name;
            const auto focus = [this, index]
            {
                focusControl(static_cast<int>(index));
            };
            const auto rename =
                [safe = juce::Component::SafePointer<Editor>(this), layer, index](const juce::String& text)
            {
                // The rename rebuilds these knobs, so it runs after the label's own callback returns.
                juce::MessageManager::callAsync(
                    [safe, layer, index, text]
                    {
                        if (safe)
                            safe->renameControl(layer, index, text);
                    });
            };
            juce::Label* label = nullptr;
            if (layer == 0 && index < 8)
            {
                auto host = std::make_unique<ParameterKnob>(
                    processor_, "x" + juce::String(static_cast<int>(index)), title);
                host->onTouched = focus;
                label = &host->title();
                knob = std::move(host);
            }
            else
            {
                const auto& range = layers[layer].at("ranges").at(index);
                auto deep = std::make_unique<LayerKnob>(processor_, layer, index,
                                                        layers[layer].at("values").at(index).get<double>(),
                                                        range.at(0).get<double>(), range.at(1).get<double>());
                deep->onTouched = focus;
                deep->title().setText(title, juce::dontSendNotification);
                label = &deep->title();
                knob = std::move(deep);
            }
            allowRename(*label, raw, rename);
            if (learned)
                label->setTooltip(coordinate + "  learned importance " + compactText(importance[index]) +
                                  ". Double-click to rename");
            else if (name != coordinate)
                label->setTooltip(coordinate + ". Double-click to rename");
            addChildComponent(*knob);
            highControls_.push_back(std::move(knob));
        }
        layerState_.setPage(highLayer_, static_cast<int>(begin), static_cast<int>(end));
        const auto pages =
            juce::String(highPage_ + 1) + "/" + juce::String(static_cast<int>((count + 7) / 8));
        if (!count)
            levelHeader_.setText("This sound has no higher control layer", juce::dontSendNotification);
        else if (learned)
            levelHeader_.setText("Ranked by " + model + ": " + juce::String(static_cast<int>(begin) + 1) +
                                     "-" + juce::String(static_cast<int>(end)) + " of " +
                                     juce::String(static_cast<int>(count)) + ", page " + pages,
                                 juce::dontSendNotification);
        else
            levelHeader_.setText(juce::String(static_cast<int>(begin)) + "-" +
                                     juce::String(static_cast<int>(end) - 1) + " of " +
                                     juce::String(static_cast<int>(count)) + " controls, page " + pages,
                                 juce::dontSendNotification);
        refreshInfluence();
        pagePrevious_.setEnabled(highPage_ > 0);
        pageNext_.setEnabled(end < count);
        xAxis_.setEnabled(count > 0);
        yAxis_.setEnabled(count > 0);
        exploration_.setEnabled(count > 0);
        for (auto& control : highControls_)
            control->setVisible(view_ == 0 || view_ == 4);
        order_.setVisible((view_ == 0 || view_ == 4) && ranked_);
        resized();
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
}
void Editor::refreshTensor()
{
    if (refreshing_)
        return;
    try
    {
        const std::array<const char*, 4> fields{"routing", "parameters", "activation", "modulation"};
        const auto index = tensorSelect_.getSelectedId() - 1;
        if (index >= 0 && index < 4)
            tensorValues_.setText(js(sound_.at("tensors").at(fields[static_cast<size_t>(index)]).dump(2)),
                                  false);
        else
        {
            nlohmann::json states = nlohmann::json::array();
            for (const auto& instance : sound_.at("instances"))
                states.push_back(
                    {{"primitive", instance.at("primitive")},
                     {"clock", instance.at("clock")},
                     {"state_size", instance.value("state_size", size_t(0))},
                     {"initial_state", instance.value("initial_state", nlohmann::json::array())},
                     {"assets", instance.value("assets", nlohmann::json::array())},
                     {"input_channels", instance.value("input_channels", size_t(0))},
                     {"audio_bindings", instance.value("audio_bindings", nlohmann::json::array())}});
            tensorValues_.setText(js(states.dump(2)), false);
        }
        const auto layers = sound_.value("control_layers", nlohmann::json::array());
        const auto layer = mappingLayer_.getSelectedId() - 1, source = matrixSource_,
                   destination = matrixDestination_;
        if (layer >= 0 && static_cast<size_t>(layer) < layers.size())
        {
            const auto& matrix = layers[static_cast<size_t>(layer)].at("matrix");
            const auto rows = matrix.at("shape").at(0).get<size_t>();
            const auto columns = matrix.at("shape").at(1).get<size_t>();
            if (static_cast<size_t>(source) < columns && static_cast<size_t>(destination) < rows)
            {
                const auto value =
                    matrix.at("values")
                        .at(static_cast<size_t>(destination) * columns + static_cast<size_t>(source))
                        .get<double>();
                matrixValue_.setText(scalarText(value), false);
                matrixSelection_.setText(
                    "From  " + controlLabel(static_cast<size_t>(layer), static_cast<size_t>(source)) +
                        "\nTo  " +
                        (static_cast<size_t>(layer) + 1 < layers.size()
                             ? controlLabel(static_cast<size_t>(layer) + 1, static_cast<size_t>(destination))
                             : parameterName(static_cast<size_t>(destination))) +
                        "\n\n" +
                        (value == 0.0
                             ? juce::String("No relationship yet; enter a coefficient to add one.")
                             : juce::String(value > 0 ? "Raises" : "Lowers") + " the destination by " +
                                   compactText(std::abs(value)) + " per unit of the source."),
                    juce::dontSendNotification);
            }
        }
        else
            matrixSelection_.setText("This sound has no control layer", juce::dontSendNotification);
        const auto report = processor_.inspection();
        inspection_.setText(report, false);
        juce::String budgetSource;
        auto usage = CapacityView::parse(report, budgetSource);
        const auto precision = processor_.scalarBytes() == 8 ? "float64" : "float32";
        capacity_.setEntries(std::move(usage),
                             "Precision " + juce::String(precision) + ". Budget from " + budgetSource + ".");
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
}
void Editor::rebuildElements()
{
    // One element per primitive, in order of first appearance; at most five, specific views first.
    for (auto& element : elements_)
        removeChildComponent(element.get());
    elements_.clear();
    try
    {
        const auto& instances = sound_.at("instances");
        const auto& parameters = sound_.at("tensors").at("parameters").at("values");
        std::vector<juce::String> primitives;
        for (const auto& instance : instances)
            if (const auto id = js(instance.at("primitive").get<std::string>());
                std::find(primitives.begin(), primitives.end(), id) == primitives.end())
                primitives.push_back(id);
        std::stable_partition(primitives.begin(), primitives.end(), [](const juce::String& id)
                              { return ElementView::kindOf(id) != ElementView::Kind::Generic; });
        if (primitives.size() > 5)
            primitives.resize(5);
        const auto base = parameters.get<std::vector<double>>();
        for (const auto& id : primitives)
        {
            std::vector<ElementView::Instance> members;
            for (size_t i = 0; i < instances.size(); ++i)
                if (js(instances[i].at("primitive").get<std::string>()) == id)
                    members.push_back({i, instances[i].at("parameters").at("offset").get<size_t>(),
                                       instances[i].at("parameters").at("count").get<size_t>()});
            std::vector<std::array<double, 2>> ranges;
            std::vector<juce::String> names;
            for (size_t p = 0; p < members.front().count; ++p)
            {
                const auto range = parameterRange(members.front().offset + p);
                ranges.push_back(range ? std::array<double, 2>{(*range)[0], (*range)[1]}
                                       : std::array<double, 2>{0.0, 0.0});
                names.push_back(parameterName(members.front().offset + p));
            }
            static const std::map<ElementView::Kind, juce::String> kindNames{
                {ElementView::Kind::Envelope, "Envelope"},  {ElementView::Kind::Lowpass, "Low-pass"},
                {ElementView::Kind::Partials, "Partials"},  {ElementView::Kind::Delay, "Delay"},
                {ElementView::Kind::Width, "Stereo width"}, {ElementView::Kind::Reverb, "Reverb"}};
            const auto kind = ElementView::kindOf(id);
            const auto title = kindNames.count(kind) ? kindNames.at(kind)
                               : members.size() > 1
                                   ? id.fromFirstOccurrenceOf(".", false, false).replaceCharacter('_', ' ')
                                   : instanceTitle(sound_, members.front().index);
            auto element = std::make_unique<ElementView>(
                look_, id, title.substring(0, 1).toUpperCase() + title.substring(1), members,
                std::move(ranges), std::move(names));
            element->setBase(base);
            const auto first = members.front().index;
            element->onOpen = [this, first]
            {
                openInstance(first);
            };
            element->onEdit = [this, members](const std::vector<std::pair<size_t, double>>& edits)
            {
                try
                {
                    auto data = nlohmann::json::parse(processor_.document().toStdString());
                    auto& values = data.at("tensors").at("parameters").at("values");
                    for (const auto& member : members)
                        for (const auto& [parameter, value] : edits)
                            if (parameter < member.count)
                                values.at(member.offset + parameter) = value;
                    juce::String error;
                    if (changeDocument(std::move(data), error))
                        showStatus("Updated " + juce::String(static_cast<int>(members.size())) +
                                   (members.size() == 1 ? " instance" : " instances"));
                }
                catch (const std::exception& e)
                {
                    showStatus(e.what(), true);
                }
            };
            addChildComponent(*element);
            elements_.push_back(std::move(element));
        }
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
}
void Editor::openInstance(size_t instance)
{
    chooseView(1);
    primitives_.selectRow(static_cast<int>(instance) + 1);
}
std::vector<double> Editor::currentLayerValues() const
{
    // The coordinates the user edits: authored values, with the eight host controls of layer 0 as automated.
    std::vector<double> values;
    const auto layers = sound_.value("control_layers", nlohmann::json::array());
    const auto layer = static_cast<size_t>(highLayer_);
    if (layer >= layers.size())
        return values;
    values = layers[layer].at("values").get<std::vector<double>>();
    if (layer == 0)
        for (size_t i = 0; i < values.size() && i < 8; ++i)
            if (auto* raw =
                    processor_.parameters().getRawParameterValue("x" + juce::String(static_cast<int>(i))))
                values[i] = raw->load();
    return values;
}
void Editor::refreshSpace()
{
    juce::StringArray names;
    const auto layers = sound_.value("control_layers", nlohmann::json::array());
    const auto layer = static_cast<size_t>(highLayer_);
    if (layer < layers.size())
        for (size_t i = 0; i < layers[layer].at("values").size(); ++i)
            names.add(controlName(layer, i));
    try
    {
        space_.setProjection(projectControls(sound_, layer, [this](size_t p) { return parameterName(p); }),
                             names);
    }
    catch (const std::exception& e)
    {
        space_.setProjection({}, {});
        showStatus(e.what(), true);
    }
    space_.setValues(currentLayerValues());
}
void Editor::applySpace(const std::vector<double>& values, bool final)
{
    const auto layer = static_cast<size_t>(space_.projection().layer);
    const auto layers = sound_.value("control_layers", nlohmann::json::array());
    if (layer >= layers.size() || values.size() != layers[layer].at("values").size())
        return;
    // Host controls follow the drag; deeper coordinates are written once, when the drag ends.
    bool deeper = false;
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (layer == 0 && i < 8)
        {
            if (auto* parameter =
                    processor_.parameters().getParameter("x" + juce::String(static_cast<int>(i))))
            {
                if (!spaceDragging_)
                    parameter->beginChangeGesture();
                parameter->setValueNotifyingHost(parameter->convertTo0to1(static_cast<float>(values[i])));
                if (final)
                    parameter->endChangeGesture();
            }
        }
        else if (values[i] != layers[layer].at("values").at(i).get<double>())
            deeper = true;
    }
    spaceDragging_ = !final;
    space_.setValues(values);
    if (final && deeper)
    {
        try
        {
            auto data = nlohmann::json::parse(processor_.document().toStdString());
            auto& stored = data.at("control_layers").at(layer).at("values");
            for (size_t i = layer == 0 ? 8 : 0; i < values.size(); ++i)
                stored.at(i) = values[i];
            juce::String error;
            changeDocument(std::move(data), error);
        }
        catch (const std::exception& e)
        {
            showStatus(e.what(), true);
        }
    }
}
bool Editor::telemetryFresh() const
{
    // Live values describe the loaded sound once a block has run since it was published.
    return processor_.telemetryBlocks.load(std::memory_order_acquire) > telemetryMark_ + 1;
}
std::vector<float> Editor::liveLayerValues() const
{
    std::vector<float> result;
    size_t count = 0;
    for (const auto& layer : sound_.value("control_layers", nlohmann::json::array()))
        count += layer.at("values").size();
    count = juce::jmin(count, processor_.liveLayerValues.size());
    for (size_t i = 0; i < count; ++i)
        result.push_back(processor_.liveLayerValues[i].load(std::memory_order_relaxed));
    return result;
}
void Editor::setCoefficient()
{
    try
    {
        const auto text = matrixValue_.getText().trim().toStdString();
        size_t used = 0;
        const auto value = std::stod(text, &used);
        if (used != text.size() || !std::isfinite(value))
            throw std::invalid_argument("Enter a finite coefficient");
        auto data = nlohmann::json::parse(processor_.document().toStdString());
        auto& matrix =
            data.at("control_layers").at(static_cast<size_t>(mappingLayer_.getSelectedId() - 1)).at("matrix");
        const auto columns = matrix.at("shape").at(1).get<size_t>();
        const auto offset =
            static_cast<size_t>(matrixDestination_) * columns + static_cast<size_t>(matrixSource_);
        matrix.at("values").at(offset) = value;
        juce::String error;
        if (changeDocument(std::move(data), error))
            showStatus("Layer coefficient updated");
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
}
void Editor::timerCallback()
{
    precision_.setText(processor_.scalarBytes() == 8 ? "Uniform DSP precision: float64"
                                                     : "Uniform DSP precision: float32",
                       juce::dontSendNotification);
    if (processor_.programRevision.load() != revision_)
        refreshSound();
    syncMode();
    undo_.setEnabled(processor_.undoManager().canUndo());
    redo_.setEnabled(processor_.undoManager().canRedo());
    scope_.repaint();
    meters_.repaint();
    exploration_.repaint();
    const auto fresh = telemetryFresh();
    if (view_ == 0)
    {
        std::vector<float> parameters, levels;
        if (fresh)
        {
            const auto count = juce::jmin(processor_.liveParameters.size(),
                                          sound_.at("tensors").at("parameters").at("values").size());
            for (size_t i = 0; i < count; ++i)
                parameters.push_back(processor_.liveParameters[i].load(std::memory_order_relaxed));
            const auto instances = juce::jmin(processor_.instanceLevels.size(),
                                              sound_.value("instances", nlohmann::json::array()).size());
            for (size_t i = 0; i < instances; ++i)
                levels.push_back(processor_.instanceLevels[i].load(std::memory_order_relaxed));
        }
        const auto rate = processor_.getSampleRate();
        for (auto& element : elements_)
            element->setLive(parameters, levels, rate);
        if (!spaceDragging_)
            space_.setValues(currentLayerValues());
        std::vector<std::pair<juce::String, std::vector<double>>> markers;
        if (highLayer_ == 0)
            for (size_t k = 0; k < snapshots_.size(); ++k)
                if (snapshots_[k].valid && selectedSnapshot_ != static_cast<int>(k))
                {
                    auto values = currentLayerValues();
                    for (size_t i = 0; i < values.size() && i < 8; ++i)
                        values[i] = snapshots_[k].controls[i];
                    markers.push_back({k == 0 ? "A" : "B", std::move(values)});
                }
        space_.setMarkers(std::move(markers));
    }
    if (view_ == 4 && fresh)
        layerState_.setValues(liveLayerValues());
    if (view_ == 1)
    {
        primitives_.repaint();
        for (size_t i = 0; i < parameterRows_.size() && i < parameterRowIndex_.size(); ++i)
        {
            const auto index = parameterRowIndex_[i];
            parameterRows_[i]->setLive(
                fresh && index >= 0 && static_cast<size_t>(index) < processor_.liveParameters.size()
                    ? std::optional<double>(processor_.liveParameters[static_cast<size_t>(index)].load(
                          std::memory_order_relaxed))
                    : std::nullopt,
                parameterLiveColour_[i]);
        }
    }
    if (view_ == 2 && fresh)
    {
        const auto layers = sound_.value("control_layers", nlohmann::json::array());
        const auto layer = static_cast<size_t>(juce::jmax(0, mappingLayer_.getSelectedId() - 1));
        const auto values = liveLayerValues();
        std::vector<float> fractions;
        size_t offset = 0;
        for (size_t l = 0; l < layer && l < layers.size(); ++l)
            offset += layers[l].at("values").size();
        if (layer < layers.size())
            for (size_t i = 0; i < layers[layer].at("values").size() && offset + i < values.size(); ++i)
            {
                const auto& range = layers[layer].at("ranges").at(i);
                const auto low = range.at(0).get<double>(), high = range.at(1).get<double>();
                fractions.push_back(high > low ? static_cast<float>((values[offset + i] - low) / (high - low))
                                               : 0.0f);
            }
        matrix_.setLive(std::move(fractions));
    }
    // The flow line carries notes and timing; this label only appears when updates were rejected.
    if (const auto failures = processor_.failures.load(); failures > 0)
    {
        telemetry_.setColour(juce::Label::textColourId, look_.theme().warn);
        telemetry_.setText(juce::String(static_cast<juce::int64>(failures)) + " rejected",
                           juce::dontSendNotification);
        telemetry_.setTooltip(juce::String(static_cast<juce::int64>(failures)) +
                              " runtime/model updates rejected; DSP retains valid controls");
    }
    repaint(footer_);
}
} // namespace alfa::tensorhost
