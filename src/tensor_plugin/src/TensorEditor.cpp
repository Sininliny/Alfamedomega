#include "TensorEditor.h"

#include "CompositionPanel.h"
#include "FactoryPrograms.h"
#include "SoundLibraryPanel.h"
#include "alfa/tensor/Universe.h"

#include <algorithm>
#include <cmath>

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
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 82, 22);
        slider.textFromValueFunction = [](double v)
        {
            return scalarText(v);
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
        auto r = getLocalBounds();
        label.setBounds(r.removeFromTop(22));
        slider.setBounds(r);
    }

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
{
    setLookAndFeel(&look_);
    setResizable(true, true);
    setResizeLimits(1040, 700, 1920, 1200);
    const auto saved = p.uiSettings();
    view_ = juce::jlimit(0, 3, static_cast<int>(saved.getProperty("view", 0)));
    setSize(static_cast<int>(saved.getProperty("width", 1280)),
            static_cast<int>(saved.getProperty("height", 800)));
    brand_.setText("ALFAMEDOMEGA", juce::dontSendNotification);
    brand_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(19, true));
    version_.setText("AUDIO TENSOR CONSOLE", juce::dontSendNotification);
    version_.setFont(alfa::ui::AlfaLookAndFeel::monoFont(10));
    for (auto* c : std::initializer_list<juce::Component*>{
             &brand_, &version_, &status_, &telemetry_, &configurationName_, &library_, &savePreset_,
             &openPreset_, &panic_, &bypass_, &a_, &b_, &capture_, &undo_, &redo_, &keyboard_, &meters_})
        addAndMakeVisible(c);
    const std::array<juce::String, 4> views{"Control layers", "Instances", "Tensor space", "Settings"};
    for (size_t i = 0; i < tabs_.size(); ++i)
    {
        addAndMakeVisible(tabs_[i]);
        tabs_[i].setButtonText(views[i]);
        tabs_[i].setClickingTogglesState(true);
        tabs_[i].onClick = [this, i]
        {
            chooseView(static_cast<int>(i));
        };
    }
    mode_.addItemList({"MIDI input only", "Audio input only", "Audio + MIDI inputs"}, 1);
    addChildComponent(mode_);
    mode_.setTooltip(
        "Native input availability and dry mix policy. This changes execution, not the inspected component.");
    modeAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(p.parameters(),
                                                                                               "mode", mode_);
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
    library_.setTooltip("Browse saved configurations without changing the current sound");
    configurationName_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(14));
    configurationName_.setTooltip(
        "Name of the current executable configuration; viewing it changes no sound");
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
        processor_.uiSettings().setProperty("xy_x", xAxis_.getSelectedId() - 1, nullptr);
        exploration_.repaint();
    };
    yAxis_.onChange = [this]
    {
        exploration_.yId = "x" + juce::String(yAxis_.getSelectedId() - 1);
        processor_.uiSettings().setProperty("xy_y", yAxis_.getSelectedId() - 1, nullptr);
        exploration_.repaint();
    };
    exploration_.xId = "x" + juce::String(xAxis_.getSelectedId() - 1);
    exploration_.yId = "x" + juce::String(yAxis_.getSelectedId() - 1);
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
    layerViewport_.setViewedComponent(&layerContent_, false);
    for (auto* c : std::initializer_list<juce::Component*>{&scope_,
                                                           &exploration_,
                                                           &xAxis_,
                                                           &yAxis_,
                                                           &layerSelect_,
                                                           &pagePrevious_,
                                                           &pageNext_,
                                                           &primitives_,
                                                           &parameterViewport_,
                                                           &routingViewport_,
                                                           &layerViewport_,
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
                                                           &mappingSource_,
                                                           &mappingDestination_,
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
                                                           &tier_,
                                                           &precision_,
                                                           &resources_,
                                                           &hostPrecision_,
                                                           &importTheme_})
        addChildComponent(c);
    layerSelect_.onChange = [this]
    {
        highLayer_ = juce::jmax(0, layerSelect_.getSelectedId() - 1);
        highPage_ = 0;
        rebuildHighControls();
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
    mappingSource_.onChange = [this]
    {
        refreshTensor();
    };
    mappingDestination_.onChange = [this]
    {
        refreshTensor();
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
    tier_.addItemList({"Lite", "Standard", "Pro", "Studio"}, 1);
    tier_.onChange = [this]
    {
        if (refreshing_)
            return;
        const juce::StringArray tierIds{"Lite", "Standard", "Pro", "Studio"};
        juce::String error;
        if (processor_.setTier(tierIds[juce::jlimit(0, 3, tier_.getSelectedId() - 1)], error))
        {
            refreshSound();
            showStatus("Tier prepared; tensor controls and parameter locks retained");
        }
        else
        {
            showStatus(error, true);
            refreshSound();
        }
    };
    precision_.setFont(alfa::ui::AlfaLookAndFeel::monoFont(12));
    resources_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
    hostPrecision_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
    resources_.setText(
        "Lite / Standard: float32. Pro / Studio: float64.\nImport four explicit resource profiles to enable "
        "capacity conversion.\nChanged candidates require sound, pitch and timing checks.\nWithout profiles, "
        "tier selection changes precision only.",
        juce::dontSendNotification);
    hostPrecision_.setText("Highest 8 host automation ports use float32.\nManual tensor and deeper-layer "
                           "editing preserve double values.",
                           juce::dontSendNotification);
    addAndMakeVisible(importTiers_);
    importTiers_.setComponentID("tier.import_profiles");
    importTiers_.onClick = [this]
    {
        chooser_ =
            std::make_unique<juce::FileChooser>("Import explicit tier profiles", juce::File{}, "*.json");
        auto safe = juce::Component::SafePointer<Editor>(this);
        chooser_->launchAsync(
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [safe](const juce::FileChooser& chooser)
            {
                if (!safe || !chooser.getResult().existsAsFile())
                    return;
                try
                {
                    const auto profiles =
                        nlohmann::json::parse(chooser.getResult().loadFileAsString().toStdString());
                    auto sound = nlohmann::json::parse(safe->processor_.document().toStdString());
                    for (const auto* tier : {"Lite", "Standard", "Pro", "Studio"})
                        if (!profiles.contains(tier) || !profiles.at(tier).at("budget").is_object())
                            throw std::invalid_argument("Four named tier budgets are required");
                    sound["tier_profiles"] = profiles;
                    juce::String error;
                    if (!safe->changeDocument(std::move(sound), error))
                        safe->showStatus(error, true);
                    else
                    {
                        safe->refreshSound();
                        safe->showStatus(
                            "Explicit tier budgets imported; conversion validates each candidate");
                    }
                }
                catch (const std::exception& error)
                {
                    safe->showStatus(error.what(), true);
                }
            });
    };
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
    layerViewport_.setViewedComponent(nullptr, false);
    setLookAndFeel(nullptr);
}
void Editor::paint(juce::Graphics& g)
{
    const auto& t = look_.theme();
    g.fillAll(t.bg);
    g.setColour(t.line);
    g.drawHorizontalLine(65, 18.0f, static_cast<float>(getWidth() - 18));
    g.drawHorizontalLine(119, 18.0f, static_cast<float>(getWidth() - 18));
    if (view_ == 0)
    {
        surface(g, t, highArea_, "CONTROL LAYERS", "X / MANY-TO-MANY");
        surface(g, t, xyArea_, "CONTROL COORDINATES", "EXPLICIT AXES");
    }
    if (view_ == 1)
    {
        surface(g, t, primitiveArea_, "PRIMITIVE ACTIVATIONS");
        surface(g, t, parameterArea_, "PARAMETER BASE VALUES");
        surface(g, t, routingArea_, "ROUTING / MODULATION");
    }
    if (view_ == 2)
    {
        surface(g, t, layerArea_, "LAYERED CONTROL MATRIX");
        surface(g, t, tensorArea_, "AUDIO TENSOR SPACE");
    }
    if (view_ == 3)
    {
        surface(g, t, settingsArea_, "INSTANCE SETTINGS", "SEPARATE FROM SOUND");
        g.setColour(t.text2);
        g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13));
        auto r = settingsArea_.reduced(24, 48);
        g.drawText("Theme", r.removeFromTop(22), juce::Justification::centredLeft);
        r.removeFromTop(46);
        g.drawText("Preset library", r.removeFromTop(22), juce::Justification::centredLeft);
        r.removeFromTop(46);
        g.drawText(store_ ? store_->directory().getFullPathName() : juce::String{},
                   r.removeFromTop(32).withWidth(470), juce::Justification::centredLeft);
        r.removeFromTop(30);
        g.drawText("Accent colour", settingsArea_.getX() + 24, settingsArea_.getY() + 232, 220, 20,
                   juce::Justification::centredLeft);
        g.drawText("Knob drag", settingsArea_.getX() + 260, settingsArea_.getY() + 232, 220, 20,
                   juce::Justification::centredLeft);
        g.drawText("Native input / dry mix policy", settingsArea_.getX() + 24, settingsArea_.getY() + 174,
                   400, 20, juce::Justification::centredLeft);
        g.drawText("Execution tier / precision", settingsArea_.getX() + 520, settingsArea_.getY() + 120,
                   juce::jmax(220, settingsArea_.getWidth() - 544), 20, juce::Justification::centredLeft);
        g.drawFittedText("Interface settings stay separate from sounds.\nTraining is paused. Device settings "
                         "are in your host / Options.",
                         settingsArea_.getX() + 24, settingsArea_.getY() + 300, 470, 64,
                         juce::Justification::centredLeft, 3);
    }
    auto strip = juce::Rectangle<int>(18, workspace_.getBottom() + 10, getWidth() - 36, 88);
    surface(g, t, strip, "");
    auto flow = strip.reduced(20, 30);
    flow.removeFromLeft(300);
    flow.removeFromRight(140);
    g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(11));
    g.setColour(t.text2);
    g.drawText("INPUT TENSORS  >  X LAYERS  >  RECURRENT DSP STATE  >  AUDIO", flow.removeFromTop(22),
               juce::Justification::centred);
    g.setColour(t.mod);
    g.drawText(juce::String(processor_.activeNotes.load()) + " NOTES  /  " +
                   juce::String(processor_.getSampleRate(), 0) + " Hz  /  " +
                   juce::String(processor_.blockMilliseconds.load(), 2) + " ms",
               flow, juce::Justification::centred);
}
void Editor::resized()
{
    if (getWidth() < 1)
        return;
    auto top = juce::Rectangle<int>(18, 12, getWidth() - 36, 44);
    auto brand = top.removeFromLeft(200);
    brand_.setBounds(brand.removeFromTop(25));
    version_.setBounds(brand);
    for (auto& tab : tabs_)
        tab.setBounds(top.removeFromLeft(104).reduced(3, 4));
    bypass_.setBounds(top.removeFromLeft(78).reduced(2, 6));
    panic_.setBounds(top.removeFromLeft(68).reduced(2, 6));
    telemetry_.setBounds(top.reduced(6));
    auto browser = juce::Rectangle<int>(18, 76, getWidth() - 36, 34);
    library_.setBounds(browser.removeFromLeft(112).reduced(2, 0));
    redo_.setBounds(browser.removeFromRight(48).reduced(2, 0));
    undo_.setBounds(browser.removeFromRight(48).reduced(2, 0));
    capture_.setBounds(browser.removeFromRight(76).reduced(2, 0));
    b_.setBounds(browser.removeFromRight(32).reduced(2, 0));
    a_.setBounds(browser.removeFromRight(32).reduced(2, 0));
    openPreset_.setBounds(browser.removeFromRight(64).reduced(2, 0));
    savePreset_.setBounds(browser.removeFromRight(80).reduced(2, 0));
    configurationName_.setBounds(browser.reduced(8, 0));
    const auto keyboardHeight = showKeyboard_.getToggleState() ? 70 : 0;
    keyboard_.setVisible(keyboardHeight != 0);
    keyboard_.setBounds(18, getHeight() - keyboardHeight - 24, getWidth() - 36, keyboardHeight);
    status_.setBounds(20, getHeight() - 22, getWidth() - 40, 20);
    workspace_ = {18, 132, getWidth() - 36, getHeight() - 132 - keyboardHeight - 124};
    const auto rightWidth = juce::jmax(280, workspace_.getWidth() / 3);
    highArea_ = workspace_;
    auto xy = highArea_.removeFromRight(rightWidth);
    highArea_.removeFromRight(12);
    xyArea_ = xy;
    scopeArea_ = highArea_;
    highArea_.setHeight(240);
    scopeArea_.removeFromTop(252);
    auto high = highArea_.reduced(14);
    high.removeFromTop(26);
    auto highToolbar = high.removeFromTop(28);
    layerSelect_.setBounds(highToolbar.removeFromLeft(180));
    pagePrevious_.setBounds(highToolbar.removeFromRight(30));
    pageNext_.setBounds(highToolbar.removeFromRight(30));
    levelHeader_.setBounds(highToolbar.reduced(8, 0));
    const auto highWidth = high.getWidth() / 8;
    for (size_t i = 0; i < highControls_.size(); ++i)
        highControls_[i]->setBounds(high.removeFromLeft(highWidth).reduced(3, 4));
    auto xp = xyArea_.reduced(14);
    xp.removeFromTop(30);
    auto axes = xp.removeFromTop(30);
    xAxis_.setBounds(axes.removeFromLeft(axes.getWidth() / 2).reduced(2, 0));
    yAxis_.setBounds(axes.reduced(2, 0));
    exploration_.setBounds(xp.reduced(0, 10));
    scope_.setBounds(scopeArea_);
    primitiveArea_ = workspace_;
    parameterArea_ = primitiveArea_;
    primitiveArea_.setWidth(224);
    parameterArea_.removeFromLeft(236);
    routingArea_ = parameterArea_.removeFromRight(juce::jmax(320, parameterArea_.getWidth() / 3));
    parameterArea_.removeFromRight(12);
    primitives_.setBounds(primitiveArea_.reduced(10).withTrimmedTop(28));
    auto params = parameterArea_.reduced(10).withTrimmedTop(26);
    auto paramTitle = params.removeFromTop(25);
    paramHeader_.setBounds(paramTitle);
    auto rawButtons = params.removeFromBottom(32);
    rawPrevious_.setBounds(rawButtons.removeFromLeft(rawButtons.getWidth() / 2).reduced(2));
    rawNext_.setBounds(rawButtons.reduced(2));
    parameterViewport_.setBounds(params);
    parameterContent_.setSize(juce::jmax(1, parameterViewport_.getWidth() - 14),
                              static_cast<int>(parameterRows_.size()) * 48);
    for (size_t i = 0; i < parameterRows_.size(); ++i)
        parameterRows_[i]->setBounds(0, static_cast<int>(i) * 48, parameterContent_.getWidth(), 48);
    auto routes = routingArea_.reduced(10).withTrimmedTop(26);
    routeHeader_.setBounds(routes.removeFromTop(25));
    auto controls = routes.removeFromBottom(166);
    auto routeRow = controls.removeFromTop(30);
    sourceKind_.setBounds(routeRow.removeFromLeft(routeRow.getWidth() * 2 / 3).reduced(2));
    sourceElement_.setBounds(routeRow.reduced(2));
    routeDestination_.setBounds(controls.removeFromTop(30).reduced(2));
    routeRow = controls.removeFromTop(30);
    routeWeightsTensor_.setBounds(routeRow.removeFromLeft(routeRow.getWidth() / 2).reduced(2));
    routeWeight_.setBounds(routeRow.reduced(2));
    addRoute_.setBounds(controls.removeFromTop(34).reduced(2));
    removeRoute_.setBounds(controls.removeFromTop(32).reduced(2));
    routingViewport_.setBounds(routes);
    routingContent_.setSize(juce::jmax(1, routingViewport_.getWidth() - 14),
                            static_cast<int>(routeRows_.size()) * 52);
    for (size_t i = 0; i < routeRows_.size(); ++i)
        routeRows_[i]->setBounds(0, static_cast<int>(i) * 52, routingContent_.getWidth(), 52);
    layerArea_ = workspace_;
    tensorArea_ = layerArea_.removeFromRight(workspace_.getWidth() * 3 / 5);
    layerArea_.removeFromRight(12);
    auto layers = layerArea_.reduced(12).withTrimmedTop(28);
    mappingLayer_.setBounds(layers.removeFromTop(30));
    auto mapForm = layers.removeFromTop(100);
    auto mapRow = mapForm.removeFromTop(32);
    mappingSource_.setBounds(mapRow.removeFromLeft(mapRow.getWidth() / 2).reduced(2));
    mappingDestination_.setBounds(mapRow.reduced(2));
    mapRow = mapForm.removeFromTop(32);
    matrixValue_.setBounds(mapRow.removeFromLeft(mapRow.getWidth() / 2).reduced(2));
    setMatrix_.setBounds(mapRow.reduced(2));
    layerViewport_.setBounds(layers);
    layerContent_.setSize(juce::jmax(1, layerViewport_.getWidth() - 14),
                          static_cast<int>(layerRows_.size()) * 48);
    for (size_t i = 0; i < layerRows_.size(); ++i)
        layerRows_[i]->setBounds(0, static_cast<int>(i) * 48, layerContent_.getWidth(), 48);
    auto tensor = tensorArea_.reduced(12).withTrimmedTop(28);
    auto tensorTools = tensor.removeFromTop(30);
    tensorSelect_.setBounds(tensorTools.removeFromLeft(190));
    importModel_.setBounds(tensorTools.removeFromRight(150).reduced(2, 0));
    exportJSON_.setBounds(tensorTools.removeFromRight(96).reduced(2, 0));
    apply_.setBounds(tensorTools.reduced(2, 0));
    tensorValues_.setBounds(tensor.removeFromTop(100).reduced(0, 6));
    compose_.setBounds(tensor.removeFromTop(30).reduced(0, 2));
    document_.setBounds(tensor.removeFromTop(tensor.getHeight() * 3 / 5).reduced(0, 5));
    inspection_.setBounds(tensor.reduced(0, 5));
    settingsArea_ = workspace_;
    const auto precisionX = settingsArea_.getX() + 520,
               precisionWidth = juce::jmax(220, settingsArea_.getWidth() - 544);
    tier_.setBounds(precisionX, settingsArea_.getY() + 144, 200, 32);
    precision_.setBounds(precisionX, settingsArea_.getY() + 184, precisionWidth, 28);
    resources_.setBounds(precisionX, settingsArea_.getY() + 214, precisionWidth, 90);
    hostPrecision_.setBounds(precisionX, settingsArea_.getY() + 310, precisionWidth, 46);
    theme_.setBounds(settingsArea_.getX() + 24, settingsArea_.getY() + 74, 220, 32);
    exportTheme_.setBounds(settingsArea_.getX() + 260, settingsArea_.getY() + 74, 150, 32);
    importTheme_.setBounds(settingsArea_.getX() + 424, settingsArea_.getY() + 74, 150, 32);
    showKeyboard_.setBounds(settingsArea_.getX() + 594, settingsArea_.getY() + 74, 150, 32);
    accent_.setBounds(settingsArea_.getX() + 24, settingsArea_.getY() + 254, 220, 32);
    gesture_.setBounds(settingsArea_.getX() + 260, settingsArea_.getY() + 254, 220, 32);
    folder_.setBounds(settingsArea_.getX() + 24, settingsArea_.getY() + 144, 220, 32);
    importTiers_.setBounds(settingsArea_.getX() + 440, settingsArea_.getY() + 144, 200, 32);
    refresh_.setBounds(settingsArea_.getX() + 260, settingsArea_.getY() + 144, 150, 32);
    mode_.setBounds(settingsArea_.getX() + 24, settingsArea_.getY() + 196, 330, 32);
    const auto bottomY = workspace_.getBottom() + 24;
    if (inputGain_)
        inputGain_->setBounds(34, bottomY, 90, 72);
    if (mix_)
        mix_->setBounds(132, bottomY, 90, 72);
    if (outputGain_)
        outputGain_->setBounds(230, bottomY, 90, 72);
    meters_.setBounds(getWidth() - 150, bottomY - 6, 114, 74);
    processor_.uiSettings().setProperty("width", getWidth(), nullptr);
    processor_.uiSettings().setProperty("height", getHeight(), nullptr);
}
void Editor::chooseView(int index)
{
    view_ = juce::jlimit(0, 3, index);
    importTiers_.setVisible(view_ == 3);
    for (size_t i = 0; i < tabs_.size(); ++i)
        tabs_[i].setToggleState(static_cast<int>(i) == view_, juce::dontSendNotification);
    for (auto* c :
         std::initializer_list<juce::Component*>{&scope_, &exploration_, &xAxis_, &yAxis_, &layerSelect_,
                                                 &pagePrevious_, &pageNext_, &levelHeader_})
        c->setVisible(view_ == 0);
    for (auto& knob : highControls_)
        knob->setVisible(view_ == 0);
    for (auto* c : std::initializer_list<juce::Component*>{
             &primitives_, &parameterViewport_, &routingViewport_, &paramHeader_, &routeHeader_, &addRoute_,
             &removeRoute_, &sourceKind_, &sourceElement_, &routeDestination_, &routeWeight_,
             &routeWeightsTensor_, &rawPrevious_, &rawNext_})
        c->setVisible(view_ == 1);
    for (auto* c : std::initializer_list<juce::Component*>{
             &layerViewport_, &mappingLayer_, &mappingSource_, &mappingDestination_, &matrixValue_,
             &setMatrix_, &document_, &inspection_, &tensorValues_, &tensorSelect_, &apply_, &exportJSON_,
             &importModel_})
        c->setVisible(view_ == 2);
    compose_.setVisible(view_ == 2);
    for (auto* c : std::initializer_list<juce::Component*>{
             &theme_, &refresh_, &folder_, &exportTheme_, &importTheme_, &accent_, &gesture_, &tier_,
             &precision_, &resources_, &hostPrecision_, &showKeyboard_, &mode_})
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
    repaint();
}
void Editor::refreshLibrary()
{
    juce::String error;
    store_->refresh(error);
    configurationName_.setText("Configuration: " + PresetStore::presetName(processor_.document()),
                               juce::dontSendNotification);
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
        refreshing_ = true;
        const juce::StringArray tierIds{"Lite", "Standard", "Pro", "Studio"};
        tier_.setSelectedId(
            juce::jmax(0, tierIds.indexOf(js(sound_.value("tier", std::string("Standard"))))) + 1,
            juce::dontSendNotification);
        refreshing_ = false;
        precision_.setText(processor_.scalarBytes() == 8 ? "Uniform DSP precision: float64"
                                                         : "Uniform DSP precision: float32",
                           juce::dontSendNotification);
        revision_ = processor_.programRevision.load();
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
            const auto title = "X[0," + juce::String(static_cast<int>(i)) + "]";
            xAxis_.addItem(title, static_cast<int>(i) + 1);
            yAxis_.addItem(title, static_cast<int>(i) + 1);
        }
        const auto capacity = static_cast<int>(juce::jmin(size_t(8), highest));
        xAxis_.setSelectedId(capacity ? juce::jlimit(1, capacity, previousX) : 0, juce::dontSendNotification);
        yAxis_.setSelectedId(capacity ? juce::jlimit(1, capacity, previousY) : 0, juce::dontSendNotification);
        exploration_.xId = "x" + juce::String(juce::jmax(0, xAxis_.getSelectedId() - 1));
        exploration_.yId = "x" + juce::String(juce::jmax(0, yAxis_.getSelectedId() - 1));
        refreshing_ = false;
        rebuildParameters();
        rebuildRoutes();
        rebuildLayers();
        rebuildHighControls();
        refreshTensor();
        configurationName_.setText("Configuration: " + PresetStore::presetName(processor_.document()),
                                   juce::dontSendNotification);
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
    const auto& t = look_.theme();
    if (selected)
    {
        g.setColour(t.accent.withAlpha(0.12f));
        g.fillRoundedRectangle(
            juce::Rectangle<float>(2, 2, static_cast<float>(width - 4), static_cast<float>(height - 4)), 4);
    }
    g.setColour(selected ? t.accent : t.text2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12, row == 0));
    if (row == 0)
    {
        g.drawText("All raw parameters", 8, 0, width - 16, height, juce::Justification::centredLeft);
        return;
    }
    try
    {
        const auto& instance = sound_.at("instances").at(static_cast<size_t>(row - 1));
        const auto id = js(instance.at("primitive").get<std::string>());
        const auto index = instance.at("activation").get<size_t>();
        const auto activation = sound_.at("tensors").at("activation").at("values").at(index).get<double>();
        g.drawText(juce::String(row - 1) + "  " + id, 8, 4, width - 16, height - 15,
                   juce::Justification::centredLeft);
        g.setColour(t.line);
        g.fillRect(8, height - 8, width - 52, 2);
        g.setColour(t.mod);
        g.fillRect(8, height - 8, static_cast<int>((width - 52) * juce::jlimit(0.0, 1.0, activation)), 2);
        g.setFont(alfa::ui::AlfaLookAndFeel::monoFont(10));
        g.drawText(juce::String(activation, 2), width - 42, 4, 34, height - 8,
                   juce::Justification::centredRight);
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
}
void Editor::rebuildParameters()
{
    parameterRows_.clear();
    try
    {
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
        }
        rawPage_ = juce::jlimit(0, juce::jmax(0, static_cast<int>((count + 63) / 64) - 1), rawPage_);
        const auto begin = start + static_cast<size_t>(rawPage_) * 64;
        const auto end = juce::jmin(start + count, begin + 64);
        const auto locks = sound_.value("parameter_locks", nlohmann::json::array());
        for (size_t index = begin; index < end && index < values.size(); ++index)
        {
            const auto locked = index < locks.size() && locks[index].get<int>() != 0;
            auto row = std::make_unique<NumericRow>(
                parameterLabel(index), parameterUnit(index), values[index].get<double>(),
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
        routeHeader_.setText(juce::String(static_cast<int>(routes.size())) + " explicit relationships",
                             juce::dontSendNotification);
        for (size_t i = 0; i < routes.size(); ++i)
        {
            const auto& route = routes[i];
            const auto tensorName = js(route.at("weights").get<std::string>());
            const auto weight = route.at("weight_element").get<size_t>();
            const auto target = route.at("parameter_element").get<size_t>();
            auto row = std::make_unique<NumericRow>(
                sourceLabel(route) + " > " + parameterLabel(target), tensorName,
                sound_.at("tensors").at(string(tensorName)).at("values").at(weight).get<double>(),
                [this, tensorName, weight](double v, juce::String& e)
                { return changeTensor(tensorName, weight, v, e); });
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
    layerRows_.clear();
    refreshing_ = true;
    try
    {
        const auto layers = sound_.value("control_layers", nlohmann::json::array());
        const auto layer = juce::jmax(0, mappingLayer_.getSelectedId() - 1),
                   previousSource = mappingSource_.getSelectedId(),
                   previousDestination = mappingDestination_.getSelectedId();
        mappingSource_.clear(juce::dontSendNotification);
        mappingDestination_.clear(juce::dontSendNotification);
        if (static_cast<size_t>(layer) < layers.size())
        {
            const auto& current = layers[static_cast<size_t>(layer)];
            const auto& values = current.at("values");
            const auto& ranges = current.at("ranges");
            for (size_t index = 0; index < values.size(); ++index)
            {
                const auto low = ranges.at(index).at(0).get<double>(),
                           high = ranges.at(index).at(1).get<double>();
                auto row = std::make_unique<NumericRow>(
                    "X[" + juce::String(layer) + "," + juce::String(static_cast<int>(index)) + "]", "value",
                    values[index].get<double>(), [this, layer, index](double value, juce::String& error)
                    { return processor_.editLayerValue(static_cast<size_t>(layer), index, value, error); },
                    std::function<void(bool)>{}, false, std::array<double, 3>{low, high, 0});
                layerContent_.addAndMakeVisible(*row);
                layerRows_.push_back(std::move(row));
                mappingSource_.addItem("X[" + juce::String(layer) + "," +
                                           juce::String(static_cast<int>(index)) + "]",
                                       static_cast<int>(index) + 1);
            }
            const auto destinations = current.at("matrix").at("shape").at(0).get<size_t>();
            for (size_t i = 0; i < destinations; ++i)
                mappingDestination_.addItem(static_cast<size_t>(layer + 1) < layers.size()
                                                ? "X[" + juce::String(layer + 1) + "," +
                                                      juce::String(static_cast<int>(i)) + "]"
                                                : parameterLabel(i),
                                            static_cast<int>(i) + 1);
            mappingSource_.setSelectedId(
                values.empty() ? 0 : juce::jlimit(1, static_cast<int>(values.size()), previousSource),
                juce::dontSendNotification);
            mappingDestination_.setSelectedId(
                destinations ? juce::jlimit(1, static_cast<int>(destinations), previousDestination) : 0,
                juce::dontSendNotification);
        }
        setMatrix_.setEnabled(mappingSource_.getSelectedId() > 0 && mappingDestination_.getSelectedId() > 0);
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
    refreshing_ = false;
    refreshTensor();
    resized();
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
        highPage_ = juce::jlimit(0, juce::jmax(0, static_cast<int>((count + 7) / 8) - 1), highPage_);
        const auto begin = static_cast<size_t>(highPage_) * 8, end = juce::jmin(count, begin + 8);
        for (size_t index = begin; index < end; ++index)
        {
            std::unique_ptr<juce::Component> knob;
            const auto title =
                "X[" + juce::String(highLayer_) + "," + juce::String(static_cast<int>(index)) + "]";
            if (layer == 0 && index < 8)
                knob = std::make_unique<ParameterKnob>(processor_,
                                                       "x" + juce::String(static_cast<int>(index)), title);
            else
            {
                const auto& range = layers[layer].at("ranges").at(index);
                knob = std::make_unique<LayerKnob>(processor_, layer, index,
                                                   layers[layer].at("values").at(index).get<double>(),
                                                   range.at(0).get<double>(), range.at(1).get<double>());
            }
            addChildComponent(*knob);
            highControls_.push_back(std::move(knob));
        }
        levelHeader_.setText(count ? juce::String(static_cast<int>(begin)) + "-" +
                                         juce::String(static_cast<int>(end)) + " / " +
                                         juce::String(static_cast<int>(count)) + " controls"
                                   : "This sound has no higher control layer",
                             juce::dontSendNotification);
        pagePrevious_.setEnabled(highPage_ > 0);
        pageNext_.setEnabled(end < count);
        xAxis_.setEnabled(count > 0);
        yAxis_.setEnabled(count > 0);
        exploration_.setEnabled(count > 0);
        for (auto& control : highControls_)
            control->setVisible(view_ == 0);
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
        const auto layer = mappingLayer_.getSelectedId() - 1, source = mappingSource_.getSelectedId() - 1,
                   destination = mappingDestination_.getSelectedId() - 1;
        if (layer >= 0 && source >= 0 && destination >= 0 && static_cast<size_t>(layer) < layers.size())
        {
            const auto& matrix = layers[static_cast<size_t>(layer)].at("matrix");
            const auto columns = matrix.at("shape").at(1).get<size_t>();
            const auto value =
                matrix.at("values")
                    .at(static_cast<size_t>(destination) * columns + static_cast<size_t>(source))
                    .get<double>();
            matrixValue_.setText(scalarText(value), false);
        }
        inspection_.setText(processor_.inspection(), false);
    }
    catch (const std::exception& e)
    {
        showStatus(e.what(), true);
    }
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
        const auto offset = static_cast<size_t>(mappingDestination_.getSelectedId() - 1) * columns +
                            static_cast<size_t>(mappingSource_.getSelectedId() - 1);
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
    undo_.setEnabled(processor_.undoManager().canUndo());
    redo_.setEnabled(processor_.undoManager().canRedo());
    scope_.repaint();
    meters_.repaint();
    exploration_.repaint();
    telemetry_.setText(juce::String(processor_.activeNotes.load()) + " notes / " +
                           juce::String(processor_.blockMilliseconds.load(), 2) + " ms",
                       juce::dontSendNotification);
    if (processor_.failures.load() > 0)
        telemetry_.setTooltip(juce::String(static_cast<juce::int64>(processor_.failures.load())) +
                              " runtime/model updates rejected; DSP retains valid controls");
    repaint(18, workspace_.getBottom() + 10, getWidth() - 36, 88);
}
} // namespace alfa::tensorhost
