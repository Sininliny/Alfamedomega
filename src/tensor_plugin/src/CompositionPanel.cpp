#include "CompositionPanel.h"

#include "PrimitiveFragmentPanel.h"

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>

namespace alfa::tensorhost
{
namespace
{
using JSON = nlohmann::json;
size_t indexValue(const juce::TextEditor& control)
{
    const auto text = control.getText().trim().toStdString();
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Coordinates must be nonnegative integers");
    size_t used = 0;
    const auto value = std::stoull(text, &used);
    if (used != text.size() || value > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("Coordinate exceeds addressable capacity");
    return static_cast<size_t>(value);
}
} // namespace
CompositionPanel::CompositionPanel(const juce::String& sound, double sampleRate, Publish publish,
                                   const alfa::ui::Theme* theme)
    : sampleRate_(sampleRate)
    , publish_(std::move(publish))
    , currentSound_(sound)
{
    if (theme)
        look_.setTheme(*theme);
    setLookAndFeel(&look_);
    setSize(980, 700);
    fragment_.setComponentID("composition.fragment");
    source_.setComponentID("composition.source");
    destination_.setComponentID("composition.destination");
    sourceKind_.setComponentID("composition.source_kind");
    targetKind_.setComponentID("composition.target_kind");
    weights_.setComponentID("composition.weights");
    recipe_.setComponentID("composition.recipe");
    budget_.setComponentID("composition.budget");
    status_.setComponentID("composition.status");
    newId_.setComponentID("composition.new_id");
    bindingId_.setComponentID("composition.binding_id");
    sourceElement_.setComponentID("composition.source_element");
    instance_.setComponentID("composition.target_instance");
    targetElement_.setComponentID("composition.target_element");
    weight_.setComponentID("composition.weight");
    for (auto* component : std::initializer_list<juce::Component*>{
             &fragment_,  &source_,        &destination_, &sourceKind_,    &targetKind_, &weights_, &newId_,
             &bindingId_, &sourceElement_, &instance_,    &targetElement_, &weight_,     &budget_,  &recipe_,
             &help_,      &status_,        &import_,      &clone_,         &remove_,     &bind_,    &unbind_,
             &prepare_,   &apply_,         &base_,        &create_})
        addAndMakeVisible(component);
    help_.setText(
        "Compose executable fragments with explicit recurrent bindings. No automatic chain is created.\n"
        "Edit capacity budgets before preparing larger combinations; Apply publishes a validated sound with "
        "Undo.",
        juce::dontSendNotification);
    help_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13));
    newId_.setText("F1");
    newId_.setTooltip("Unique ID for the imported or cloned fragment");
    bindingId_.setText("B0");
    bindingId_.setTooltip("Unique binding ID; also used by Remove binding");
    sourceElement_.setText("0");
    sourceElement_.setTooltip("Source fragment's flattened output or state coordinate");
    instance_.setText("0");
    instance_.setTooltip("Destination fragment's local primitive instance for audio operands");
    targetElement_.setText("0");
    targetElement_.setTooltip("Destination audio operand channel or fragment parameter coordinate");
    weight_.setText("1");
    weight_.setTooltip("Explicit routing coefficient");
    sourceKind_.addItemList({"output_history", "state"}, 1);
    sourceKind_.setSelectedId(1, juce::dontSendNotification);
    targetKind_.addItemList({"audio", "parameter"}, 1);
    targetKind_.setSelectedId(1, juce::dontSendNotification);
    weights_.addItemList({"routing", "modulation"}, 1);
    weights_.setSelectedId(1, juce::dontSendNotification);
    for (auto* editor : {&budget_, &recipe_})
    {
        editor->setMultiLine(true);
        editor->setReturnKeyStartsNewLine(true);
        editor->setFont(alfa::ui::AlfaLookAndFeel::monoFont(12));
    }
    budget_.setTooltip(
        "Explicit capacity budget JSON. Preparing a recipe applies this budget, never prunes primitives.");
    recipe_.setTooltip(
        "Complete editable composition recipe including fragment tensors, assets and bindings");
    const auto data = JSON::parse(sound.toStdString());
    budget_.setText(data.at("budget").dump(2), false);
    composer_ = std::make_unique<TensorComposition>(sampleRate_, budget_.getText(),
                                                    data.value("tier", std::string{}));
    juce::String error;
    bool restored = false;
    if (data.contains("composition"))
    {
        restored = composer_->restoreDocument(sound, error);
        if (!restored)
        {
            needsBase_ = true;
            report("Sound changed since its recipe. Use edited sound as base to retain current values.",
                   true);
        }
    }
    if (!restored && !needsBase_)
    {
        auto base = data;
        base.erase("composition");
        if (!composer_->add("F0", base.dump(), error))
            report(error, true);
    }
    base_.setEnabled(needsBase_);
    base_.onClick = [this]
    {
        try
        {
            auto base = JSON::parse(currentSound_.toStdString());
            base.erase("composition");
            juce::String error;
            auto candidate = std::make_unique<TensorComposition>(sampleRate_, budget_.getText(),
                                                                 base.value("tier", std::string{}));
            if (candidate->add("F0", base.dump(), error))
            {
                composer_ = std::move(candidate);
                needsBase_ = false;
                base_.setEnabled(false);
                refresh();
                report("Current edited sound retained as F0");
            }
            else
                report(error, true);
        }
        catch (const std::exception& error)
        {
            report(error.what(), true);
        }
    };
    refresh();
    import_.onClick = [this]
    {
        importFragment();
    };
    create_.onClick = [this]
    {
        createFragment();
    };
    clone_.onClick = [this]
    {
        juce::String error;
        if (!prepareRecipe())
            return;
        if (composer_->clone(fragment_.getText(), newId_.getText().trim(), error))
            refresh();
        else
            report(error, true);
    };
    remove_.onClick = [this]
    {
        juce::String error;
        if (!prepareRecipe())
            return;
        if (composer_->remove(fragment_.getText(), error))
            refresh();
        else
            report(error, true);
    };
    bind_.onClick = [this]
    {
        addBinding();
    };
    unbind_.onClick = [this]
    {
        juce::String error;
        if (!prepareRecipe())
            return;
        if (composer_->unbind(bindingId_.getText().trim(), error))
            refresh();
        else
            report(error, true);
    };
    prepare_.onClick = [this]
    {
        if (prepareRecipe())
        {
            refresh();
            report("Recipe prepared; press Apply to publish");
        }
    };
    apply_.onClick = [this]
    {
        if (!prepareRecipe())
            return;
        juce::String error;
        if (publish_(composer_->document(), error))
            report("Composition published to the sound; Undo is available");
        else
            report(error, true);
    };
}
CompositionPanel::~CompositionPanel()
{
    if (primitiveDialog_)
        delete primitiveDialog_.getComponent();
    setLookAndFeel(nullptr);
}
void CompositionPanel::createFragment()
{
    if (primitiveDialog_)
    {
        primitiveDialog_->toFront(true);
        return;
    }
    if (!prepareRecipe())
        return;
    const auto document = JSON::parse(composer_->document().toStdString());
    const auto tier = document.value(
        "tier", document.value("scalar_type", std::string{}) == "float64" ? "Pro" : "Standard");
    const auto id = newId_.getText().trim();
    auto safe = juce::Component::SafePointer<CompositionPanel>(this);
    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = "Create primitive fragment";
    options.dialogBackgroundColour = look_.theme().bg;
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = false;
    options.componentToCentreAround = this;
    options.content.setOwned(new PrimitiveFragmentPanel(
        sampleRate_, tier,
        [safe, id](const juce::String& fragment, juce::String& error)
        {
            if (!safe)
            {
                error = "Composition closed";
                return false;
            }
            if (!safe->composer_->add(id, fragment, error))
                return false;
            safe->refresh();
            safe->report("Primitive fragment inserted as " + id);
            return true;
        },
        &look_.theme()));
    primitiveDialog_ = options.launchAsync();
}
void CompositionPanel::report(const juce::String& text, bool error)
{
    status_.setColour(juce::Label::textColourId, error ? look_.theme().warn : look_.theme().text2);
    status_.setText(text, juce::dontSendNotification);
}
bool CompositionPanel::prepareRecipe()
{
    if (needsBase_)
    {
        report("Use edited sound as base before modifying this composition", true);
        return false;
    }
    try
    {
        auto candidate = JSON::parse(recipe_.getText().toStdString());
        candidate["budget"] = JSON::parse(budget_.getText().toStdString());
        juce::String error;
        if (!composer_->restore(candidate.dump(), error))
        {
            report(error, true);
            return false;
        }
        return true;
    }
    catch (const std::exception& error)
    {
        report(error.what(), true);
        return false;
    }
}
void CompositionPanel::refresh()
{
    recipe_.setText(composer_->recipe(), false);
    const auto ids = composer_->fragmentIds();
    for (auto* combo : {&fragment_, &source_, &destination_})
    {
        const auto prior = combo->getText();
        combo->clear(juce::dontSendNotification);
        combo->addItemList(ids, 1);
        combo->setSelectedId(ids.isEmpty() ? 0 : juce::jmax(0, ids.indexOf(prior)) + 1,
                             juce::dontSendNotification);
    }
}
void CompositionPanel::importFragment()
{
    if (!prepareRecipe())
        return;
    chooser_ =
        std::make_unique<juce::FileChooser>("Insert executable tensor fragment", juce::File{}, "*.json");
    auto safe = juce::Component::SafePointer<CompositionPanel>(this);
    chooser_->launchAsync(
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [safe](const juce::FileChooser& chooser)
        {
            if (!safe || !chooser.getResult().existsAsFile())
                return;
            juce::String error;
            if (safe->composer_->add(safe->newId_.getText().trim(), chooser.getResult().loadFileAsString(),
                                     error))
            {
                safe->refresh();
                safe->report("Fragment inserted with its own state, parameters and authored mappings");
            }
            else
                safe->report(error, true);
        });
}
void CompositionPanel::addBinding()
{
    if (!prepareRecipe())
        return;
    try
    {
        TensorComposition::Binding binding;
        binding.id = bindingId_.getText().trim();
        binding.sourceFragment = source_.getText();
        binding.destinationFragment = destination_.getText();
        binding.source = sourceKind_.getText();
        binding.sourceElement = indexValue(sourceElement_);
        binding.destination = targetKind_.getText();
        binding.destinationInstance = indexValue(instance_);
        binding.destinationElement = indexValue(targetElement_);
        binding.weights = weights_.getText();
        const auto text = weight_.getText().trim().toStdString();
        size_t used = 0;
        binding.weight = std::stod(text, &used);
        if (used != text.size() || !std::isfinite(binding.weight))
            throw std::invalid_argument("Coefficient must be finite");
        juce::String error;
        if (composer_->bind(binding, error))
        {
            refresh();
            report("Recurrent binding added; all updates still read committed previous state");
        }
        else
            report(error, true);
    }
    catch (const std::exception& error)
    {
        report(error.what(), true);
    }
}
void CompositionPanel::paint(juce::Graphics& g)
{
    g.fillAll(look_.theme().bg);
    g.setColour(look_.theme().text2);
    g.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
    g.drawText("Fragment / new ID", 20, 78, 400, 20, juce::Justification::centredLeft);
    const std::array<const char*, 8> labels{
        "Binding ID",           "Source fragment",  "Source representation", "Source coordinate",
        "Destination fragment", "Destination kind", "Primitive instance",    "Target coordinate"};
    for (size_t i = 0; i < labels.size(); ++i)
        g.drawText(labels[i], 20 + static_cast<int>(i % 4) * 235, 150 + static_cast<int>(i / 4) * 65, 225, 20,
                   juce::Justification::centredLeft);
    g.drawText("Coefficient", 20, 280, 225, 20, juce::Justification::centredLeft);
    g.drawText("Weight tensor", 725, 280, 225, 20, juce::Justification::centredLeft);
    g.drawText("Explicit resource budget", 20, 350, 260, 20, juce::Justification::centredLeft);
    g.drawText("Fragment tensors and recurrent composition recipe", 320, 350, 630, 20,
               juce::Justification::centredLeft);
}
void CompositionPanel::resized()
{
    help_.setBounds(20, 12, getWidth() - 40, 60);
    fragment_.setBounds(20, 100, 220, 32);
    newId_.setBounds(255, 100, 150, 32);
    import_.setBounds(420, 100, 180, 32);
    clone_.setBounds(615, 100, 120, 32);
    remove_.setBounds(750, 100, 180, 32);
    const std::array<juce::Component*, 8> controls{&bindingId_,     &source_,       &sourceKind_,
                                                   &sourceElement_, &destination_,  &targetKind_,
                                                   &instance_,      &targetElement_};
    for (size_t i = 0; i < controls.size(); ++i)
        controls[i]->setBounds(20 + static_cast<int>(i % 4) * 235, 172 + static_cast<int>(i / 4) * 65, 225,
                               32);
    weight_.setBounds(20, 302, 225, 32);
    bind_.setBounds(255, 302, 225, 32);
    unbind_.setBounds(490, 302, 225, 32);
    weights_.setBounds(725, 302, 225, 32);
    budget_.setBounds(20, 375, 285, getHeight() - 450);
    recipe_.setBounds(320, 375, getWidth() - 340, getHeight() - 450);
    prepare_.setBounds(20, getHeight() - 65, 200, 32);
    apply_.setBounds(230, getHeight() - 65, 200, 32);
    base_.setBounds(440, getHeight() - 65, 245, 32);
    create_.setBounds(725, getHeight() - 65, 225, 32);
    status_.setBounds(20, getHeight() - 29, getWidth() - 40, 25);
}
} // namespace alfa::tensorhost
