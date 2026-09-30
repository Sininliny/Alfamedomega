#include "TensorEditor.h"

#include <nlohmann/json.hpp>
namespace alfa::tensorhost
{
Editor::Editor(Processor& p)
    : AudioProcessorEditor(p)
    , processor_(p)
    , keyboard_(p.keyboard, juce::MidiKeyboardComponent::horizontalKeyboard)
{
    addAndMakeVisible(keyboard_);
    title_.setText("Alfamedomega | Audio Tensor Space", juce::dontSendNotification);
    hint_.setText("Executable tensor workspace. Musical intentions affect sound only through explicit "
                  "learned/edited tensor bindings.",
                  juce::dontSendNotification);
    for (auto* component : std::initializer_list<juce::Component*>{
             &title_, &hint_, &status_, &mode_, &layer_, &document_, &inspection_, &apply_, &open_, &save_,
             &references_, &loadReference_, &importModel_})
        addAndMakeVisible(component);
    mode_.addItemList({"Instrument", "FX", "Hybrid"}, 1);
    mode_.setSelectedId(p.mode.load() + 1);
    references_.addItemList({"MIDI projection demonstration", "Audio recurrence demonstration",
                             "Hybrid interaction demonstration"},
                            1);
    references_.setSelectedId(1);
    loadReference_.onClick = [this]
    {
        const auto index = references_.getSelectedId() - 1;
        document_.setText(Processor::referenceDocument(index));
        juce::String error;
        if (processor_.loadJSON(document_.getText(), error))
        {
            mode_.setSelectedId(index + 1);
            status_.setText("Explicit demonstration loaded; no trained model", juce::dontSendNotification);
        }
        else
            status_.setText(error, juce::dontSendNotification);
    };
    mode_.onChange = [this]
    {
        processor_.mode.store(mode_.getSelectedId() - 1);
    };
    layer_.addItemList({"Musical intentions", "Advanced inspection", "Expert tensor JSON"}, 1);
    layer_.setSelectedId(1);
    layer_.onChange = [this]
    {
        document_.setVisible(layer_.getSelectedId() == 3);
        inspection_.setVisible(layer_.getSelectedId() != 3);
    };
    document_.setMultiLine(true);
    document_.setReturnKeyStartsNewLine(true);
    document_.setText(p.document());
    document_.setFont(juce::Font(juce::FontOptions(14.0f)));
    inspection_.setMultiLine(true);
    inspection_.setReadOnly(true);
    inspection_.setVisible(true);
    document_.setVisible(false);
    const std::array<juce::String, 5> names{"Brightness", "Warmth", "Movement", "Aggression", "Organic"};
    for (size_t i = 0; i < sliders_.size(); ++i)
    {
        addAndMakeVisible(sliders_[i]);
        addAndMakeVisible(labels_[i]);
        labels_[i].setText(names[i], juce::dontSendNotification);
        sliders_[i].setRange(-1, 1, 0.001);
        sliders_[i].setValue(p.intentions[i].load());
        sliders_[i].onValueChange = [this, i]
        {
            processor_.intentions[i].store(static_cast<float>(sliders_[i].getValue()));
        };
    }
    apply_.onClick = [this]
    {
        juce::String error;
        status_.setText(processor_.loadJSON(document_.getText(), error)
                            ? "Compiled; publishes at next audio block"
                            : error,
                        juce::dontSendNotification);
    };
    open_.onClick = [this]
    {
        chooser_ =
            std::make_unique<juce::FileChooser>("Open AOM tensor configuration", juce::File{}, "*.json");
        auto safe = juce::Component::SafePointer<Editor>(this);
        chooser_->launchAsync(juce::FileBrowserComponent::openMode |
                                  juce::FileBrowserComponent::canSelectFiles,
                              [safe](const juce::FileChooser& c)
                              {
                                  if (safe && c.getResult().existsAsFile())
                                      safe->document_.setText(c.getResult().loadFileAsString());
                              });
    };
    importModel_.onClick = [this]
    {
        chooser_ = std::make_unique<juce::FileChooser>("Import runtime tensor compiler weights", juce::File{},
                                                       "*.json");
        auto safe = juce::Component::SafePointer<Editor>(this);
        chooser_->launchAsync(
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [safe](const juce::FileChooser& c)
            {
                if (!safe || !c.getResult().existsAsFile())
                    return;
                try
                {
                    auto configuration = nlohmann::json::parse(safe->document_.getText().toStdString());
                    configuration["runtimeModel"] =
                        nlohmann::json::parse(c.getResult().loadFileAsString().toStdString());
                    const juce::String text(configuration.dump(2));
                    juce::String error;
                    if (safe->processor_.loadJSON(text, error))
                    {
                        safe->document_.setText(text);
                        safe->status_.setText("Runtime compiler model loaded and validated",
                                              juce::dontSendNotification);
                    }
                    else
                        safe->status_.setText(error, juce::dontSendNotification);
                }
                catch (const std::exception& error)
                {
                    safe->status_.setText(error.what(), juce::dontSendNotification);
                }
            });
    };
    save_.onClick = [this]
    {
        chooser_ =
            std::make_unique<juce::FileChooser>("Save AOM tensor configuration", juce::File{}, "*.json");
        auto safe = juce::Component::SafePointer<Editor>(this);
        chooser_->launchAsync(
            juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles,
            [safe](const juce::FileChooser& c)
            {
                if (safe && c.getResult() != juce::File{})
                    safe->status_.setText(c.getResult().replaceWithText(safe->document_.getText())
                                              ? "Configuration saved"
                                              : "Save failed",
                                          juce::dontSendNotification);
            });
    };
    setSize(1000, 760);
    startTimerHz(4);
}
void Editor::resized()
{
    auto area = getLocalBounds().reduced(16);
    title_.setBounds(area.removeFromTop(32));
    hint_.setBounds(area.removeFromTop(32));
    auto row = area.removeFromTop(36);
    mode_.setBounds(row.removeFromLeft(150).reduced(2));
    layer_.setBounds(row.removeFromLeft(220).reduced(2));
    open_.setBounds(row.removeFromLeft(160).reduced(2));
    save_.setBounds(row.removeFromLeft(160).reduced(2));
    apply_.setBounds(row.reduced(2));
    row = area.removeFromTop(36);
    references_.setBounds(row.removeFromLeft(350).reduced(2));
    loadReference_.setBounds(row.removeFromLeft(280).reduced(2));
    importModel_.setBounds(row.reduced(2));
    row = area.removeFromTop(78);
    const auto width = row.getWidth() / 5;
    for (size_t i = 0; i < 5; ++i)
    {
        auto cell = row.removeFromLeft(width);
        labels_[i].setBounds(cell.removeFromTop(22));
        sliders_[i].setBounds(cell);
    }
    keyboard_.setBounds(area.removeFromBottom(72));
    status_.setBounds(area.removeFromBottom(42));
    document_.setBounds(area);
    inspection_.setBounds(area);
}
void Editor::timerCallback()
{
    if (layer_.getSelectedId() == 1)
    {
        inspection_.setText(
            "Play the keyboard, use MIDI performance, or route live audio input.\n\nMusical intentions are "
            "shared across all capacity levels.\nTheir sound relationships come from the loaded "
            "representation.\n\nUse Advanced inspection to examine the executable behavior.\nUse Expert "
            "tensor JSON to edit parameters, routing and protected controls.\n\nEmbedded examples "
            "demonstrate deterministic execution; they contain no trained model.");
        return;
    }
    auto report =
        processor_.inspection() +
        "\nProcessed samples: " + juce::String(static_cast<juce::int64>(processor_.samples.load())) +
        " | rejected events: " + juce::String(static_cast<juce::int64>(processor_.failures.load())) +
        " | measured block ms: " + juce::String(processor_.blockMilliseconds.load(), 3) +
        " | compiler inference ms/block: " + juce::String(processor_.modelMilliseconds.load(), 3) +
        "\nCurrent state (first 16 elements): ";
    for (const auto& value : processor_.stateSnapshot)
        report += juce::String(value.load(), 4) + " ";
    inspection_.setText(report);
}
} // namespace alfa::tensorhost
