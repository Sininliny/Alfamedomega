#include "CompositionPanel.h"
#include "TensorProcessor.h"

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace
{
using JSON = nlohmann::json;
using alfa::tensorhost::CompositionPanel;
using alfa::tensorhost::TensorComposition;
template <typename Control>
Control& control(juce::Component& panel, const juce::String& id)
{
    for (auto* child : panel.getChildren())
        if (child->getComponentID() == id)
        {
            auto* typed = dynamic_cast<Control*>(child);
            if (typed == nullptr)
                throw std::runtime_error("Composition control has unexpected type: " + id.toStdString());
            return *typed;
        }
    throw std::runtime_error("Composition control is unavailable: " + id.toStdString());
}
void click(juce::Component& panel, const juce::String& text)
{
    for (auto* child : panel.getChildren())
        if (auto* button = dynamic_cast<juce::Button*>(child);
            button != nullptr && button->getButtonText() == text)
        {
            REQUIRE(button->isEnabled());
            button->triggerClick();
            juce::MessageManager::getInstance()->runDispatchLoopUntil(30);
            return;
        }
    throw std::runtime_error("Composition button is unavailable: " + text.toStdString());
}
juce::String fixture()
{
    auto sound = JSON::parse(juce::File(juce::String(ALFA_SOURCE_DIR))
                                 .getChildFile("programs/fx.aom.json")
                                 .loadFileAsString()
                                 .toStdString());
    sound["instances"][0]["primitive"] = "math.feedback_cell";
    sound["instances"][0]["initial_state"] = {.5};
    sound["instances"][0]["input_channels"] = 1;
    sound["tensors"]["parameters"]["values"] = {.25};
    sound["budget"]["stored_tensor_elements"] = 3;
    return sound.dump(2);
}
void allowTwoFragments(juce::Component& panel)
{
    auto& editor = control<juce::TextEditor>(panel, "composition.budget");
    auto budget = JSON::parse(editor.getText().toStdString());
    for (const auto* field : {"active_primitives", "state_elements", "parameter_elements", "output_elements",
                              "stored_tensor_elements"})
        budget[field] = budget[field].get<std::size_t>() * 2;
    budget["routing_elements"] = 2;
    budget["modulation_elements"] = 2;
    budget["projection_terms"] = 2;
    budget["stored_tensor_elements"] = 8;
    editor.setText(budget.dump(2));
}
} // namespace

TEST_CASE(
    "Composition panel edits capacity clones and prepares explicit modulation bindings before publishing",
    "[tensor-host][composition-panel]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    alfa::tensorhost::Processor processor;
    processor.prepareToPlay(48000, 128);
    juce::String error;
    REQUIRE(processor.loadJSON(fixture(), error));
    const auto original = processor.document();
    int publications = 0;
    CompositionPanel panel(original, 48000,
                           [&](const juce::String& sound, juce::String& failure)
                           {
                               ++publications;
                               return processor.editConfigurationJSON(sound, failure);
                           });
    panel.setVisible(true);
    allowTwoFragments(panel);
    control<juce::TextEditor>(panel, "composition.new_id").setText("F1");
    click(panel, "Clone");
    auto& recipeEditor = control<juce::TextEditor>(panel, "composition.recipe");
    REQUIRE(JSON::parse(recipeEditor.getText().toStdString())["fragments"].size() == 2);
    control<juce::ComboBox>(panel, "composition.source").setSelectedId(1, juce::dontSendNotification);
    control<juce::ComboBox>(panel, "composition.destination").setSelectedId(2, juce::dontSendNotification);
    auto& weights = control<juce::ComboBox>(panel, "composition.weights");
    for (int item = 0; item < weights.getNumItems(); ++item)
        if (weights.getItemText(item) == "modulation")
            weights.setSelectedId(weights.getItemId(item), juce::dontSendNotification);
    REQUIRE(weights.getText() == "modulation");
    control<juce::TextEditor>(panel, "composition.weight").setText("0.75");
    click(panel, "Add binding");
    const auto prepared = JSON::parse(recipeEditor.getText().toStdString());
    REQUIRE(prepared["recurrent_bindings"].size() == 1);
    CHECK(prepared["recurrent_bindings"][0]["weights"] == "modulation");
    CHECK(prepared["recurrent_bindings"][0]["destination_fragment"] == "F1");
    click(panel, "Prepare recipe");
    CHECK(publications == 0);
    CHECK(processor.document() == original);
    click(panel, "Apply to sound");
    REQUIRE(publications == 1);
    const auto published = JSON::parse(processor.document().toStdString());
    CHECK(published["instances"].size() == 2);
    CHECK(published["instances"][1]["audio_bindings"][0]["weights"] == "modulation");
    REQUIRE(processor.undoManager().undo());
    CHECK(processor.document() == original);
    REQUIRE(processor.undoManager().redo());
    CHECK(JSON::parse(processor.document().toStdString()) == published);

    auto output = juce::File(juce::String(ALFA_SOURCE_DIR))
                      .getChildFile("build/composition-panel-tests")
                      .getChildFile("composition-panel-" + juce::Uuid().toString() + ".png");
    REQUIRE(output.getParentDirectory().createDirectory().wasOk());
    auto stream = output.createOutputStream();
    REQUIRE(stream != nullptr);
    stream->setPosition(0);
    stream->truncate();
    const auto snapshot = panel.createComponentSnapshot(panel.getLocalBounds());
    REQUIRE(snapshot.isValid());
    REQUIRE(juce::PNGImageFormat{}.writeImageToStream(snapshot, *stream));
}

TEST_CASE("Composition panel rejects invalid editable recipes and insufficient capacity without publishing",
          "[tensor-host][composition-panel]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    int publications = 0;
    CompositionPanel panel(fixture(), 48000,
                           [&](const juce::String&, juce::String&)
                           {
                               ++publications;
                               return true;
                           });
    auto& recipe = control<juce::TextEditor>(panel, "composition.recipe");
    const auto original = recipe.getText();
    recipe.setText("{invalid");
    click(panel, "Prepare recipe");
    click(panel, "Apply to sound");
    CHECK(publications == 0);
    CHECK(control<juce::Label>(panel, "composition.status").getText().isNotEmpty());
    recipe.setText(original);
    control<juce::TextEditor>(panel, "composition.new_id").setText("F1");
    click(panel, "Clone"); // Current budget allows just one cell.
    CHECK(JSON::parse(recipe.getText().toStdString())["fragments"].size() == 1);
    CHECK(publications == 0);
    allowTwoFragments(panel);
    click(panel, "Clone");
    CHECK(JSON::parse(recipe.getText().toStdString())["fragments"].size() == 2);
}

TEST_CASE("Composition panel requires explicit adoption of edited sound before publishing a stale recipe",
          "[tensor-host][composition-panel]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    auto base = JSON::parse(fixture().toStdString());
    TensorComposition composition(48000, base["budget"].dump());
    juce::String error;
    REQUIRE(composition.add("original", fixture(), error));
    auto edited = JSON::parse(composition.document().toStdString());
    edited["tensors"]["parameters"]["values"][0] = .625;
    int publications = 0;
    juce::String published;
    CompositionPanel panel(edited.dump(), 48000,
                           [&](const juce::String& sound, juce::String&)
                           {
                               ++publications;
                               published = sound;
                               return true;
                           });
    click(panel, "Apply to sound");
    CHECK(publications == 0);
    CHECK(control<juce::Label>(panel, "composition.status").getText().contains("base"));
    click(panel, "Use edited sound as base");
    click(panel, "Apply to sound");
    REQUIRE(publications == 1);
    CHECK(JSON::parse(published.toStdString())["tensors"]["parameters"]["values"][0] == .625);
    CHECK(JSON::parse(published.toStdString())["composition"]["fragments"][0]["id"] == "F0");
}
