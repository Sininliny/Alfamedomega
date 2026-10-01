#include "SoundLibraryPanel.h"
#include "TensorProcessor.h"

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace
{
using JSON = nlohmann::json;
using alfa::tensorhost::SoundLibraryPanel;
template <typename Control>
Control& find(juce::Component& panel, const char* id)
{
    for (auto* child : panel.getChildren())
        if (child->getComponentID() == id)
            if (auto* typed = dynamic_cast<Control*>(child))
                return *typed;
    throw std::runtime_error(std::string("Library control unavailable: ") + id);
}
void pump()
{
    juce::MessageManager::getInstance()->runDispatchLoopUntil(25);
}
std::vector<SoundLibraryPanel::Entry> entries()
{
    return {{"example-a", "Alpha", "Examples", {}, alfa::tensorhost::Processor::referenceDocument(0)},
            {"example-b", "Beta", "Examples", {}, alfa::tensorhost::Processor::referenceDocument(1)},
            {"user-c", "Gamma", "User", {}, alfa::tensorhost::Processor::referenceDocument(2)}};
}
void category(juce::ComboBox& selector, const juce::String& name)
{
    for (int item = 0; item < selector.getNumItems(); ++item)
        if (selector.getItemText(item) == name)
        {
            selector.setSelectedId(selector.getItemId(item), juce::sendNotificationSync);
            return;
        }
    throw std::runtime_error("Library category unavailable");
}
} // namespace

TEST_CASE("Library browsing searching categories and keyboard navigation only inspect sound documents",
          "[tensor-host][sound-library]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    int publications = 0;
    SoundLibraryPanel panel(entries(),
                            [&](const juce::String&, juce::String&)
                            {
                                ++publications;
                                return true;
                            });
    panel.setVisible(true);
    auto& list = find<juce::ListBox>(panel, "library.list");
    REQUIRE(list.getListBoxModel()->getNumRows() == 3);
    CHECK(list.getSelectedRow() == -1);
    CHECK_FALSE(find<juce::Button>(panel, "library.load").isEnabled());
    list.selectRow(0);
    pump();
    CHECK(find<juce::TextEditor>(panel, "library.preview").getText().contains("math.sine"));
    REQUIRE(list.keyPressed(juce::KeyPress(juce::KeyPress::downKey)));
    pump();
    CHECK(list.getSelectedRow() == 1);
    CHECK(find<juce::TextEditor>(panel, "library.preview").getText().contains("math.one_pole_lowpass"));
    list.keyPressed(juce::KeyPress(juce::KeyPress::returnKey));
    pump();
    CHECK(publications == 0);
    const auto snapshot = juce::File(juce::String(ALFA_SOURCE_DIR))
                              .getChildFile("build")
                              .getChildFile("sound-library-" + juce::Uuid().toString() + ".png");
    auto stream = snapshot.createOutputStream();
    REQUIRE(stream != nullptr);
    REQUIRE(juce::PNGImageFormat{}.writeImageToStream(panel.createComponentSnapshot(panel.getLocalBounds()),
                                                      *stream));
    auto& search = find<juce::TextEditor>(panel, "library.search");
    search.setText("Gamma");
    pump();
    CHECK(list.getListBoxModel()->getNumRows() == 1);
    list.selectRow(0);
    pump();
    CHECK(find<juce::TextEditor>(panel, "library.preview").getText().contains("Name: Gamma"));
    search.setText("");
    pump();
    category(find<juce::ComboBox>(panel, "library.category"), "Examples");
    pump();
    CHECK(list.getListBoxModel()->getNumRows() == 2);
    list.selectRow(1);
    pump();
    CHECK(publications == 0);
}

TEST_CASE("Only explicit Load sound publishes the complete inspected JSON", "[tensor-host][sound-library]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    auto original = entries();
    auto document = JSON::parse(original[1].factoryDocument.toStdString());
    document["metadata"] = {{"author", "User"}, {"unknown_field", {1, 2, 3}}};
    original[1].factoryDocument = document.dump(2);
    const auto expected = original[1].factoryDocument;
    int publications = 0;
    juce::String published;
    SoundLibraryPanel panel(original,
                            [&](const juce::String& sound, juce::String&)
                            {
                                ++publications;
                                published = sound;
                                return true;
                            });
    original.clear(); // The panel owns stable Entry copies.
    auto& list = find<juce::ListBox>(panel, "library.list");
    list.selectRow(1);
    pump();
    REQUIRE(publications == 0);
    auto& load = find<juce::Button>(panel, "library.load");
    REQUIRE(load.isEnabled());
    load.triggerClick();
    pump();
    CHECK(publications == 1);
    CHECK(published == expected);
    CHECK(JSON::parse(published.toStdString())["metadata"] == document["metadata"]);
    load.triggerClick();
    pump();
    CHECK(publications == 2);
}

TEST_CASE("Malformed or changed file inspection cannot publish cached sound data",
          "[tensor-host][sound-library]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    const auto file = juce::File(juce::String(ALFA_SOURCE_DIR))
                          .getChildFile("build")
                          .getChildFile("library-panel-" + juce::Uuid().toString() + ".json");
    REQUIRE(file.getParentDirectory().createDirectory().wasOk());
    REQUIRE(file.replaceWithText("{malformed"));
    int publications = 0;
    SoundLibraryPanel panel({{"file", "Selected file", "User", file, {}}},
                            [&](const juce::String&, juce::String&)
                            {
                                ++publications;
                                return true;
                            });
    auto& list = find<juce::ListBox>(panel, "library.list");
    list.selectRow(0);
    pump();
    CHECK(publications == 0);
    CHECK_FALSE(find<juce::Button>(panel, "library.load").isEnabled());
    CHECK(find<juce::TextEditor>(panel, "library.preview").getText().contains("Cannot inspect"));
    REQUIRE(file.replaceWithText(alfa::tensorhost::Processor::referenceDocument(0)));
    list.deselectAllRows();
    list.selectRow(0);
    pump();
    auto& load = find<juce::Button>(panel, "library.load");
    REQUIRE(load.isEnabled());
    REQUIRE(file.replaceWithText("{changed-invalid"));
    load.triggerClick();
    pump();
    CHECK(publications == 0);
    CHECK_FALSE(load.isEnabled());
    CHECK(file.deleteFile());
}
