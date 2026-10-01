#include "SoundLibraryPanel.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace alfa::tensorhost
{
namespace
{
using JSON = nlohmann::json;
}
SoundLibraryPanel::SoundLibraryPanel(std::vector<Entry> entries, Publish publish,
                                     const alfa::ui::Theme* theme)
    : entries_(std::move(entries))
    , publish_(std::move(publish))
{
    if (theme != nullptr)
        look_.setTheme(*theme);
    setLookAndFeel(&look_);
    setSize(900, 620);
    for (auto* component : std::initializer_list<juce::Component*>{&search_, &preview_, &category_, &list_,
                                                                   &heading_, &footer_, &status_, &load_})
        addAndMakeVisible(component);
    search_.setComponentID("library.search");
    category_.setComponentID("library.category");
    list_.setComponentID("library.list");
    preview_.setComponentID("library.preview");
    status_.setComponentID("library.status");
    load_.setComponentID("library.load");
    heading_.setText("Sound library / inspect complete configurations", juce::dontSendNotification);
    footer_.setText("Selecting or searching only inspects. Load sound explicitly replaces the current "
                    "complete configuration.",
                    juce::dontSendNotification);
    heading_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(16));
    footer_.setFont(alfa::ui::AlfaLookAndFeel::labelFont(12));
    search_.setTextToShowWhenEmpty("Search name, category or ID", look_.theme().text2);
    search_.onTextChange = [this]
    {
        filter();
    };
    category_.addItem("All categories", 1);
    for (const auto& entry : entries_)
        if (!categoryValues_.contains(entry.category))
            categoryValues_.add(entry.category);
    categoryValues_.sort(true);
    for (int index = 0; index < categoryValues_.size(); ++index)
        category_.addItem(categoryValues_[index].isEmpty() ? "(No category)" : categoryValues_[index],
                          index + 2);
    category_.setSelectedId(1, juce::dontSendNotification);
    category_.onChange = [this]
    {
        filter();
    };
    preview_.setMultiLine(true);
    preview_.setReadOnly(true);
    preview_.setFont(alfa::ui::AlfaLookAndFeel::monoFont(12));
    list_.setModel(this);
    list_.setRowHeight(40);
    load_.setEnabled(false);
    load_.onClick = [this]
    {
        loadSelected();
    };
    filter();
}
SoundLibraryPanel::~SoundLibraryPanel()
{
    list_.setModel(nullptr);
    setLookAndFeel(nullptr);
}
void SoundLibraryPanel::paint(juce::Graphics& graphics)
{
    graphics.fillAll(look_.theme().bg);
}
void SoundLibraryPanel::resized()
{
    heading_.setBounds(20, 10, getWidth() - 40, 30);
    search_.setBounds(20, 50, 360, 32);
    category_.setBounds(395, 50, getWidth() - 415, 32);
    list_.setBounds(20, 96, 320, getHeight() - 196);
    preview_.setBounds(355, 96, getWidth() - 375, getHeight() - 196);
    footer_.setBounds(20, getHeight() - 91, getWidth() - 40, 35);
    load_.setBounds(20, getHeight() - 48, 160, 32);
    status_.setBounds(195, getHeight() - 48, getWidth() - 215, 36);
}
int SoundLibraryPanel::getNumRows()
{
    return static_cast<int>(filtered_.size());
}
void SoundLibraryPanel::paintListBoxItem(int row, juce::Graphics& graphics, int width, int height,
                                         bool selected)
{
    if (row < 0 || static_cast<std::size_t>(row) >= filtered_.size())
        return;
    const auto& entry = entries_[filtered_[static_cast<std::size_t>(row)]];
    graphics.setColour(selected ? look_.theme().panel : look_.theme().bg);
    graphics.fillAll();
    graphics.setColour(look_.theme().text);
    graphics.setFont(alfa::ui::AlfaLookAndFeel::labelFont(13));
    graphics.drawText(entry.name, 8, 2, width - 16, height / 2, juce::Justification::centredLeft);
    graphics.setColour(look_.theme().text2);
    graphics.setFont(alfa::ui::AlfaLookAndFeel::labelFont(11));
    graphics.drawText(entry.category, 8, height / 2, width - 16, height / 2 - 2,
                      juce::Justification::centredLeft);
}
void SoundLibraryPanel::filter()
{
    // Refiltering clears inspection. It never loads or auditions the first match.
    list_.deselectAllRows();
    filtered_.clear();
    const auto query = search_.getText().trim();
    const auto category =
        category_.getSelectedId() == 1 ? juce::String{} : categoryValues_[category_.getSelectedId() - 2];
    for (std::size_t index = 0; index < entries_.size(); ++index)
    {
        const auto& entry = entries_[index];
        const bool matchesCategory = category_.getSelectedId() == 1 || entry.category == category;
        if (matchesCategory &&
            (query.isEmpty() || entry.name.containsIgnoreCase(query) ||
             entry.category.containsIgnoreCase(query) || entry.id.containsIgnoreCase(query)))
            filtered_.push_back(index);
    }
    list_.updateContent();
    preview_.setText(
        "Select a sound to inspect its configuration. Browsing does not change the current sound.", false);
    load_.setEnabled(false);
    status_.setText(juce::String(static_cast<int>(filtered_.size())) + " sounds", juce::dontSendNotification);
}
bool SoundLibraryPanel::inspect(const Entry& entry, juce::String& document, juce::String& error)
{
    try
    {
        if (entry.file != juce::File{})
        {
            if (!entry.file.existsAsFile())
                throw std::invalid_argument("Selected sound file is unavailable");
            document = entry.file.loadFileAsString();
        }
        else
            document = entry.factoryDocument;
        const auto data = JSON::parse(document.toStdString());
        if (!data.is_object() || data.at("schema") != "aom.tensor/1" || !data.at("instances").is_array())
            throw std::invalid_argument("Selected item is not a complete tensor configuration");
        juce::String summary =
            "Name: " + entry.name + "\nCategory: " + entry.category + "\nID: " + entry.id + "\n";
        summary += "Primitives: " + juce::String(static_cast<int>(data.at("instances").size())) + "\n";
        for (const auto& instance : data.at("instances"))
            summary += juce::String(instance.at("primitive").get<std::string>()) + " | clock " +
                       juce::String(instance.at("clock").dump()) + " | state " +
                       juce::String(instance.at("state_size").dump()) + "\n";
        summary += "\nTensor coordinates:\n";
        for (const auto& [name, tensor] : data.at("tensors").items())
            summary += juce::String(name) + ": axes " + juce::String(tensor.at("axes").dump()) + " shape " +
                       juce::String(tensor.at("shape").dump()) + "\n";
        summary += "\nFull configuration JSON:\n" + juce::String(data.dump(2));
        preview_.setText(summary, false);
        error.clear();
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        document.clear();
        preview_.setText("Cannot inspect selected sound: " + error, false);
        return false;
    }
}
void SoundLibraryPanel::selectedRowsChanged(int row)
{
    load_.setEnabled(false);
    if (row < 0 || static_cast<std::size_t>(row) >= filtered_.size())
        return;
    juce::String document, error;
    const auto valid = inspect(entries_[filtered_[static_cast<std::size_t>(row)]], document, error);
    load_.setEnabled(valid);
    status_.setColour(juce::Label::textColourId, valid ? look_.theme().text2 : look_.theme().warn);
    status_.setText(valid ? "Inspection only. Press Load sound to replace the current sound." : error,
                    juce::dontSendNotification);
}
void SoundLibraryPanel::loadSelected()
{
    const auto row = list_.getSelectedRow();
    if (row < 0 || static_cast<std::size_t>(row) >= filtered_.size())
        return;
    juce::String document, error;
    if (!inspect(entries_[filtered_[static_cast<std::size_t>(row)]], document, error))
    {
        load_.setEnabled(false);
        status_.setText(error, juce::dontSendNotification);
        return;
    }
    const auto loaded = publish_(document, error);
    status_.setColour(juce::Label::textColourId, loaded ? look_.theme().text2 : look_.theme().warn);
    status_.setText(loaded ? "Complete sound configuration loaded explicitly" : error,
                    juce::dontSendNotification);
}
} // namespace alfa::tensorhost
