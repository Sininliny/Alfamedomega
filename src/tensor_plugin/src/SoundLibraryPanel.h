#pragma once

#include "alfa/ui/AlfaLookAndFeel.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <vector>

namespace alfa::tensorhost
{
class SoundLibraryPanel final : public juce::Component, private juce::ListBoxModel
{
public:
    struct Entry
    {
        juce::String id, name, category;
        juce::File file;
        juce::String factoryDocument;
    };
    using Publish = std::function<bool(const juce::String&, juce::String&)>;
    SoundLibraryPanel(std::vector<Entry>, Publish, const alfa::ui::Theme* theme = nullptr);
    ~SoundLibraryPanel() override;
    void paint(juce::Graphics&) override;
    void resized() override;

private:
    std::vector<Entry> entries_;
    std::vector<std::size_t> filtered_;
    juce::StringArray categoryValues_;
    Publish publish_;
    alfa::ui::AlfaLookAndFeel look_;
    juce::TextEditor search_, preview_;
    juce::ComboBox category_;
    juce::ListBox list_;
    juce::Label heading_, footer_, status_;
    juce::TextButton load_{"Load sound"};
    int getNumRows() override;
    void paintListBoxItem(int, juce::Graphics&, int, int, bool) override;
    void selectedRowsChanged(int) override;
    void filter();
    bool inspect(const Entry&, juce::String& document, juce::String& error);
    void loadSelected();
};
} // namespace alfa::tensorhost
