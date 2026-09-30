#include "alfa/ui/UiState.h"

namespace alfa::ui
{

UiState::UiState()
    : state(createDefault())
{
}

juce::ValueTree UiState::createDefault()
{
    juce::ValueTree t{type};
    t.setProperty(Ids::view, "perform", nullptr);
    t.setProperty(Ids::theme, "graphite", nullptr);
    t.setProperty(Ids::accent, "", nullptr);
    t.setProperty(Ids::scale, 1.0, nullptr);
    t.setProperty(Ids::density, "comfortable", nullptr);
    t.setProperty(Ids::knobGesture, "vertical", nullptr);
    t.setProperty(Ids::valueDisplay, "units", nullptr);
    t.setProperty(Ids::macrosPerPage, 8, nullptr);
    t.setProperty(Ids::macroPage, 0, nullptr);
    t.setProperty(Ids::showBlend, true, nullptr);
    t.setProperty(Ids::showMacros, true, nullptr);
    t.setProperty(Ids::showTimbre, true, nullptr);
    t.setProperty(Ids::showVoice, true, nullptr);
    t.setProperty(Ids::showKeyboard, true, nullptr);
    t.setProperty(Ids::showFlowStrip, false, nullptr);
    t.setProperty(Ids::flowDetail, "student", nullptr);
    t.setProperty(Ids::flowExplain, true, nullptr);
    t.setProperty(Ids::flowHidden, "", nullptr);
    t.setProperty(Ids::flowSelected, 2, nullptr);
    t.setProperty(Ids::soundLayer, 0, nullptr);
    t.setProperty(Ids::locks, "", nullptr);
    t.setProperty(Ids::nearbyMode, "my_taste", nullptr);
    return t;
}

void UiState::restore(const juce::ValueTree& saved)
{
    if (!saved.hasType(type))
        return;

    // Copy property by property so listeners on `state` stay attached and any
    // property added in a newer version keeps its default.
    auto defaults = createDefault();
    for (int i = 0; i < defaults.getNumProperties(); ++i)
    {
        const auto name = defaults.getPropertyName(i);
        state.setProperty(name, saved.getProperty(name, defaults.getProperty(name)), nullptr);
    }
}

juce::StringArray UiState::splitList(const juce::String& s)
{
    juce::StringArray items;
    items.addTokens(s, ",", "");
    items.removeEmptyStrings();
    return items;
}

bool UiState::isStepHidden(const juce::String& stepId) const
{
    return splitList(getString(Ids::flowHidden)).contains(stepId);
}

void UiState::setStepHidden(const juce::String& stepId, bool hidden)
{
    auto items = splitList(getString(Ids::flowHidden));
    items.removeString(stepId);
    if (hidden)
        items.add(stepId);
    set(Ids::flowHidden, items.joinIntoString(","));
}

juce::StringArray UiState::lockedParameters() const
{
    return splitList(getString(Ids::locks));
}

bool UiState::isLocked(const juce::String& paramId) const
{
    return lockedParameters().contains(paramId);
}

void UiState::setLocked(const juce::String& paramId, bool locked)
{
    auto items = lockedParameters();
    items.removeString(paramId);
    if (locked)
        items.add(paramId);
    set(Ids::locks, items.joinIntoString(","));
}

juce::File UiState::userSettingsFile()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Alfamedomega")
        .getChildFile("ui-defaults.xml");
}

void UiState::saveAsUserDefault() const
{
    // Locks and the current view belong to this instance, not to the user's
    // defaults.
    auto copy = state.createCopy();
    copy.setProperty(Ids::locks, "", nullptr);
    copy.setProperty(Ids::view, "perform", nullptr);

    auto file = userSettingsFile();
    file.getParentDirectory().createDirectory();
    if (auto xml = copy.createXml())
        xml->writeTo(file);
}

void UiState::loadUserDefault()
{
    if (auto xml = juce::XmlDocument::parse(userSettingsFile()))
        restore(juce::ValueTree::fromXml(*xml));
}

} // namespace alfa::ui
