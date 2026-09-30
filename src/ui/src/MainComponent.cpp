#include "alfa/ui/Views.h"

namespace alfa::ui
{

int gapFor(const UiState& ui)
{
    return ui.getString(UiState::Ids::density) == "compact" ? 6 : 10;
}

namespace
{
const juce::StringArray viewIds{"perform", "signal", "sound", "customize"};
}

// HeaderBar
// ---------------------------------------------------------------------------

HeaderBar::HeaderBar(EditorContext& c, LiveData& l)
    : ctx(c)
    , live(l)
    , viewSwitch({"Perform", "Signal flow", "Sound design", "Customize"},
                 [this](int i) { ctx.ui.set(UiState::Ids::view, viewIds[i]); })
{
    addAndMakeVisible(viewSwitch);
    viewSwitch.setSelected(juce::jmax(0, viewIds.indexOf(ctx.ui.getString(UiState::Ids::view))));

    prevPreset.setTooltip("Previous preset");
    nextPreset.setTooltip("Next preset");
    presetName.setTooltip("Browse presets");
    prevPreset.onClick = [this]
    {
        const auto n = ctx.host.numPresets();
        ctx.host.loadPreset((ctx.host.currentPreset() - 1 + n) % n, true);
        refreshPreset();
    };
    nextPreset.onClick = [this]
    {
        ctx.host.loadPreset((ctx.host.currentPreset() + 1) % ctx.host.numPresets(), true);
        refreshPreset();
    };
    presetName.onClick = [this]
    {
        showPresetMenu();
    };
    addAndMakeVisible(prevPreset);
    addAndMakeVisible(presetName);
    addAndMakeVisible(nextPreset);

    // Like and dislike feed preference learning, so they only show while it is on.
    like.setTooltip("I like this sound (teaches your preference model)");
    dislike.setTooltip("Not for me (teaches your preference model)");
    for (auto* b : {&like, &dislike})
    {
        b->setClickingTogglesState(false);
        addChildComponent(b);
    }
    like.onClick = [this]
    {
        ctx.host.rateSound(ctx.host.soundRating() == 1 ? 0 : 1);
        refreshRating();
    };
    dislike.onClick = [this]
    {
        ctx.host.rateSound(ctx.host.soundRating() == -1 ? 0 : -1);
        refreshRating();
    };

    for (size_t i = 0; i < params::kTiers.size(); ++i)
    {
        const auto& t = params::kTiers[i];
        tierBox.addItem(juce::String(t.name) + "  " + juce::String(juce::roundToInt(t.totalParamsMillions)) +
                            "M",
                        static_cast<int>(i) + 1);
    }
    tierBox.setTooltip("Model size: bigger models sound more detailed and cost more CPU");
    tierBox.setTitle("Model size");
    addAndMakeVisible(tierBox);
    tierAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        ctx.params, params::id::modelTier, tierBox);

    live.addListener(this);
    ctx.ui.tree().addListener(this);
    refreshPreset();
}

HeaderBar::~HeaderBar()
{
    live.removeListener(this);
    ctx.ui.tree().removeListener(this);
}

void HeaderBar::refreshPreset()
{
    const auto i = ctx.host.currentPreset();
    presetName.setButtonText(ctx.host.presetName(i) + "   " + ctx.host.presetCategory(i) + " · made on " +
                             params::kTiers[static_cast<size_t>(ctx.host.presetMadeOn(i))].name);
}

void HeaderBar::refreshRating()
{
    const bool learning = ctx.host.learningEnabled();
    if (learning != like.isVisible())
    {
        like.setVisible(learning);
        dislike.setVisible(learning);
        resized();
    }
    like.setToggleState(ctx.host.soundRating() == 1, juce::dontSendNotification);
    dislike.setToggleState(ctx.host.soundRating() == -1, juce::dontSendNotification);
}

void HeaderBar::showPresetMenu()
{
    juce::PopupMenu menu;
    std::map<juce::String, juce::PopupMenu> byCategory;
    for (int i = 0; i < ctx.host.numPresets(); ++i)
        byCategory[ctx.host.presetCategory(i)].addItem(ctx.host.presetName(i), true,
                                                       i == ctx.host.currentPreset(),
                                                       [this, i]
                                                       {
                                                           ctx.host.loadPreset(i, false);
                                                           refreshPreset();
                                                       });
    for (auto& [category, sub] : byCategory)
        menu.addSubMenu(category, sub);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&presetName));
}

void HeaderBar::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    if (id == UiState::Ids::view)
        viewSwitch.setSelected(juce::jmax(0, viewIds.indexOf(ctx.ui.getString(UiState::Ids::view))));
}

void HeaderBar::liveDataUpdated()
{
    const auto& s = live.snapshot();
    const auto& tier = params::kTiers[static_cast<size_t>(ctx.selectedTier())];
    const auto latencyMs =
        static_cast<float>(ctx.host.latencySamples()) / static_cast<float>(s.sampleRate) * 1000.0f;
    auto text =
        "CPU " + juce::String(juce::roundToInt(s.cpu * 100.0f)) + "%  " + juce::String(latencyMs, 1) + " ms";
    if (static_cast<int>(ctx.host.loadedTier()) != static_cast<int>(ctx.selectedTier()))
        text = "loading " + juce::String(tier.name) + "...";
    statsWarn = s.cpu > 0.7f;
    if (text != statsText)
    {
        statsText = text;
        repaint(statsArea);
    }
    refreshPreset();
    refreshRating();
}

void HeaderBar::paint(juce::Graphics& g)
{
    const auto& t = ctx.theme();
    g.setColour(t.panel);
    g.fillRect(getLocalBounds());
    g.setColour(t.line);
    g.fillRect(getLocalBounds().removeFromBottom(1));

    g.setColour(t.text);
    g.setFont(AlfaLookAndFeel::labelFont(19.0f, true));
    g.drawText("Alfamedomega", getLocalBounds().withTrimmedLeft(18).withWidth(170),
               juce::Justification::centredLeft, false);

    g.setColour(statsWarn ? t.warn : t.text2);
    g.setFont(AlfaLookAndFeel::monoFont(12.5f));
    g.drawText(statsText, statsArea, juce::Justification::centredRight, false);
}

void HeaderBar::resized()
{
    auto b = getLocalBounds().reduced(12, 12);
    b.removeFromLeft(170);
    viewSwitch.setBounds(b.removeFromLeft(420));
    b.removeFromLeft(14);
    prevPreset.setBounds(b.removeFromLeft(30));
    presetName.setBounds(b.removeFromLeft(like.isVisible() ? 206 : 262).reduced(4, 0));
    nextPreset.setBounds(b.removeFromLeft(30));
    if (like.isVisible())
    {
        b.removeFromLeft(4);
        like.setBounds(b.removeFromLeft(26));
        dislike.setBounds(b.removeFromLeft(26));
    }
    b.removeFromLeft(10);
    tierBox.setBounds(b.removeFromLeft(136));
    statsArea = b.withTrimmedLeft(6);
}

// MainComponent
// -------------------------------------------------------------------------

MainComponent::MainComponent(EditorContext& c)
    : ctx(c)
    , live(c)
    , header(c, live)
    , perform(c, live)
    , signalFlow(c, live)
    , soundDesign(c, live)
    , customize(c)
{
    setSize(kDesignWidth, kDesignHeight);
    addAndMakeVisible(header);
    addChildComponent(perform);
    addChildComponent(signalFlow);
    addChildComponent(soundDesign);
    addChildComponent(customize);
    ctx.ui.tree().addListener(this);
    applyTheme();
    showView();
}

MainComponent::~MainComponent()
{
    ctx.ui.tree().removeListener(this);
}

void MainComponent::applyTheme()
{
    auto theme = Theme::builtIn(ctx.ui.getString(UiState::Ids::theme));
    const auto accent = ctx.ui.getString(UiState::Ids::accent);
    if (accent.isNotEmpty())
        theme.accent = juce::Colour::fromString("ff" + accent.removeCharacters("#"));
    ctx.lnf.setTheme(theme);
    sendLookAndFeelChange();
    repaint();
}

void MainComponent::showView()
{
    const auto view = ctx.ui.getString(UiState::Ids::view);
    perform.setVisible(view == "perform");
    signalFlow.setVisible(view == "signal");
    soundDesign.setVisible(view == "sound");
    customize.setVisible(view == "customize");
}

void MainComponent::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    if (id == UiState::Ids::view)
        showView();
    else if (id == UiState::Ids::theme || id == UiState::Ids::accent)
        applyTheme();
    else if (id == UiState::Ids::density)
        for (auto* view :
             std::initializer_list<juce::Component*>{&perform, &signalFlow, &soundDesign, &customize})
            view->resized();
}

void MainComponent::paint(juce::Graphics& g)
{
    g.fillAll(ctx.theme().bg);
}

void MainComponent::resized()
{
    auto b = getLocalBounds();
    header.setBounds(b.removeFromTop(60));
    const auto gap = gapFor(ctx.ui);
    b.reduce(gap + 2, gap);
    for (auto* view :
         std::initializer_list<juce::Component*>{&perform, &signalFlow, &soundDesign, &customize})
        view->setBounds(b);
}

} // namespace alfa::ui
