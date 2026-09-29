#include "PluginEditor.h"

namespace alfa
{

AlfamedomegaEditor::AlfamedomegaEditor(AlfamedomegaProcessor& p)
    : AudioProcessorEditor(p)
    , processor(p)
    , context{p.parameters(), p.telemetry(), p.uiState(), p.keyboardState(), p, lookAndFeel}
{
    setLookAndFeel(&lookAndFeel);
    content = std::make_unique<ui::MainComponent>(context);
    addAndMakeVisible(*content);

    // The resize setup below calls resized() with provisional sizes; don't store
    // those as the user's scale.
    applyingScale = true;
    setResizable(true, true);
    setResizeLimits(ui::kDesignWidth * 3 / 4, ui::kDesignHeight * 3 / 4, ui::kDesignWidth * 2,
                    ui::kDesignHeight * 2);
    if (auto* c = getConstrainer())
        c->setFixedAspectRatio(static_cast<double>(ui::kDesignWidth) / ui::kDesignHeight);

    processor.uiState().tree().addListener(this);
    applyingScale = false;
    applyScale();
}

AlfamedomegaEditor::~AlfamedomegaEditor()
{
    processor.uiState().tree().removeListener(this);
    content.reset();
    setLookAndFeel(nullptr);
}

void AlfamedomegaEditor::applyScale()
{
    const auto scale = juce::jlimit(0.75, 2.0, processor.uiState().getDouble(ui::UiState::Ids::scale));
    const juce::ScopedValueSetter<bool> guard(applyingScale, true);
    setSize(juce::roundToInt(ui::kDesignWidth * scale), juce::roundToInt(ui::kDesignHeight * scale));
    resized();
}

void AlfamedomegaEditor::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id)
{
    if (id == ui::UiState::Ids::scale && !applyingScale)
        applyScale();
}

void AlfamedomegaEditor::paint(juce::Graphics& g)
{
    g.fillAll(lookAndFeel.theme().bg);
}

void AlfamedomegaEditor::resized()
{
    const auto scale = static_cast<float>(getWidth()) / static_cast<float>(ui::kDesignWidth);
    content->setBounds(0, 0, ui::kDesignWidth, ui::kDesignHeight);
    content->setTransform(juce::AffineTransform::scale(scale));

    // A drag on the window corner becomes the new interface size.
    if (!applyingScale)
    {
        const auto stored = processor.uiState().getDouble(ui::UiState::Ids::scale);
        if (std::abs(stored - static_cast<double>(scale)) > 0.005)
        {
            const juce::ScopedValueSetter<bool> guard(applyingScale, true);
            processor.uiState().set(ui::UiState::Ids::scale, static_cast<double>(scale));
        }
    }
}

} // namespace alfa
