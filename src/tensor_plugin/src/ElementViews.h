#pragma once
#include "ConsoleWidgets.h"

#include <functional>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <vector>

namespace alfa::tensorhost
{
// Title of one instance: the words all its parameter names share ("Voice 1"), else its primitive.
juce::String instanceTitle(const nlohmann::json& sound, size_t instance);

// One sound element drawn the way it sounds: an envelope as its curve, a low-pass as its response, an
// additive oscillator as its partials. Instances of the same primitive share one view and are drawn on
// top of each other (eight voices are eight curves). Curves use the parameters the runtime used in the
// last block when they are known, else the authored values; handles always edit the authored values of
// every instance in the view at once.
class ElementView final : public juce::Component, public juce::SettableTooltipClient
{
public:
    enum class Kind
    {
        Envelope,
        Lowpass,
        Partials,
        Delay,
        Width,
        Reverb,
        Generic
    };
    struct Instance
    {
        size_t index = 0, offset = 0, count = 0;
    };
    ElementView(alfa::ui::AlfaLookAndFeel&, juce::String primitive, juce::String title, std::vector<Instance>,
                std::vector<std::array<double, 2>> ranges, std::vector<juce::String> names);
    static Kind kindOf(const juce::String& primitive);
    Kind kind() const noexcept { return kind_; }
    const juce::String& primitive() const noexcept { return primitive_; }
    const std::vector<Instance>& instances() const noexcept { return instances_; }
    // Authored parameter tensor of the sound.
    void setBase(const std::vector<double>& parameters);
    // Parameters the runtime used (empty when not known yet) and instance output peaks.
    void setLive(const std::vector<float>& parameters, const std::vector<float>& levels, double sampleRate);
    // Parameters where the current movement is taking the sound (empty: no heading). Drawn dashed, with
    // the band between now and the heading amber where it rises and blue where it falls.
    void setHeading(const std::vector<double>& parameters);
    // Called once per finished drag with (local parameter, new authored value) pairs.
    std::function<void(const std::vector<std::pair<size_t, double>>&)> onEdit;
    std::function<void()> onOpen;
    void paint(juce::Graphics&) override;
    void mouseDown(const juce::MouseEvent&) override;
    void mouseDrag(const juce::MouseEvent&) override;
    void mouseUp(const juce::MouseEvent&) override;
    void mouseMove(const juce::MouseEvent&) override;
    void mouseDoubleClick(const juce::MouseEvent&) override;

private:
    alfa::ui::AlfaLookAndFeel& look_;
    juce::String primitive_, title_;
    Kind kind_;
    std::vector<Instance> instances_;
    std::vector<std::array<double, 2>> ranges_;
    std::vector<juce::String> names_;
    std::vector<std::vector<double>> base_, live_, next_;
    std::vector<float> levels_;
    double sampleRate_ = 48000;
    std::vector<std::pair<size_t, double>> pending_; // drag preview
    int handle_ = -1, hover_ = -1;
    juce::Rectangle<float> plot() const;
    juce::Rectangle<float> curveArea() const;
    double nextValue(size_t instance, size_t parameter) const;
    double value(size_t instance, size_t parameter, bool authored = false) const;
    double clampParameter(size_t parameter, double v) const;
    std::vector<juce::Point<float>> handles() const;
    void paintEnvelope(juce::Graphics&, juce::Rectangle<float>);
    void paintLowpass(juce::Graphics&, juce::Rectangle<float>);
    void paintPartials(juce::Graphics&, juce::Rectangle<float>);
    void paintDelay(juce::Graphics&, juce::Rectangle<float>);
    void paintWidth(juce::Graphics&, juce::Rectangle<float>);
    void paintReverb(juce::Graphics&, juce::Rectangle<float>);
    void paintGeneric(juce::Graphics&, juce::Rectangle<float>);
};

// The control space of one X layer projected onto the two directions that change the sound most.
//   J  = M_K ... M_l  (composed authored matrices, parameters x controls; clamps are not modelled)
//   J~ = Dp^-1 J Dx   (Dp: parameter range widths, Dx: control half ranges; unbounded or locked rows drop)
//   v1, v2 = eigenvectors of J~^T J~ with the largest eigenvalues (right singular vectors of J~)
//   u = Dx^-1 (x - mid),  position = (v1 . u, v2 . u)
// A drag to position p' moves u by V (p' - p), the smallest change with that projection, then clamps.
struct ControlProjection
{
    bool valid = false;
    size_t layer = 0;
    std::vector<double> mid, half, first, second;
    double firstShare = 0, secondShare = 0;
    // The parameters each direction moves most, signed, strongest first.
    std::vector<std::pair<juce::String, double>> firstMoves, secondMoves;
    // J Dx: parameter change per unit of u, rows are parameters. rowScale is Dp^-1 (0 for locked or
    // unbounded parameters), so rowScale * perUnit is J~.
    std::vector<std::vector<double>> perUnit;
    std::vector<double> rowScale;
};
// Parameters after a step du (in units of each control's half range) from `parameters`, linear through
// the authored matrices and clamped to each parameter's range.
std::vector<double> stepParameters(const ControlProjection&, const std::vector<double>& parameters,
                                   const std::vector<double>& du,
                                   const std::vector<std::array<double, 2>>& ranges);
ControlProjection projectControls(const nlohmann::json& sound, size_t layer,
                                  const std::function<juce::String(size_t)>& parameterName);

class SoundSpace final : public juce::Component, public juce::SettableTooltipClient
{
public:
    explicit SoundSpace(alfa::ui::AlfaLookAndFeel&);
    void setProjection(ControlProjection, juce::StringArray controlNames);
    const ControlProjection& projection() const noexcept { return projection_; }
    // Current control values of the projected layer; adds to the trail when it moved.
    void setValues(const std::vector<double>& values);
    // Parameters at the current values (live when known) and their ranges, for the sensitivity shading.
    void setParameters(const std::vector<double>& parameters, std::vector<std::array<double, 2>> ranges);
    // The current step in control units (half ranges), drawn as an arrow from the current point.
    void setHeading(std::vector<double> du);
    void setMarkers(std::vector<std::pair<juce::String, std::vector<double>>> markers);
    // New layer values while dragging (final = mouse released).
    std::function<void(const std::vector<double>&, bool final)> onMove;
    juce::Point<float> positionOf(const std::vector<double>& values) const;
    void paint(juce::Graphics&) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent&) override;
    void mouseDrag(const juce::MouseEvent&) override;
    void mouseUp(const juce::MouseEvent&) override;

private:
    alfa::ui::AlfaLookAndFeel& look_;
    ControlProjection projection_;
    juce::StringArray names_;
    std::vector<double> values_, dragStart_, parameters_, heading_;
    std::vector<std::array<double, 2>> ranges_;
    juce::Point<float> dragOrigin_;
    std::vector<juce::Point<float>> trail_;
    std::vector<std::pair<juce::String, std::vector<double>>> markers_;
    juce::Image field_;
    juce::Colour fieldColour_;
    bool fieldDirty_ = true;
    double extentX_ = 1, extentY_ = 1;
    bool dragging_ = false;
    juce::Rectangle<float> pad() const;
    juce::Point<float> toScreen(juce::Point<double>) const;
    juce::Point<double> planeOf(const std::vector<double>& values) const;
    std::vector<double> valuesAt(juce::Point<float> to) const;
    void computeField();
};
// What the current heading changes, one row per primitive group: the parameter of the group that moves
// most, as a signed fraction of its range (amber up, blue down).
class HeadingView final : public juce::Component, public juce::SettableTooltipClient
{
public:
    struct Row
    {
        juce::String group, parameter;
        juce::Colour colour;
        double change = 0; // fraction of the parameter's range
    };
    explicit HeadingView(alfa::ui::AlfaLookAndFeel&);
    void setRows(std::vector<Row> rows, juce::String source);
    // Height that shows `rows` groups without empty space at this width.
    static int preferredHeight(size_t rows, int width);
    void paint(juce::Graphics&) override;
    void mouseMove(const juce::MouseEvent&) override;

private:
    alfa::ui::AlfaLookAndFeel& look_;
    std::vector<Row> rows_;
    juce::String source_;
    std::vector<juce::Rectangle<float>> rowAreas_;
};
} // namespace alfa::tensorhost
