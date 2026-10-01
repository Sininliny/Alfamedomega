#pragma once

#include <juce_core/juce_core.h>

#include <cstddef>

namespace alfa::tensorhost
{
// Message-thread composition. Stable fragment IDs are independent of tensor
// offsets; lowering remaps all coordinates and validates a complete executable.
class TensorComposition
{
public:
    struct Binding
    {
        juce::String id, sourceFragment, destinationFragment;
        // Only prior committed state/output is accepted for inter-fragment links.
        juce::String source = "output_history";
        std::size_t sourceElement = 0;
        // "parameter" or "audio". Audio selects a local primitive instance.
        juce::String destination = "audio";
        std::size_t destinationInstance = 0, destinationElement = 0;
        juce::String weights = "routing";
        double weight = 1;
    };

    // Budget is explicit JSON with the same capacity fields as an AOM document.
    // An optional target tier explicitly authorizes uniform precision conversion.
    TensorComposition(double sampleRate, juce::String budgetJSON, juce::String targetTier = {});
    bool add(const juce::String& fragmentId, const juce::String& document, juce::String& error);
    bool clone(const juce::String& existingId, const juce::String& newId, juce::String& error);
    bool remove(const juce::String& fragmentId, juce::String& error);
    bool bind(const Binding&, juce::String& error);
    bool unbind(const juce::String& bindingId, juce::String& error);
    bool restore(const juce::String& recipeJSON, juce::String& error);
    // Reopen a sound only when its flattened data still matches the recipe.
    // Explicitly edited flattened documents must be added as a new fragment.
    bool restoreDocument(const juce::String& documentJSON, juce::String& error);
    [[nodiscard]] juce::String recipe() const { return recipe_; }
    [[nodiscard]] juce::String document() const { return document_; }
    [[nodiscard]] juce::StringArray fragmentIds() const;

private:
    double sampleRate_;
    juce::String recipe_, document_;
    bool commit(const juce::String& candidateRecipe, juce::String& error);
};
} // namespace alfa::tensorhost
