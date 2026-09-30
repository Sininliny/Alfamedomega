"""Hook for an optional general text-audio model (design.md s5.4). None ships in this version.

Text search understands only the user's own words: preset and folder names, tags, descriptions
and hand labels (model.WordModel). A general model, such as a CLAP-style text-audio embedding,
could later help with words the user has never used. The rule is that the user's vocabulary is
the final authority:

- a word the user has used is scored only by the user's word map, whatever a general model says;
- a general model scores only words the user has never used, at a lower weight, and the UI says
  that the word was guessed rather than learned;
- the general model runs locally and is optional; without it, unknown words are ignored and
  reported back to the user.

A general model is any object with `score(word, descriptor) -> float` in roughly the same units
as a word's log-likelihood ratio (0 = no opinion, positive = the sound fits the word).
"""

GENERAL_WEIGHT = 0.5


class GeneralTextModel:
    """Interface only. A real one would load a local model file and embed text and audio."""

    name = "none"

    def score(self, word, descriptor):
        raise NotImplementedError


def text_score(words, z, descriptor, parsed, unknown, general=None, weight=GENERAL_WEIGHT):
    """User words through the personal map; unknown words through the general model, if any.
    parsed: [(word, sign)] the user knows; unknown: [(word, sign)] the user has never used."""
    s = words.query_score(z, parsed) if (words is not None and parsed) else 0.0
    if general is not None:
        s += weight * sum(sign * general.score(w, descriptor) for w, sign in unknown)
    return s
