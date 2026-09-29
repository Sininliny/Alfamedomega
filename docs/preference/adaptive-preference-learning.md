# Adaptive Preference Learning

- **Status:** Product spec, written by Sinin (2026-09-29). This file is the spec as given, unedited.
- **Design:** [design.md](design.md) describes how it fits the training method and the plugin.

Alfamedomega is not limited to reproducing reference audio. It is designed to learn what an individual user considers a desirable sound. Randomly generated synthesis parameters provide effectively unlimited training data for learning the relationship between DSP structures and audio, while user behavior provides an additional preference signal that teaches the system which regions of that synthesis space are actually musically useful.

Preference learning can happen implicitly during normal production. If a user loads a sound, plays it extensively, keeps it in a project, saves it, reuses it, or makes only minor adjustments instead of immediately replacing it, Alfamedomega can treat these actions as weak positive feedback. Sounds that are rapidly discarded provide negative evidence. These signals can be combined using contextual preference learning or reinforcement-learning-style optimization, allowing future sound generation to gradually move toward the user's personal timbral preferences rather than merely producing technically valid random sounds.

The user may also provide explicit positive examples by supplying any collection of sounds they consider desirable. Alfamedomega can optionally monitor a user-selected sample or preset directory, analyze its audio contents, and learn semantic relationships from filenames and directory structure. A library containing names such as `warm_pad`, `airy_vocal`, `glass_bell`, `dark_bass`, or `aggressive_lead` therefore becomes both a timbral preference dataset and a weakly labeled text-audio dataset, allowing text descriptions to become associated with regions of the learned synthesis space.

This creates two complementary learning objectives:

```
Reference WAV
    ↓
"What sound should I reproduce?"
    ↓
Inverse synthesis


User library + production behavior
    ↓
"What sounds does this user tend to like?"
    ↓
Preference model


Text labels / filenames
    ↓
"What does this user mean by 'airy', 'warm', or 'dark'?"
    ↓
Personal text–timbre mapping
```

The long-term goal is therefore not merely to generate a patch that resembles an input waveform, but to generate new synthesizable sounds that are increasingly likely to match the user's own aesthetic preferences, while allowing those preferences to be expressed through examples, natural-language labels, and ordinary production behavior.

For privacy and reproducibility, preference learning can be performed locally, with source libraries remaining on the user's machine; only derived embeddings, model updates, or preference weights need to be retained.
