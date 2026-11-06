Alfamedomega Suite (plugin + standalone + AMO Centre)
=====================================================

Install (Windows 10/11, 64-bit)
1. Unzip the whole archive into any folder (don't run it from inside the zip).
2. Double-click Install.bat. The first install downloads Python, PyTorch and a few other
   parts, about 400 MB, so it takes a few minutes.
   Windows will ask for administrator rights once: that is only to copy the VST3 plugin
   into C:\Program Files\Common Files\VST3. Click "Yes".
3. When the window says "Done." you're finished. You can delete the unzipped folder.

After installing
- Two icons appear on your desktop:
  Alfamedomega   the standalone synth (plays without a DAW; pick your audio device and
                 MIDI keyboard in its settings)
  AMO Centre     the control centre
- In your DAW (FL Studio, Ableton Live, Reaper, ...) rescan plugins and load
  Alfamedomega (VST3).

Turn a sample into a sound: WAV to Preset
1. Open AMO Centre. "WAV to Preset" is the first page; the model is already built in.
2. Drag .wav files onto the dashed box (or click it to choose files). Single-note
   one-shots work best: one note, a clean start, no reverb, no other instruments.
3. The pitch of the sample is detected automatically. If it's wrong, choose the sample's
   note under "Sample pitch".
4. When it's done, open the plugin's (or the standalone's) sound library and press
   "Refresh library". The new sounds are in the "WAV import" category.
   The preset files are saved in %APPDATA%\Alfamedomega\TensorPresets.

Good to know
- This is not sample playback. The model looks for the closest sound that our own synth
  structures can make, then builds a playable preset from it. The more a sample sounds
  like a synth, the closer the result; voices, full songs and complex acoustic
  instruments only come out roughly similar.
- Very low notes (below about F#1 / MIDI 34) give weaker results.
- Conversion runs on the CPU and takes from a few seconds to about a minute per sample.
  It only uses two threads, so the computer stays quiet.
- Start AMO Centre from its desktop or Start-menu shortcut.
- Uninstall: Start menu > Alfamedomega > Uninstall Alfamedomega. Presets you made are kept.

If something goes wrong, send a screenshot of the window to Sinin.
