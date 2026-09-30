// alfa-render-host: headless plugin host for Local Plugin Assimilation.
//
//   alfa-render-host formats                     formats this build can load, as JSON
//   alfa-render-host scan [--programs] FILE...   plugins inside each file, as JSON
//   alfa-render-host render JOB.json             render notes for presets; progress on stdout
//
// The control center (alfacc.assimilate) starts one process per plugin, so a plugin that
// crashes or hangs costs one job, not the app. Protocol: docs/control-center/plugin-assimilation.md.
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <cmath>
#include <iostream>

using namespace juce;

namespace
{
void out (const String& line)
{
    std::cout << line.toStdString() << std::endl; // endl flushes, so progress arrives live
}

[[noreturn]] void fail (const String& message, int code = 2)
{
    std::cerr << "error: " << message.toStdString() << std::endl;
    std::exit (code);
}

var toJson (const PluginDescription& d)
{
    auto* o = new DynamicObject();
    o->setProperty ("name", d.name);
    o->setProperty ("vendor", d.manufacturerName);
    o->setProperty ("format", d.pluginFormatName);
    o->setProperty ("uid", d.createIdentifierString());
    o->setProperty ("file", d.fileOrIdentifier);
    o->setProperty ("version", d.version);
    o->setProperty ("category", d.category);
    o->setProperty ("is_instrument", d.isInstrument);
    o->setProperty ("inputs", d.numInputChannels);
    o->setProperty ("outputs", d.numOutputChannels);
    return var (o);
}

struct Host
{
    Host() { formats.addDefaultFormats(); }

    OwnedArray<PluginDescription> find (const String& file)
    {
        OwnedArray<PluginDescription> found;
        for (auto* f : formats.getFormats())
            if (f->fileMightContainThisPluginType (file))
                f->findAllTypesForFile (found, file);
        return found;
    }

    std::unique_ptr<AudioPluginInstance> open (const PluginDescription& d, double sr, int block)
    {
        String err;
        auto p = formats.createPluginInstance (d, sr, block, err);
        if (p == nullptr)
            fail ("could not open " + d.name + ": " + err, 3);
        return p;
    }

    AudioPluginFormatManager formats;
};

int cmdFormats (Host& host)
{
    Array<var> names;
    for (auto* f : host.formats.getFormats())
        names.add (f->getName());
    out (JSON::toString (var (names), true));
    return 0;
}

int cmdScan (Host& host, const StringArray& args)
{
    const bool withPrograms = args.contains ("--programs");
    Array<var> plugins;
    for (const auto& file : args)
    {
        if (file.startsWith ("--"))
            continue;
        for (auto* d : host.find (file))
        {
            auto v = toJson (*d);
            if (withPrograms && d->isInstrument)
            {
                auto p = host.open (*d, 48000.0, 512);
                Array<var> programs;
                for (int i = 0; i < p->getNumPrograms(); ++i)
                    programs.add (p->getProgramName (i));
                v.getDynamicObject()->setProperty ("programs", programs);
                // The scanned description can report 0 channels before a bus layout exists.
                v.getDynamicObject()->setProperty ("outputs", p->getTotalNumOutputChannels());
            }
            plugins.add (v);
        }
    }
    auto* o = new DynamicObject();
    o->setProperty ("plugins", plugins);
    out (JSON::toString (var (o), true));
    return 0;
}

// Render blocks until `samples` have been produced, appending them to `dest` if given.
struct Renderer
{
    Renderer (AudioPluginInstance& p, double sr, int block) : plugin (p), sampleRate (sr), blockSize (block)
    {
        const int ch = jmax (2, plugin.getTotalNumInputChannels(), plugin.getTotalNumOutputChannels());
        buffer.setSize (ch, blockSize);
        plugin.setNonRealtime (true);
        plugin.prepareToPlay (sampleRate, blockSize);
    }

    ~Renderer() { plugin.releaseResources(); }

    void silence (double seconds)
    {
        MidiBuffer midi;
        for (int64 left = (int64) (seconds * sampleRate); left > 0; left -= blockSize)
            process (midi);
    }

    // Returns the rendered audio: note held for dur_s, then the tail.
    AudioBuffer<float> note (int midiNote, int velocity, double durS, double tailS, bool autoTail)
    {
        const int outCh = jlimit (1, 2, plugin.getTotalNumOutputChannels());
        const auto noteLen = (int64) std::llround (durS * sampleRate);
        const auto maxLen = noteLen + (int64) std::llround (tailS * sampleRate);
        AudioBuffer<float> result (outCh, (int) maxLen);
        result.clear();
        int64 pos = 0;
        int64 quietSince = -1;
        const auto quietNeeded = (int64) (0.05 * sampleRate);
        bool offSent = false;
        while (pos < maxLen)
        {
            MidiBuffer midi;
            if (pos == 0)
                midi.addEvent (MidiMessage::noteOn (1, midiNote, (uint8) jlimit (1, 127, velocity)), 0);
            if (! offSent && noteLen < pos + blockSize)
            {
                midi.addEvent (MidiMessage::noteOff (1, midiNote), (int) jmax ((int64) 0, noteLen - pos));
                offSent = true;
            }
            process (midi);
            const int n = (int) jmin ((int64) blockSize, maxLen - pos);
            for (int ch = 0; ch < outCh; ++ch)
                result.copyFrom (ch, (int) pos, buffer, ch, 0, n);
            if (autoTail && offSent && pos > noteLen)
            {
                const float mag = buffer.getMagnitude (0, n);
                if (mag < 3.2e-5f) // -90 dBFS
                    quietSince = quietSince < 0 ? pos : quietSince;
                else
                    quietSince = -1;
                if (quietSince >= 0 && pos + n - quietSince >= quietNeeded)
                {
                    pos += n;
                    break;
                }
            }
            pos += n;
        }
        result.setSize (outCh, (int) jmin (pos, maxLen), true);
        // Clear what is left of this note so the next one starts clean.
        MidiBuffer off;
        off.addEvent (MidiMessage::allNotesOff (1), 0);
        off.addEvent (MidiMessage::allSoundOff (1), 0);
        process (off);
        plugin.reset();
        return result;
    }

    void process (MidiBuffer& midi)
    {
        buffer.clear();
        plugin.processBlock (buffer, midi);
    }

    AudioPluginInstance& plugin;
    double sampleRate;
    int blockSize;
    AudioBuffer<float> buffer;
};

bool writeWav (const File& file, const AudioBuffer<float>& audio, double sr)
{
    file.getParentDirectory().createDirectory();
    file.deleteFile();
    std::unique_ptr<OutputStream> stream = std::make_unique<FileOutputStream> (file);
    WavAudioFormat wav;
    auto writer = wav.createWriterFor (stream, AudioFormatWriterOptions {}
                                                   .withSampleRate (sr)
                                                   .withNumChannels (audio.getNumChannels())
                                                   .withBitsPerSample (32)
                                                   .withSampleFormat (AudioFormatWriterOptions::SampleFormat::floatingPoint));
    return writer != nullptr && writer->writeFromAudioSampleBuffer (audio, 0, audio.getNumSamples());
}

bool applyPreset (AudioPluginInstance& p, const var& preset, String& err)
{
    const auto kind = preset["kind"].toString();
    if (kind == "default")
        return true;
    if (kind == "program")
    {
        const int i = preset["index"];
        if (i < 0 || i >= p.getNumPrograms())
        {
            err = "program " + String (i) + " out of range";
            return false;
        }
        p.setCurrentProgram (i);
        return true;
    }
    if (kind == "vstpreset")
    {
        MemoryBlock data;
        if (! File (preset["path"].toString()).loadFileAsData (data))
        {
            err = "can't read " + preset["path"].toString();
            return false;
        }
#if JUCE_PLUGINHOST_VST3
        if (VST3PluginFormat::setStateFromVSTPresetFile (&p, data))
            return true;
#endif
        err = "plugin refused the .vstpreset";
        return false;
    }
    if (kind == "state")
    {
        MemoryBlock data;
        if (! data.fromBase64Encoding (preset["data"].toString()))
        {
            err = "bad state data";
            return false;
        }
        p.setStateInformation (data.getData(), (int) data.getSize());
        return true;
    }
    err = "unknown preset kind " + kind;
    return false;
}

int cmdRender (Host& host, const File& jobFile)
{
    const auto job = JSON::parse (jobFile);
    if (! job.isObject())
        fail ("can't parse " + jobFile.getFullPathName());
    const auto file = job["plugin"].toString();
    const auto uid = job["uid"].toString();
    const double sr = job.getProperty ("sample_rate", 48000.0);
    const int block = job.getProperty ("block", 512);
    const double settle = job.getProperty ("settle_s", 0.3);
    const File outDir (job["out_dir"].toString());

    auto found = host.find (file);
    PluginDescription* desc = nullptr;
    for (auto* d : found)
        if (uid.isEmpty() || d->createIdentifierString() == uid)
            desc = d;
    if (desc == nullptr)
        fail ("no plugin " + (uid.isEmpty() ? file : uid) + " in " + file, 3);

    auto plugin = host.open (*desc, sr, block);
    // Instruments get stereo out and no audio in, if the plugin allows it.
    auto layout = plugin->getBusesLayout();
    for (auto& in : layout.inputBuses)
        in = AudioChannelSet::disabled();
    if (! layout.outputBuses.isEmpty())
        layout.outputBuses.getReference (0) = AudioChannelSet::stereo();
    plugin->setBusesLayout (layout);

    const auto* presets = job["presets"].getArray();
    const auto* notes = job["notes"].getArray();
    if (presets == nullptr || notes == nullptr)
        fail ("job needs presets and notes arrays");
    const int total = presets->size() * notes->size();
    int done = 0;
    Array<var> results;

    Renderer r (*plugin, sr, block);
    for (const auto& preset : *presets)
    {
        String err;
        const auto key = preset["key"].toString();
        if (! applyPreset (*plugin, preset, err))
        {
            out ("warning: preset " + key + ": " + err);
            done += notes->size();
            out ("progress " + String (done) + "/" + String (total));
            continue;
        }
        r.silence (settle); // let the plugin load samples and settle after a preset change
        for (const auto& n : *notes)
        {
            const auto id = n["id"].toString();
            auto audio = r.note (n["midi"], n["velocity"], n["dur_s"], n["tail_s"],
                                 n.getProperty ("tail_mode", "auto").toString() == "auto");
            const auto wavFile = outDir.getChildFile (key).getChildFile (id + ".wav");
            if (! writeWav (wavFile, audio, sr))
                fail ("can't write " + wavFile.getFullPathName(), 4);
            auto* o = new DynamicObject();
            o->setProperty ("preset", key);
            o->setProperty ("note", id);
            o->setProperty ("wav", wavFile.getFullPathName());
            o->setProperty ("seconds", audio.getNumSamples() / sr);
            o->setProperty ("peak", audio.getMagnitude (0, audio.getNumSamples()));
            results.add (var (o));
            out ("progress " + String (++done) + "/" + String (total));
        }
    }
    auto* o = new DynamicObject();
    o->setProperty ("plugin", toJson (*desc));
    o->setProperty ("renders", results);
    const auto resultFile = outDir.getChildFile ("renders.json");
    resultFile.replaceWithText (JSON::toString (var (o)));
    out ("result " + resultFile.getFullPathName());
    return 0;
}
} // namespace

int main (int argc, char** argv)
{
    ScopedJuceInitialiser_GUI juce; // plugins expect a message thread; this thread becomes it
    StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (String::fromUTF8 (argv[i]));
    if (args.isEmpty() || args[0] == "--help" || args[0] == "-h")
    {
        out ("usage: alfa-render-host formats | scan [--programs] FILE... | render JOB.json");
        return args.isEmpty() ? 1 : 0;
    }
    Host host;
    const auto cmd = args[0];
    args.remove (0);
    if (cmd == "formats")
        return cmdFormats (host);
    if (cmd == "scan")
        return cmdScan (host, args);
    if (cmd == "render" && args.size() == 1)
        return cmdRender (host, File::getCurrentWorkingDirectory().getChildFile (args[0]));
    fail ("unknown command " + cmd, 1);
}
