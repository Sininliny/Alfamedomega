#include "alfa/plugin/PreferenceLog.h"

namespace alfa::preference
{

juce::File controlCenterHome()
{
    const auto env = juce::SystemStats::getEnvironmentVariable("ALFA_CC_HOME", {});
    if (env.isNotEmpty())
        return juce::File(env);

#if JUCE_MAC
    return juce::File::getSpecialLocation(juce::File::userHomeDirectory)
        .getChildFile("Library/Application Support/Alfamedomega/ControlCenter");
#elif JUCE_WINDOWS
    auto appData = juce::SystemStats::getEnvironmentVariable("APPDATA", {});
    auto base = appData.isNotEmpty() ? juce::File(appData)
                                     : juce::File::getSpecialLocation(juce::File::userHomeDirectory);
    return base.getChildFile("Alfamedomega").getChildFile("ControlCenter");
#else
    const auto xdg = juce::SystemStats::getEnvironmentVariable("XDG_DATA_HOME", {});
    auto base =
        xdg.isNotEmpty()
            ? juce::File(xdg)
            : juce::File::getSpecialLocation(juce::File::userHomeDirectory).getChildFile(".local/share");
    return base.getChildFile("Alfamedomega").getChildFile("ControlCenter");
#endif
}

// LearningSwitch
// ---------------------------------------------------------------------------

Consent Consent::fromVar(const juce::var& json)
{
    Consent c;
    c.learn = json.getProperty("learn", false).isBool() && static_cast<bool>(json["learn"]);
    const auto cats = json.getProperty("categories", {});
    const auto on = [&](const char* name)
    {
        if (!cats.isObject())
            return c.learn;
        const auto v = cats.getProperty(name, false);
        return c.learn && v.isBool() && static_cast<bool>(v);
    };
    c.performance = on("performance");
    c.parameters = on("parameters");
    c.suggestions = on("suggestions");
    c.workflow = on("workflow");
    const auto pers = json.getProperty("personalization", 1.0);
    c.personalization = pers.isDouble() || pers.isInt() || pers.isInt64()
                            ? juce::jlimit(0.0f, 1.0f, static_cast<float>(static_cast<double>(pers)))
                            : 1.0f;
    return c;
}

LearningSwitch::LearningSwitch(juce::File preferenceFolder)
    : folder(std::move(preferenceFolder))
{
}

Consent LearningSwitch::consent() const
{
    // Another instance or the control center may change it; re-read the file
    // when it changes, at most once a second.
    const auto now = juce::Time::getCurrentTime();
    if (lastChecked != juce::Time() && (now - lastChecked).inSeconds() < 1.0)
        return cached;
    lastChecked = now;

    const auto file = folder.getChildFile("plugin.json");
    const auto modified = file.getLastModificationTime();
    if (modified == lastModified && lastModified != juce::Time())
        return cached;
    lastModified = modified;
    cached = file.existsAsFile() ? Consent::fromVar(juce::JSON::parse(file)) : Consent{};
    return cached;
}

void LearningSwitch::setOn(bool on)
{
    auto obj = juce::JSON::parse(folder.getChildFile("plugin.json"));
    if (!obj.isObject())
        obj = juce::var(new juce::DynamicObject());
    obj.getDynamicObject()->setProperty("learn", on);
    write(obj);
}

void LearningSwitch::setConsent(const Consent& c)
{
    auto obj = juce::JSON::parse(folder.getChildFile("plugin.json"));
    if (!obj.isObject())
        obj = juce::var(new juce::DynamicObject());
    auto* cats = new juce::DynamicObject();
    cats->setProperty("performance", c.performance);
    cats->setProperty("parameters", c.parameters);
    cats->setProperty("suggestions", c.suggestions);
    cats->setProperty("workflow", c.workflow);
    obj.getDynamicObject()->setProperty("learn", c.learn);
    obj.getDynamicObject()->setProperty("categories", juce::var(cats));
    obj.getDynamicObject()->setProperty("personalization", std::round(c.personalization * 100.0f) / 100.0f);
    write(obj);
}

void LearningSwitch::write(const juce::var& obj)
{
    const auto file = folder.getChildFile("plugin.json");
    folder.createDirectory();
    file.replaceWithText(juce::JSON::toString(obj, true));
    cached = Consent::fromVar(obj);
    lastChecked = juce::Time::getCurrentTime();
    lastModified = file.getLastModificationTime();
}

juce::StringArray LearningSwitch::learnedSummary() const
{
    juce::File newest;
    for (const auto& f :
         folder.getChildFile("profiles").findChildFiles(juce::File::findFiles, true, "summary.json"))
        if (newest == juce::File() || f.getLastModificationTime() > newest.getLastModificationTime())
            newest = f;

    juce::StringArray lines;
    const auto explain =
        newest.existsAsFile() ? juce::JSON::parse(newest).getProperty("explain", {}) : juce::var();
    if (const auto* array = explain.getArray())
        for (const auto& line : *array)
            lines.add(line.toString());
    return lines;
}

// Patches
// ---------------------------------------------------------------------------

juce::var patchToVar(const params::SoundShape& shape, const std::array<float, params::kNumKnobs>& knobs)
{
    auto obj = new juce::DynamicObject();
    obj->setProperty("method", params::kMethods[static_cast<size_t>(shape.method)].id);
    if (shape.method == params::Method::hybrid)
    {
        juce::Array<juce::var> pair{params::kMethods[static_cast<size_t>(shape.hybridA)].id,
                                    params::kMethods[static_cast<size_t>(shape.hybridB)].id};
        obj->setProperty("pair", pair);
    }
    else
    {
        obj->setProperty("pair", juce::var());
    }
    juce::Array<juce::var> k;
    for (auto v : knobs)
        k.add(std::round(static_cast<double>(v) * 10000.0) / 10000.0);
    obj->setProperty("knobs", k);
    return obj;
}

juce::var capturePatch(const juce::AudioProcessorValueTreeState& state)
{
    std::array<float, params::kNumKnobs> knobs{};
    for (int i = 0; i < params::kNumKnobs; ++i)
        if (const auto* p = state.getRawParameterValue(params::id::knob(i)))
            knobs[static_cast<size_t>(i)] = p->load();
    return patchToVar(params::readShape(state), knobs);
}

// PlayCounter
// ---------------------------------------------------------------------------

void PlayCounter::process(const juce::MidiBuffer& midi, int numSamples) noexcept
{
    int64_t heldAcc = 0;
    int pos = 0;
    for (const auto meta : midi)
    {
        const auto at = juce::jlimit(0, numSamples, meta.samplePosition);
        heldAcc += static_cast<int64_t>(heldCount) * (at - pos);
        pos = at;

        const auto msg = meta.getMessage();
        if (msg.isNoteOn())
        {
            const auto n = static_cast<size_t>(msg.getNoteNumber());
            if (!held[n])
            {
                held[n] = true;
                ++heldCount;
            }
            onAt[n] = clock + at;
            notes.fetch_add(1, std::memory_order_relaxed);
            pitchSum.fetch_add(msg.getNoteNumber(), std::memory_order_relaxed);
        }
        else if (msg.isNoteOff())
        {
            const auto n = static_cast<size_t>(msg.getNoteNumber());
            if (held[n])
            {
                held[n] = false;
                --heldCount;
                lengthSamples.fetch_add(clock + at - onAt[n], std::memory_order_relaxed);
                endedNotes.fetch_add(1, std::memory_order_relaxed);
            }
        }
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
        {
            held.fill(false);
            heldCount = 0;
        }
    }
    heldAcc += static_cast<int64_t>(heldCount) * (numSamples - pos);
    heldSamples.fetch_add(heldAcc, std::memory_order_relaxed);
    clock += numSamples;
}

void PlayCounter::reset() noexcept
{
    take();
}

PlayCounter::Totals PlayCounter::take() noexcept
{
    Totals t;
    t.heldSamples = heldSamples.exchange(0);
    t.notes = notes.exchange(0);
    t.pitchSum = pitchSum.exchange(0);
    t.lengthSamples = lengthSamples.exchange(0);
    t.endedNotes = endedNotes.exchange(0);
    return t;
}

// EventWriter
// ---------------------------------------------------------------------------

EventWriter::EventWriter()
    : juce::Thread("Alfamedomega event log")
{
    startThread(juce::Thread::Priority::background);
}

EventWriter::~EventWriter()
{
    signalThreadShouldExit();
    wake.signal();
    stopThread(2000);
    drain();
}

void EventWriter::append(const juce::File& file, const juce::String& line)
{
    {
        const juce::ScopedLock sl(lock);
        queue.emplace_back(file, line);
    }
    wake.signal();
}

void EventWriter::flush()
{
    drain();
}

void EventWriter::run()
{
    while (!threadShouldExit())
    {
        wake.wait(1000);
        drain();
    }
}

void EventWriter::drain()
{
    std::vector<std::pair<juce::File, juce::String>> batch;
    {
        const juce::ScopedLock sl(lock);
        batch.swap(queue);
    }
    // Group consecutive lines per file so each file is opened once per batch.
    size_t i = 0;
    while (i < batch.size())
    {
        const auto& file = batch[i].first;
        juce::String text;
        size_t j = i;
        for (; j < batch.size() && batch[j].first == file; ++j)
            text << batch[j].second << "\n";
        file.getParentDirectory().createDirectory();
        juce::FileOutputStream out(file); // opens at the end of an existing file
        if (out.openedOk())
            out.writeText(text, false, false, "\n");
        i = j;
    }
}

// Recorder
// ---------------------------------------------------------------------------

juce::String sourceName(Recorder::Source source)
{
    switch (source)
    {
    case Recorder::Source::browser:
        return "browser";
    case Recorder::Source::project:
        return "project";
    case Recorder::Source::generate:
        return "generate";
    case Recorder::Source::init:
        return "init";
    case Recorder::Source::user:
        break;
    }
    return "user";
}

Recorder::Recorder(juce::AudioProcessorValueTreeState& s, PlayCounter& c, Options o)
    : state(s)
    , counter(c)
    , options(std::move(o))
    , inst(juce::Uuid().toString().substring(0, 16))
{
    sampleRate.store(options.sampleRate);
    createdAt = now();
    if (options.startTimer)
        startTimerHz(1);
}

Recorder::~Recorder()
{
    stopTimer();
    if (hasExposure && learning())
        emit("close");
}

double Recorder::now() const
{
    return options.now ? options.now() : static_cast<double>(juce::Time::currentTimeMillis()) / 1000.0;
}

bool Recorder::learning() const
{
    return options.learning != nullptr && options.learning->isOn();
}

Consent Recorder::consent() const
{
    return options.learning != nullptr ? options.learning->consent() : Consent{};
}

void Recorder::emit(const juce::String& type, juce::DynamicObject::Ptr fields)
{
    auto obj = fields != nullptr ? fields : juce::DynamicObject::Ptr(new juce::DynamicObject());
    obj->setProperty("t", std::round(now() * 1000.0) / 1000.0);
    obj->setProperty("type", type);
    obj->setProperty("inst", inst);

    const auto date = juce::Time(static_cast<juce::int64>(now() * 1000.0)).formatted("%Y-%m-%d");
    const auto file = options.learning->eventsFolder().getChildFile(date + ".jsonl");
    writer->append(file, juce::JSON::toString(juce::var(obj.get()), juce::JSON::FormatOptions{}.withSpacing(
                                                                        juce::JSON::Spacing::none)));
}

void Recorder::startExposure(Source source)
{
    counter.reset(); // what was played before belongs to no exposure we log
    hasExposure = true;
    exposureStart = now();
    kept = false;
    currentRating = 0;
    lastPatch = capturePatch(state);
    pendingPatch = juce::var();

    juce::DynamicObject::Ptr f = new juce::DynamicObject();
    f->setProperty("patch", lastPatch);
    f->setProperty("source", sourceName(source));
    if (consent().workflow)
        f->setProperty("project", inst); // no host exposes a project id; the instance id stands in
    emit("load", f);
}

void Recorder::endExposure()
{
    hasExposure = false;
    pendingPatch = juce::var();
}

void Recorder::flushPlay()
{
    const auto p = counter.take();
    if (p.notes == 0 && p.heldSamples == 0)
        return;
    const auto c = consent();
    if (!c.workflow && !c.performance)
        return;
    const auto sr = sampleRate.load();
    juce::DynamicObject::Ptr f = new juce::DynamicObject();
    if (c.workflow)
        f->setProperty("seconds", std::round(static_cast<double>(p.heldSamples) / sr * 100.0) / 100.0);
    if (c.performance)
    {
        f->setProperty("notes", p.notes);
        f->setProperty("pitch_mean", p.notes > 0 ? static_cast<double>(p.pitchSum) / p.notes : 60.0);
        f->setProperty("len_mean",
                       p.endedNotes > 0 ? static_cast<double>(p.lengthSamples) / p.endedNotes / sr : 0.5);
    }
    emit("play", f);
}

void Recorder::loaded(Source source)
{
    pendingLoad = source;
    tick();
}

void Recorder::projectRestored(const juce::String& instanceId)
{
    {
        const juce::ScopedLock sl(restoreLock);
        restoredId = instanceId;
    }
    restoreRequested.store(true);
}

void Recorder::rate(int value)
{
    currentRating = juce::jlimit(-1, 1, value);
    if (currentRating == 0 || !consent().suggestions)
        return;
    if (!hasExposure)
        startExposure(Source::user);
    juce::DynamicObject::Ptr f = new juce::DynamicObject();
    f->setProperty("value", currentRating);
    emit("rate", f);
}

void Recorder::tick()
{
    const auto t = now();

    if (restoreRequested.exchange(false))
    {
        const juce::ScopedLock sl(restoreLock);
        if (restoredId.isNotEmpty())
            inst = restoredId;
        pendingLoad = Source::project;
    }

    if (!learning())
    {
        if (hasExposure)
            endExposure();
        counter.reset();
        pendingLoad.reset();
        saveRequested.store(false);
        return;
    }

    if (pendingLoad.has_value())
    {
        if (hasExposure)
            flushPlay();
        startExposure(*pendingLoad);
        pendingLoad.reset();
        return;
    }

    if (!hasExposure)
    {
        // A fresh instance: wait for a project restore before calling it "init".
        if (t - createdAt < options.initDelaySeconds)
            return;
        startExposure(Source::init);
        return;
    }

    flushPlay();

    // Edits: log the sound once it has stopped changing for a moment.
    const auto patch = capturePatch(state);
    if (juce::JSON::toString(patch) != juce::JSON::toString(lastPatch))
    {
        if (juce::JSON::toString(patch) != juce::JSON::toString(pendingPatch))
        {
            pendingPatch = patch;
            pendingSince = t;
        }
        else if (t - pendingSince >= options.editSettleSeconds)
        {
            if (consent().parameters)
            {
                juce::DynamicObject::Ptr f = new juce::DynamicObject();
                f->setProperty("patch", patch);
                emit("edit", f);
            }
            lastPatch = patch;
            pendingPatch = juce::var();
        }
    }

    // Hosts ask for state on autosave and undo too, so only count a save as
    // "kept" once per exposure and after the sound has been loaded a while.
    if (saveRequested.exchange(false) && !kept && t - exposureStart >= options.keptAfterSeconds)
    {
        kept = true;
        if (consent().workflow)
            emit("project_save");
    }
}

} // namespace alfa::preference
