#include "alfa/ui/Theme.h"

namespace alfa::ui
{
namespace
{
constexpr const char* graphiteJson = R"({
  "name": "Graphite",
  "bg": "#111317", "panel": "#1B1E24", "line": "#2A2E36",
  "text": "#EBE9E4", "text2": "#8B9099",
  "accent": "#F2A35E", "mod": "#6AA2F5", "warn": "#E27B62"
})";

constexpr const char* paperJson = R"({
  "name": "Paper",
  "bg": "#E9E7E2", "panel": "#F4F3EF", "line": "#DDDAD3",
  "text": "#23252A", "text2": "#6B6E75",
  "accent": "#D9822F", "mod": "#3B78D8", "warn": "#C0502F"
})";

constexpr const char* contrastJson = R"({
  "name": "High contrast",
  "bg": "#000000", "panel": "#0B0B0B", "line": "#6E6E6E",
  "text": "#FFFFFF", "text2": "#E6E6E6",
  "accent": "#FFB000", "mod": "#5CC8FF", "warn": "#FF6B4A"
})";

juce::Colour parseColour(const juce::var& v, juce::Colour fallback)
{
    auto s = v.toString().trim().removeCharacters("#");
    if (s.length() != 6 && s.length() != 8)
        return fallback;
    if (s.length() == 6)
        s = "ff" + s;
    return juce::Colour(static_cast<juce::uint32>(s.getHexValue64()));
}

juce::String toHex(juce::Colour c)
{
    return "#" + c.toDisplayString(false);
}
} // namespace

Theme Theme::fromJson(const juce::String& json)
{
    static const Theme fallback = []
    {
        Theme t;
        t.name = "Graphite";
        t.bg = juce::Colour(0xff111317);
        t.panel = juce::Colour(0xff1b1e24);
        t.line = juce::Colour(0xff2a2e36);
        t.text = juce::Colour(0xffebe9e4);
        t.text2 = juce::Colour(0xff8b9099);
        t.accent = juce::Colour(0xfff2a35e);
        t.mod = juce::Colour(0xff6aa2f5);
        t.warn = juce::Colour(0xffe27b62);
        return t;
    }();

    const auto parsed = juce::JSON::parse(json);
    if (!parsed.isObject())
        return fallback;

    Theme t;
    t.name = parsed.getProperty("name", fallback.name).toString();
    t.bg = parseColour(parsed["bg"], fallback.bg);
    t.panel = parseColour(parsed["panel"], fallback.panel);
    t.line = parseColour(parsed["line"], fallback.line);
    t.text = parseColour(parsed["text"], fallback.text);
    t.text2 = parseColour(parsed["text2"], fallback.text2);
    t.accent = parseColour(parsed["accent"], fallback.accent);
    t.mod = parseColour(parsed["mod"], fallback.mod);
    t.warn = parseColour(parsed["warn"], fallback.warn);
    return t;
}

juce::String Theme::toJson() const
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty("name", name);
    obj->setProperty("bg", toHex(bg));
    obj->setProperty("panel", toHex(panel));
    obj->setProperty("line", toHex(line));
    obj->setProperty("text", toHex(text));
    obj->setProperty("text2", toHex(text2));
    obj->setProperty("accent", toHex(accent));
    obj->setProperty("mod", toHex(mod));
    obj->setProperty("warn", toHex(warn));
    return juce::JSON::toString(juce::var(obj));
}

Theme Theme::graphite()
{
    return fromJson(graphiteJson);
}
Theme Theme::paper()
{
    return fromJson(paperJson);
}
Theme Theme::highContrast()
{
    return fromJson(contrastJson);
}

Theme Theme::builtIn(const juce::String& id)
{
    if (id == "paper")
        return paper();
    if (id == "contrast")
        return highContrast();
    return graphite();
}

juce::StringArray Theme::builtInIds()
{
    return {"graphite", "paper", "contrast"};
}
juce::StringArray Theme::builtInNames()
{
    return {"Graphite", "Paper", "High contrast"};
}

} // namespace alfa::ui
