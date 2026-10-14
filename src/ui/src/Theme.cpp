#include "alfa/ui/Theme.h"

namespace alfa::ui
{
namespace
{
constexpr const char* graphiteJson = R"({
  "name": "Graphite",
  "bg": "#17181C", "panel": "#202227", "line": "#2E3138",
  "text": "#EAE6DF", "text2": "#A29E95",
  "accent": "#E2A54E", "mod": "#7BA9D8", "warn": "#DE8068"
})";

constexpr const char* paperJson = R"({
  "name": "Paper",
  "bg": "#ECE8E1", "panel": "#F7F5F0", "line": "#D9D3C8",
  "text": "#2A2824", "text2": "#67635B",
  "accent": "#B06A16", "mod": "#3A71AB", "warn": "#B3492E"
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
        t.bg = juce::Colour(0xff17181c);
        t.panel = juce::Colour(0xff202227);
        t.line = juce::Colour(0xff2e3138);
        t.text = juce::Colour(0xffeae6df);
        t.text2 = juce::Colour(0xffa29e95);
        t.accent = juce::Colour(0xffe2a54e);
        t.mod = juce::Colour(0xff7ba9d8);
        t.warn = juce::Colour(0xffde8068);
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
