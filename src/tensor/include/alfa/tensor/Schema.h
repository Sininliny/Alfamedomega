#pragma once
// rename-compat: whole file

#include <string_view>

namespace alfa::tensor
{
// Schema and ontology ids were spelled "aom.*" before the project abbreviation became AMO.
// Programs, presets, models and host sessions saved then must keep loading, so a legacy
// "aom." prefix matches the same "amo." id. Writers always emit "amo.".
inline bool schemaIs(std::string_view value, std::string_view expected) noexcept
{
    if (value == expected)
        return true;
    constexpr std::string_view legacy = "aom."; // rename-compat
    constexpr std::string_view current = "amo.";
    return value.substr(0, 4) == legacy && expected.substr(0, 4) == current &&
           value.substr(4) == expected.substr(4);
}

// Overload for JSON fields: false for anything that is not a string.
template <typename Json>
    requires requires(const Json& j) { j.is_string(); }
bool schemaIs(const Json& field, std::string_view expected)
{
    return field.is_string() &&
           schemaIs(std::string_view(field.template get_ref<const typename Json::string_t&>()), expected);
}
} // namespace alfa::tensor
