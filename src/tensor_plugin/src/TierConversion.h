#pragma once
#include <string>
#include <string_view>
namespace alfa::tensorhost
{
// Off-thread candidate search. Product policies and validation fixtures are authored,
// never inferred from old method quotas. Failed conversion leaves the host untouched.
bool convertTier(std::string_view sound, std::string_view tier, double rate, std::string& converted,
                 std::string& error);
} // namespace alfa::tensorhost
