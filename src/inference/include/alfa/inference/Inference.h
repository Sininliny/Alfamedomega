#pragma once

#include <string_view>

namespace alfa::inference
{

/// Placeholder so the target links. The inference owner replaces this file.
[[nodiscard]] std::string_view componentName() noexcept;

} // namespace alfa::inference
