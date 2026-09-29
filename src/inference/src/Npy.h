#pragma once

// Reads a 2-D little-endian float32 C-order .npy file (numpy's format, versions 1 to 3).
// Message thread only.

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace alfa::inference
{

/// Returns false and sets error if the file is not a 2-D '<f4' C-order array.
bool readNpyFloat2d(const std::filesystem::path& path, std::vector<float>& data, std::size_t& rows,
                    std::size_t& columns, std::string& error);

} // namespace alfa::inference
