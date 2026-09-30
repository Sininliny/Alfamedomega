#include "Npy.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace alfa::inference
{

bool readNpyFloat2d(const std::filesystem::path& path, std::vector<float>& data, std::size_t& rows,
                    std::size_t& columns, std::string& error)
{
    const auto fail = [&](const std::string& what)
    {
        error = path.filename().string() + ": " + what;
        return false;
    };
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return fail("cannot open");

    std::array<char, 8> magic{};
    in.read(magic.data(), magic.size());
    if (!in || std::memcmp(magic.data(), "\x93NUMPY", 6) != 0)
        return fail("not a .npy file");
    const auto major = static_cast<unsigned char>(magic[6]);
    std::size_t headerLength = 0;
    if (major == 1)
    {
        std::array<unsigned char, 2> b{};
        in.read(reinterpret_cast<char*>(b.data()), 2);
        headerLength = b[0] | (std::size_t{b[1]} << 8);
    }
    else if (major == 2 || major == 3)
    {
        std::array<unsigned char, 4> b{};
        in.read(reinterpret_cast<char*>(b.data()), 4);
        headerLength =
            b[0] | (std::size_t{b[1]} << 8) | (std::size_t{b[2]} << 16) | (std::size_t{b[3]} << 24);
    }
    else
    {
        return fail("unsupported .npy version");
    }
    std::string header(headerLength, '\0');
    in.read(header.data(), static_cast<std::streamsize>(headerLength));
    if (!in)
        return fail("truncated header");

    if (header.find("'descr': '<f4'") == std::string::npos)
        return fail("expected float32 little-endian ('<f4')");
    if (header.find("'fortran_order': False") == std::string::npos)
        return fail("expected C order");
    const auto open = header.find("'shape': (");
    const auto close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos)
        return fail("no shape");
    const auto shape = header.substr(open + 10, close - open - 10);
    const auto comma = shape.find(',');
    try
    {
        rows = std::stoull(shape.substr(0, comma));
        columns = comma == std::string::npos ? 0 : std::stoull(shape.substr(comma + 1));
    }
    catch (const std::exception&)
    {
        return fail("unreadable shape (" + shape + ")");
    }
    if (rows == 0 || columns == 0)
        return fail("expected a 2-D array, got shape (" + shape + ")");

    data.resize(rows * columns);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!in)
        return fail("truncated data");
    return true;
}

} // namespace alfa::inference
