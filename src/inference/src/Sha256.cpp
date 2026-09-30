// Minimal SHA-256 (FIPS 180-4) for checking model files against the manifest. Message thread only.

#include "alfa/inference/ModelLibrary.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>

namespace alfa::inference
{
namespace
{

constexpr std::array<std::uint32_t, 64> kK{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

constexpr std::uint32_t rotr(std::uint32_t x, int n) noexcept
{
    return (x >> n) | (x << (32 - n));
}

class Sha256
{
public:
    void update(const unsigned char* data, std::size_t size)
    {
        for (std::size_t i = 0; i < size; ++i)
        {
            block_[blockSize_++] = data[i];
            if (blockSize_ == 64)
            {
                compress();
                blockSize_ = 0;
            }
        }
        bits_ += static_cast<std::uint64_t>(size) * 8u;
    }

    std::array<unsigned char, 32> finish()
    {
        const std::uint64_t bits = bits_;
        const unsigned char pad = 0x80;
        update(&pad, 1);
        const unsigned char zero = 0;
        while (blockSize_ != 56)
            update(&zero, 1);
        std::array<unsigned char, 8> length{};
        for (std::size_t i = 0; i < 8; ++i)
            length[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        update(length.data(), length.size());

        std::array<unsigned char, 32> digest{};
        for (std::size_t i = 0; i < 8; ++i)
            for (std::size_t j = 0; j < 4; ++j)
                digest[i * 4 + j] = static_cast<unsigned char>(h_[i] >> (24 - 8 * j));
        return digest;
    }

private:
    void compress()
    {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16; ++i)
            w[i] = (std::uint32_t{block_[i * 4]} << 24) | (std::uint32_t{block_[i * 4 + 1]} << 16) |
                   (std::uint32_t{block_[i * 4 + 2]} << 8) | std::uint32_t{block_[i * 4 + 3]};
        for (std::size_t i = 16; i < 64; ++i)
        {
            const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (std::size_t i = 0; i < 64; ++i)
        {
            const auto t1 =
                h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
            const auto t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }

    std::array<std::uint32_t, 8> h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<unsigned char, 64> block_{};
    std::size_t blockSize_ = 0;
    std::uint64_t bits_ = 0;
};

} // namespace

std::string sha256OfFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return {};
    Sha256 sha;
    std::array<char, 1 << 16> buffer{};
    while (in)
    {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto got = static_cast<std::size_t>(in.gcount());
        sha.update(reinterpret_cast<const unsigned char*>(buffer.data()), got);
    }
    if (in.bad())
        return {};

    std::string hex;
    hex.reserve(64);
    constexpr char kDigits[] = "0123456789abcdef";
    for (const auto byte : sha.finish())
    {
        hex.push_back(kDigits[byte >> 4]);
        hex.push_back(kDigits[byte & 0xf]);
    }
    return hex;
}

} // namespace alfa::inference
