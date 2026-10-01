#include "alfa/tensor/Precision.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
namespace alfa::tensor
{
std::string_view scalarTypeName(ScalarType type) noexcept
{
    return type == ScalarType::Float64 ? "float64" : "float32";
}
bool resolvePrecisionJSON(std::string_view text, PrecisionProfile& output, std::string& error)
{
    try
    {
        const auto document = nlohmann::json::parse(text);
        PrecisionProfile candidate;
        if (document.contains("tier"))
        {
            candidate.tier = document.at("tier").get<std::string>();
            if (candidate.tier == "Lite" || candidate.tier == "Standard")
                candidate.scalarType = ScalarType::Float32;
            else if (candidate.tier == "Pro" || candidate.tier == "Studio")
                candidate.scalarType = ScalarType::Float64;
            else
                throw std::invalid_argument("Unknown precision tier");
        }
        if (document.contains("scalar_type"))
        {
            const auto type = document.at("scalar_type").get<std::string>();
            ScalarType selected;
            if (type == "float32")
                selected = ScalarType::Float32;
            else if (type == "float64")
                selected = ScalarType::Float64;
            else
                throw std::invalid_argument("Unsupported runtime scalar_type; no quantized DSP substitution");
            if (!candidate.tier.empty() && selected != candidate.scalarType)
                throw std::invalid_argument("scalar_type contradicts the tier's uniform datatype");
            candidate.scalarType = selected;
        }
        output = std::move(candidate);
        error.clear();
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
} // namespace alfa::tensor
