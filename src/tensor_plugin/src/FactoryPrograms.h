#pragma once

#include <span>
#include <string>

namespace alfa::tensorhost
{
struct FactoryProgram
{
    std::string id;
    std::string name;
    std::string category;
    std::string description;
    std::string document;
};

// Authored executable configurations of one tensor ontology. Construction, parsing,
// validation and selection belong to the message thread, never the audio callback.
std::span<const FactoryProgram> factoryPrograms();
bool validateFactoryPrograms(double sampleRate, std::string& error);
} // namespace alfa::tensorhost
