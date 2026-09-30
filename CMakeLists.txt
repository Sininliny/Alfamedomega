cmake_minimum_required(VERSION 3.25)

project(Alfamedomega
    VERSION 0.1.0
    DESCRIPTION "Universal audio tensor compiler and stateful DSP runtime"
    LANGUAGES C CXX)

# Options -----------------------------------------------------------------------
option(ALFA_BUILD_PLUGIN "Build the JUCE plugin (downloads JUCE)" ON)
option(ALFA_BUILD_TESTS "Build unit tests (downloads Catch2)" ${PROJECT_IS_TOP_LEVEL})
option(ALFA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(ALFA_ENABLE_SANITIZERS "Build with AddressSanitizer and UBSan (Clang/GCC only)" OFF)
set(ALFA_AAX_SDK_PATH "" CACHE PATH "Path to the AAX SDK; AAX is built only when set")

# Toolchain-wide settings -------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Hidden symbols by default: a plugin shares its process with every other plugin.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# DLL MSVC runtime (/MD). ONNX Runtime's prebuilt static library is compiled with /MD, and
# mixing CRTs in one binary fails to link. The installer ships the VC++ redistributable.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
if(MSVC)
    # A DAW may already have loaded an older msvcp140.dll; std::mutex built with VS 17.10+
    # then crashes in it. Keep the pre-17.10 non-constexpr constructor (packaging plan).
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(AlfaCompilerOptions)
include(Dependencies)

if(ALFA_BUILD_TESTS)
    enable_testing()
    include(AlfaTesting)
endif()

# Components ----------------------------------------------------------------------
# Each directory below has one owner (see CONTRIBUTING.md). Only the owner edits it.
add_subdirectory(src/dsp)         # Alfa::dsp        synthesis engines, JUCE-free
add_subdirectory(src/inference)   # Alfa::inference  model loading and real-time inference host
add_subdirectory(src/tensor)      # Alfa::tensor     tensor compilation and recurrent state runtime

if(ALFA_BUILD_PLUGIN)
    add_subdirectory(src/tensor_plugin) # Native tensor instrument/FX/hybrid host
    add_subdirectory(src/ui)      # Alfa::ui         look-and-feel, components, theming
    add_subdirectory(src/plugin)  # Alfamedomega     VST3 / AU / AAX / Standalone
endif()
