# Pinned third-party dependencies, fetched with CPM.
# Set CPM_SOURCE_CACHE (env or cache var) to share downloads between build trees:
#   export CPM_SOURCE_CACHE=$HOME/.cache/CPM
# Component-specific dependencies (for example anira and ONNX Runtime) are added
# by the owning component's CMakeLists.txt, also through CPM, and pinned there.

include(CPM)

if(ALFA_BUILD_TESTS)
    CPMAddPackage(
        NAME Catch2
        GITHUB_REPOSITORY catchorg/Catch2
        VERSION 3.9.1
        GIT_SHALLOW TRUE
        SYSTEM TRUE)
    list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
endif()

if(ALFA_BUILD_PLUGIN)
    # JUCE is used under AGPLv3 until the licence decision is made (see README).
    CPMAddPackage(
        NAME JUCE
        GITHUB_REPOSITORY juce-framework/JUCE
        GIT_TAG 8.0.9
        GIT_SHALLOW TRUE
        SYSTEM TRUE)
endif()
