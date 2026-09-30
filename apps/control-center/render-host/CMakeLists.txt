# alfa-render-host: a headless plugin host for Local Plugin Assimilation
# (docs/control-center/plugin-assimilation.md). The control center runs one process per plugin,
# so a crashing third-party plugin takes down only its own render, never the app.
#
# Builds on its own:
#   cmake -S apps/control-center/render-host -B /tmp/alfa-render-host -G Ninja
#   cmake --build /tmp/alfa-render-host
cmake_minimum_required(VERSION 3.25)
project(AlfaRenderHost VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()

option(ALFA_RENDER_HOST_TEST_SYNTH "Also build the small VST3 instrument the tests render" ON)
# VST2 hosting needs Steinberg's VST2 SDK headers, which are no longer licensed to new
# developers and cannot be redistributed. Point this at a licensed copy to enable it.
set(ALFA_VST2_SDK_PATH "" CACHE PATH "Path to a licensed VST2 SDK; VST2 hosting is built only when set")

if(NOT COMMAND juce_add_console_app)
    set(_repo_cmake "${CMAKE_CURRENT_LIST_DIR}/../../../cmake")
    list(APPEND CMAKE_MODULE_PATH "${_repo_cmake}")
    include(CPM)
    # Same pin as cmake/Dependencies.cmake. JUCE is used under AGPLv3 until the licence decision.
    CPMAddPackage(NAME JUCE GITHUB_REPOSITORY juce-framework/JUCE GIT_TAG 8.0.9 GIT_SHALLOW TRUE SYSTEM TRUE)
endif()

juce_add_console_app(alfa_render_host PRODUCT_NAME "alfa-render-host")
target_sources(alfa_render_host PRIVATE src/Main.cpp)
# Console apps ignore PRODUCT_NAME, so name the binary here.
set_target_properties(alfa_render_host PROPERTIES OUTPUT_NAME "alfa-render-host")
target_compile_definitions(alfa_render_host PRIVATE
    JUCE_PLUGINHOST_VST3=1
    JUCE_PLUGINHOST_AU=$<BOOL:${APPLE}>
    JUCE_PLUGINHOST_LADSPA=0
    JUCE_PLUGINHOST_LV2=0
    JUCE_PLUGINHOST_VST=$<BOOL:${ALFA_VST2_SDK_PATH}>
    JUCE_USE_CURL=0
    JUCE_WEB_BROWSER=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0)
if(ALFA_VST2_SDK_PATH)
    juce_set_vst2_sdk_path("${ALFA_VST2_SDK_PATH}")
endif()
target_link_libraries(alfa_render_host PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_formats
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)

if(ALFA_RENDER_HOST_TEST_SYNTH)
    add_subdirectory(test-synth)
endif()
