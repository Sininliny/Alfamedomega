# Owner: "Build the plugin shell and UI" thread. The plugin target for every format.

# Parameter layout, presets and telemetry: shared by the processor, the UI and tests.
# INTERFACE for the same reason as Alfa::ui (JUCE modules compile once, in the consumer).
add_library(alfa_plugin_core INTERFACE)
add_library(Alfa::pluginCore ALIAS alfa_plugin_core)
target_sources(alfa_plugin_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ParameterLayout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PresetLibrary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PreferenceLog.cpp)
target_include_directories(alfa_plugin_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(alfa_plugin_core INTERFACE
    juce::juce_audio_processors)

# Processor, editor and stub backend. The plugin target and the tests both link it,
# so tests exercise exactly the code the formats ship.
add_library(alfa_plugin_shell INTERFACE)
add_library(Alfa::pluginShell ALIAS alfa_plugin_shell)
target_sources(alfa_plugin_shell INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PluginProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PluginEditor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/StubBackend.cpp)
target_include_directories(alfa_plugin_shell INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(alfa_plugin_shell INTERFACE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)
target_link_libraries(alfa_plugin_shell INTERFACE
    Alfa::pluginCore
    Alfa::ui
    Alfa::dsp
    Alfa::inference
    juce::juce_audio_utils
    juce::juce_dsp)

set(ALFA_FORMATS VST3 Standalone)
if(APPLE)
    list(APPEND ALFA_FORMATS AU)
endif()
if(ALFA_AAX_SDK_PATH)
    juce_set_aax_sdk_path("${ALFA_AAX_SDK_PATH}")
    list(APPEND ALFA_FORMATS AAX)
endif()

# Identity codes are PROVISIONAL. They must be fixed before the first public build,
# because changing them breaks saved sessions and presets (packaging plan, section 2).
juce_add_plugin(Alfamedomega
    PRODUCT_NAME "Alfamedomega"
    COMPANY_NAME "Alfamedomega"
    BUNDLE_ID "com.alfamedomega.alfamedomega"
    PLUGIN_MANUFACTURER_CODE Alfm
    PLUGIN_CODE Afmo
    FORMATS ${ALFA_FORMATS}
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES Instrument Synth
    AAX_CATEGORY AAX_ePlugInCategory_SWGenerators
    AU_MAIN_TYPE kAudioUnitType_MusicDevice)

target_link_libraries(Alfamedomega
    PRIVATE
        Alfa::pluginShell
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

if(ALFA_BUILD_TESTS)
    add_subdirectory(tests)
endif()
