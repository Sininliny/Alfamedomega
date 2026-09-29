# Owner: "Build the plugin shell and UI" thread. The plugin target for every format.
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
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES Instrument Synth
    AU_MAIN_TYPE kAudioUnitType_MusicDevice)

target_sources(Alfamedomega PRIVATE
    src/PluginProcessor.cpp
    src/PluginEditor.cpp)

target_compile_definitions(Alfamedomega PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(Alfamedomega
    PRIVATE
        Alfa::dsp
        Alfa::inference
        Alfa::ui
        juce::juce_audio_utils
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
