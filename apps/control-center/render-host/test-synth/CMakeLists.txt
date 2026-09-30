# A tiny VST3 instrument with three programs, so the render host and the pitch-range finder
# can be tested end to end without any third-party plugin installed.
juce_add_plugin(alfa_test_synth
    PRODUCT_NAME "AlfaTestSynth"
    COMPANY_NAME "Alfamedomega"
    PLUGIN_MANUFACTURER_CODE Alft
    PLUGIN_CODE Atst
    FORMATS VST3
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    COPY_PLUGIN_AFTER_BUILD FALSE)
target_sources(alfa_test_synth PRIVATE TestSynth.cpp)
target_compile_definitions(alfa_test_synth PUBLIC
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0 JUCE_DISPLAY_SPLASH_SCREEN=0)
target_link_libraries(alfa_test_synth PRIVATE
    juce::juce_audio_utils
    juce::juce_recommended_config_flags)
