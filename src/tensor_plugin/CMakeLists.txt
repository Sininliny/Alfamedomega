juce_add_binary_data(alfa_tensor_reference HEADER_NAME TensorReference.h NAMESPACE TensorReference SOURCES
    ${PROJECT_SOURCE_DIR}/programs/instrument.aom.json
    ${PROJECT_SOURCE_DIR}/programs/fx.aom.json
    ${PROJECT_SOURCE_DIR}/programs/hybrid.aom.json)
add_library(alfa_tensor_plugin_shell INTERFACE)
target_sources(alfa_tensor_plugin_shell INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TensorProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TensorEditor.cpp)
target_include_directories(alfa_tensor_plugin_shell INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(alfa_tensor_plugin_shell INTERFACE
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
target_link_libraries(alfa_tensor_plugin_shell INTERFACE
    Alfa::tensor alfa_tensor_reference juce::juce_audio_utils nlohmann_json::nlohmann_json)
juce_add_plugin(AlfamedomegaTensor
    PRODUCT_NAME "Alfamedomega Tensor" COMPANY_NAME "Alfamedomega"
    BUNDLE_ID "com.alfamedomega.tensor" PLUGIN_MANUFACTURER_CODE Alfm PLUGIN_CODE Aftn
    FORMATS Standalone VST3 IS_SYNTH FALSE NEEDS_MIDI_INPUT TRUE NEEDS_MIDI_OUTPUT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES Fx Instrument)
target_link_libraries(AlfamedomegaTensor PRIVATE alfa_tensor_plugin_shell
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)
target_sources(AlfamedomegaTensor_Standalone PRIVATE src/StandaloneMain.cpp)
target_compile_definitions(AlfamedomegaTensor_Standalone PRIVATE JUCE_USE_CUSTOM_PLUGIN_STANDALONE_APP=1)
if(ALFA_BUILD_TESTS)
    alfa_add_test(alfa_tensor_host_tests SOURCES tests/HostTests.cpp
        LIBRARIES alfa_tensor_plugin_shell)
    target_compile_definitions(alfa_tensor_host_tests PRIVATE ALFA_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1 JUCE_STANDALONE_APPLICATION=1 JUCE_MODAL_LOOPS_PERMITTED=1)
    target_link_libraries(alfa_tensor_host_tests PRIVATE juce::juce_recommended_config_flags)
endif()
