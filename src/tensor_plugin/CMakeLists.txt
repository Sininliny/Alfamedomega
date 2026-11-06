juce_add_binary_data(alfa_tensor_reference HEADER_NAME TensorReference.h NAMESPACE TensorReference SOURCES
    ${PROJECT_SOURCE_DIR}/programs/instrument.amo.json
    ${PROJECT_SOURCE_DIR}/programs/fx.amo.json
    ${PROJECT_SOURCE_DIR}/programs/hybrid.amo.json)
add_library(alfa_tensor_plugin_shell INTERFACE)
target_sources(alfa_tensor_plugin_shell INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TensorProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TensorExecution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PresetStore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FactoryPrograms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TensorComposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TierConversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CompositionPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PrimitiveFragmentPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SoundLibraryPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ElementViews.cpp
    ${PROJECT_SOURCE_DIR}/src/ui/src/Theme.cpp
    ${PROJECT_SOURCE_DIR}/src/ui/src/AlfaLookAndFeel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TensorEditor.cpp)
target_include_directories(alfa_tensor_plugin_shell INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/ui/include)
target_compile_definitions(alfa_tensor_plugin_shell INTERFACE
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
target_link_libraries(alfa_tensor_plugin_shell INTERFACE
    Alfa::tensor alfa_tensor_reference juce::juce_audio_utils juce::juce_dsp juce::juce_cryptography nlohmann_json::nlohmann_json)
add_library(Alfa::tensorHost ALIAS alfa_tensor_plugin_shell)
if(ALFA_BUILD_TESTS)
    alfa_add_test(alfa_tensor_host_tests SOURCES
        tests/HostTests.cpp tests/PresetStoreTests.cpp tests/FactoryProgramTests.cpp
        tests/ProductHostTests.cpp tests/RealtimeTests.cpp tests/CompositionTests.cpp tests/CompositionPanelTests.cpp
        tests/PrimitiveFragmentTests.cpp tests/ProductSafetyTests.cpp tests/SoundLibraryPanelTests.cpp
        tests/HostCompatibilityTests.cpp tests/WavImportPresetTests.cpp
        LIBRARIES alfa_tensor_plugin_shell)
    target_compile_definitions(alfa_tensor_host_tests PRIVATE ALFA_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1 JUCE_STANDALONE_APPLICATION=1 JUCE_MODAL_LOOPS_PERMITTED=1 JUCE_PLUGINHOST_VST3=1)
    target_link_libraries(alfa_tensor_host_tests PRIVATE juce::juce_recommended_config_flags)
endif()
