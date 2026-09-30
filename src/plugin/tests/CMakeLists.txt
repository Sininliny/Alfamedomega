# Tests compile the same processor, editor and UI sources as the plugin formats,
# as plain executables. They use JUCE's recommended warnings rather than
# alfa_add_test's, because JUCE's own module sources compile inside these targets.
set(ALFA_JUCE_APP_DEFINITIONS
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
    JUCE_STANDALONE_APPLICATION=1
    JUCE_MODAL_LOOPS_PERMITTED=1 # lets tests pump the message loop for async UI updates
    JucePlugin_Name="Alfamedomega")

add_executable(alfa_plugin_tests PluginTests.cpp)
target_link_libraries(alfa_plugin_tests PRIVATE
    Catch2::Catch2WithMain
    Alfa::pluginShell
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
target_compile_definitions(alfa_plugin_tests PRIVATE ${ALFA_JUCE_APP_DEFINITIONS})
# The inference tests' dummy models, when Python could export them.
if(TARGET alfa_inference_test_models)
    add_dependencies(alfa_plugin_tests alfa_inference_test_models)
    target_compile_definitions(alfa_plugin_tests PRIVATE
        ALFA_TEST_MODELS_DIR="${CMAKE_BINARY_DIR}/src/inference/tests/models")
endif()
catch_discover_tests(alfa_plugin_tests
    TEST_PREFIX "alfa_plugin_tests:"
    DISCOVERY_MODE PRE_TEST)

# Renders every view to PNG for visual review: alfa_ui_snapshot <output-dir> [theme].
# On Linux it needs a display (run it under xvfb-run).
add_executable(alfa_ui_snapshot UiSnapshot.cpp)
target_link_libraries(alfa_ui_snapshot PRIVATE
    Alfa::pluginShell
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
target_compile_definitions(alfa_ui_snapshot PRIVATE ${ALFA_JUCE_APP_DEFINITIONS})
