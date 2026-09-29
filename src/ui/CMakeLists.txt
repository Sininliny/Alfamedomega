# Owner: "Build the plugin shell and UI" thread. Look-and-feel, theming, views and visualisations.
# An INTERFACE library: its sources compile inside the plugin target, so JUCE modules
# are built exactly once (JUCE modules must not be linked into two static libraries).
add_library(alfa_ui INTERFACE)
add_library(Alfa::ui ALIAS alfa_ui)

target_sources(alfa_ui INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/AlfaLookAndFeel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Controls.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CustomizeView.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LiveData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MainComponent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PerformView.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SignalFlowView.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SoundDesignView.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Theme.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/UiState.cpp)
target_include_directories(alfa_ui INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Alfa::pluginCore (parameter layout, telemetry) is defined in src/plugin.
target_link_libraries(alfa_ui INTERFACE
    Alfa::pluginCore
    juce::juce_gui_basics
    juce::juce_audio_utils
    juce::juce_dsp)
