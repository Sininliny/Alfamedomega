# Owner: "Build the plugin shell and UI" thread. Look-and-feel, theming and editor components.
# An INTERFACE library: its sources compile inside the plugin target, so JUCE modules
# are built exactly once (JUCE modules must not be linked into two static libraries).
add_library(alfa_ui INTERFACE)
add_library(Alfa::ui ALIAS alfa_ui)

target_sources(alfa_ui INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/AlfaLookAndFeel.cpp)
target_include_directories(alfa_ui INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(alfa_ui INTERFACE
    juce::juce_gui_basics)
