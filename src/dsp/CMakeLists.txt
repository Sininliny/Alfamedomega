# Owner: "Build the DSP engines" thread. Real-time synthesis engines, no JUCE dependency.
add_library(alfa_dsp STATIC
    src/Dsp.cpp)
add_library(Alfa::dsp ALIAS alfa_dsp)

target_include_directories(alfa_dsp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(alfa_dsp PUBLIC cxx_std_20)
alfa_set_compile_options(alfa_dsp)

if(ALFA_BUILD_TESTS)
    add_subdirectory(tests)
endif()
