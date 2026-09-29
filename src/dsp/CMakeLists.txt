# Owner: "Build the DSP engines" thread. Real-time DSP primitives driven by the network's
# control frames (include/alfa/dsp/ControlFrame.h). No JUCE dependency.

include(CPM)

# pffft: SIMD real FFT (BSD-style licence). Only the float sources are compiled.
CPMAddPackage(
    NAME pffft
    GIT_REPOSITORY https://github.com/marton78/pffft.git
    GIT_TAG v1.1.0
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY YES)

add_library(alfa_pffft STATIC
    ${pffft_SOURCE_DIR}/src/pffft.c
    ${pffft_SOURCE_DIR}/src/pffft_common.c)
target_include_directories(alfa_pffft SYSTEM PUBLIC ${pffft_SOURCE_DIR}/include/pffft)
target_include_directories(alfa_pffft PRIVATE ${pffft_SOURCE_DIR}/src ${pffft_SOURCE_DIR}/include/pffft)
target_compile_definitions(alfa_pffft PUBLIC PFFFT_STATIC_DEFINE PRIVATE _USE_MATH_DEFINES)
if(NOT MSVC)
    target_link_libraries(alfa_pffft PRIVATE m)
endif()

# Every .cpp in src/ belongs to the library; each primitive lives in its own file.
file(GLOB ALFA_DSP_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(alfa_dsp STATIC ${ALFA_DSP_SOURCES})
add_library(Alfa::dsp ALIAS alfa_dsp)

target_include_directories(alfa_dsp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(alfa_dsp PUBLIC cxx_std_20)
target_link_libraries(alfa_dsp PRIVATE alfa_pffft)
alfa_set_compile_options(alfa_dsp)
# Lets the compiler vectorise the sinusoid banks' reductions without -ffast-math.
if(NOT MSVC)
    target_compile_options(alfa_dsp PRIVATE -fopenmp-simd)
endif()

# Writes layout.json (the frame layout for the training and inference sides).
add_executable(alfa_dsp_layout tools/DumpLayout.cpp)
target_link_libraries(alfa_dsp_layout PRIVATE Alfa::dsp)
alfa_set_compile_options(alfa_dsp_layout)

# CPU benchmark per tier: cmake --build <dir> --target alfa_dsp_bench && <dir>/.../alfa_dsp_bench
add_executable(alfa_dsp_bench EXCLUDE_FROM_ALL bench/Bench.cpp)
target_link_libraries(alfa_dsp_bench PRIVATE Alfa::dsp)
alfa_set_compile_options(alfa_dsp_bench)

if(ALFA_BUILD_TESTS)
    add_subdirectory(tests)
endif()
