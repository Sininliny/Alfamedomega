# Owner: "Build the inference host" thread. Model loading and real-time-safe inference.
#
# anira (Apache-2.0) runs inference on its own static thread pool and turns the declared
# maximum inference time into a fixed latency. It downloads a prebuilt ONNX Runtime, which is
# linked statically so no onnxruntime shared library can clash with another plugin's copy
# (docs/packaging/packaging-plan.md section 2.1). Only the ONNX Runtime backend is enabled.

set(ALFA_ANIRA_VERSION 2.3.0)

CPMAddPackage(
    NAME anira
    GITHUB_REPOSITORY anira-project/anira
    VERSION ${ALFA_ANIRA_VERSION}
    GIT_TAG v${ALFA_ANIRA_VERSION}
    GIT_SHALLOW TRUE
    SYSTEM TRUE
    OPTIONS
        "BUILD_SHARED_LIBS OFF"
        "ANIRA_WITH_ONNXRUNTIME ON"
        "ANIRA_WITH_LIBTORCH OFF"
        "ANIRA_WITH_LITERT OFF"
        "ANIRA_WITH_TFLITE OFF"
        "ANIRA_WITH_EXECUTORCH OFF"
        "ANIRA_WITH_BENCHMARK OFF"
        "ANIRA_WITH_EXAMPLES OFF"
        "ANIRA_WITH_TESTS OFF"
        "ANIRA_WITH_INSTALL OFF")

# anira fetches nlohmann_json for its own config loader; reuse it for manifest.json.
if(NOT TARGET nlohmann_json::nlohmann_json)
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.3
        SYSTEM TRUE)
endif()

add_library(alfa_inference STATIC
    src/FramePipeline.cpp
    src/InferenceHost.cpp
    src/ModelLibrary.cpp
    src/Sha256.cpp)
add_library(Alfa::inference ALIAS alfa_inference)

target_include_directories(alfa_inference
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# The control-frame layout is the DSP component's header-only contract
# (alfa/dsp/ControlFrame.h). Only its include path is used: no link dependency on Alfa::dsp.
target_include_directories(alfa_inference PUBLIC
    $<TARGET_PROPERTY:alfa_dsp,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_features(alfa_inference PUBLIC cxx_std_20)
target_link_libraries(alfa_inference
    PRIVATE anira::anira anira::onnxruntime nlohmann_json::nlohmann_json)
# The public headers are plain C++; anira and ONNX Runtime stay private to the library.
foreach(dep anira onnxruntime)
    get_target_property(dep_includes anira::${dep} INTERFACE_INCLUDE_DIRECTORIES)
    if(dep_includes)
        target_include_directories(alfa_inference SYSTEM PRIVATE ${dep_includes})
    endif()
endforeach()
alfa_set_compile_options(alfa_inference)

if(ALFA_BUILD_TESTS)
    add_subdirectory(tests)
endif()
