# Dummy models for the tests are generated at build time by tools/export_dummy_models.py, which
# needs Python 3 with numpy and onnx. Without them the model-dependent tests report SKIPPED.
find_package(Python3 COMPONENTS Interpreter)
set(ALFA_TEST_MODELS_DIR "${CMAKE_CURRENT_BINARY_DIR}/models")
set(ALFA_HAVE_TEST_MODELS 0)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy, onnx"
        RESULT_VARIABLE onnx_import_result
        OUTPUT_QUIET ERROR_QUIET)
    if(onnx_import_result EQUAL 0)
        set(ALFA_HAVE_TEST_MODELS 1)
    endif()
endif()
if(ALFA_HAVE_TEST_MODELS)
    set(exporter ${CMAKE_CURRENT_SOURCE_DIR}/../tools/export_dummy_models.py)
    add_custom_command(
        OUTPUT ${ALFA_TEST_MODELS_DIR}/manifest.json
        COMMAND ${Python3_EXECUTABLE} ${exporter} ${ALFA_TEST_MODELS_DIR}
        DEPENDS ${exporter}
        COMMENT "Exporting dummy ONNX models for the inference tests")
    add_custom_target(alfa_inference_test_models DEPENDS ${ALFA_TEST_MODELS_DIR}/manifest.json)
else()
    message(WARNING "Python 3 with numpy and onnx not found: inference tests that need models will be skipped "
                    "(pip install numpy onnx)")
endif()

function(alfa_inference_test name)
    alfa_add_test(${name} ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    # FramePipeline.h is private and includes ONNX Runtime; the pipeline tests use it directly.
    get_target_property(ort_includes anira::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${name} SYSTEM PRIVATE ${ort_includes})
    target_link_libraries(${name} PRIVATE anira::onnxruntime)
    target_compile_definitions(${name} PRIVATE
        ALFA_TEST_MODELS_DIR="${ALFA_TEST_MODELS_DIR}"
        ALFA_HAVE_TEST_MODELS=${ALFA_HAVE_TEST_MODELS})
    if(ALFA_HAVE_TEST_MODELS)
        add_dependencies(${name} alfa_inference_test_models)
    endif()
endfunction()

alfa_inference_test(alfa_inference_tests
    SOURCES
        FrameClockTests.cpp
        ModelLibraryTests.cpp
        PipelineTests.cpp
        HostTests.cpp
    LIBRARIES Alfa::inference)

# Counts heap allocations on the audio thread by replacing malloc and operator new for the whole
# test process. Works with any compiler; kept in its own executable so it cannot disturb others.
# Not under ASan, which owns malloc itself.
if(NOT ALFA_ENABLE_SANITIZERS)
    alfa_inference_test(alfa_inference_alloc_tests
        SOURCES AllocationGuard.cpp RealtimeSafetyTests.cpp
        LIBRARIES Alfa::inference)
endif()

# RealtimeSanitizer (Clang 20+) additionally catches locks, blocking calls and system calls.
# Configure with CXX=clang++-20 to build it; CI can add that as its own job.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS -fsanitize=realtime)
set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=realtime)
check_cxx_source_compiles("int main() { return 0; }" ALFA_HAVE_RTSAN)
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_LINK_OPTIONS)
if(ALFA_HAVE_RTSAN)
    alfa_inference_test(alfa_inference_rtsan_tests
        SOURCES RtsanTests.cpp
        LIBRARIES Alfa::inference)
    target_compile_options(alfa_inference_rtsan_tests PRIVATE -fsanitize=realtime)
    target_link_options(alfa_inference_rtsan_tests PRIVATE -fsanitize=realtime)
endif()
