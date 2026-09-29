# Owner: "Build the inference host" thread. Model loading and real-time-safe inference.
# anira and ONNX Runtime are added here with CPMAddPackage when the owner wires them in.
add_library(alfa_inference STATIC
    src/Inference.cpp)
add_library(Alfa::inference ALIAS alfa_inference)

target_include_directories(alfa_inference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(alfa_inference PUBLIC cxx_std_20)
alfa_set_compile_options(alfa_inference)

if(ALFA_BUILD_TESTS)
    add_subdirectory(tests)
endif()
