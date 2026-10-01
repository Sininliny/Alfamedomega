file(GLOB ALFA_TENSOR_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(alfa_tensor STATIC ${ALFA_TENSOR_SOURCES})
add_library(Alfa::tensor ALIAS alfa_tensor)
target_include_directories(alfa_tensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(alfa_tensor PRIVATE ${PROJECT_SOURCE_DIR}/src/dsp/include)
target_compile_features(alfa_tensor PUBLIC cxx_std_20)
target_link_libraries(alfa_tensor PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(alfa_tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)
alfa_set_compile_options(alfa_tensor)

add_library(alfa_tensor_c SHARED tools/TensorC.cpp)
target_link_libraries(alfa_tensor_c PRIVATE Alfa::tensor nlohmann_json::nlohmann_json)
alfa_set_compile_options(alfa_tensor_c)

if(ALFA_BUILD_TESTS)
    file(GLOB ALFA_TENSOR_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
    alfa_add_test(alfa_tensor_tests
        SOURCES ${ALFA_TENSOR_TESTS}
        LIBRARIES Alfa::tensor nlohmann_json::nlohmann_json)
    target_compile_definitions(alfa_tensor_tests PRIVATE ALFA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    target_include_directories(alfa_tensor_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/dsp/include)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(ALFA_AOM_TRAINING_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH
            "Python with NumPy and PyTorch for the AOM learner tests")
        add_test(NAME alfa_aom_training_contracts
            COMMAND ${ALFA_AOM_TRAINING_PYTHON} -m unittest
                    training.tests.test_aom_contracts training.tests.test_aom_hardware -v)
        set_tests_properties(alfa_aom_training_contracts PROPERTIES
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
        execute_process(COMMAND ${ALFA_AOM_TRAINING_PYTHON} -c "import numpy, torch"
            RESULT_VARIABLE ALFA_AOM_TORCH_AVAILABLE OUTPUT_QUIET ERROR_QUIET)
        if(ALFA_AOM_TORCH_AVAILABLE EQUAL 0)
            add_test(NAME alfa_aom_training_learners
                COMMAND ${ALFA_AOM_TRAINING_PYTHON} -m unittest
                    training.tests.test_aom_learner training.tests.test_aom_local_learning
                    training.tests.test_aom_runtime_model training.tests.test_aom_tiers -v)
            set_tests_properties(alfa_aom_training_learners PROPERTIES
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
        else()
            message(STATUS "AOM learner tests require NumPy/PyTorch; set ALFA_AOM_TRAINING_PYTHON to that environment")
        endif()
    endif()
endif()

# Explicit profiling tool; timings are machine-specific and never a test threshold.
add_executable(alfa_tensor_precision_bench bench/PrecisionBench.cpp)
target_link_libraries(alfa_tensor_precision_bench PRIVATE Alfa::tensor)
alfa_set_compile_options(alfa_tensor_precision_bench)

add_executable(alfa_tensor_sparse_bench bench/SparseRuntimeBench.cpp)
target_link_libraries(alfa_tensor_sparse_bench PRIVATE Alfa::tensor)
alfa_set_compile_options(alfa_tensor_sparse_bench)
