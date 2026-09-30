file(GLOB ALFA_TENSOR_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(alfa_tensor STATIC ${ALFA_TENSOR_SOURCES})
add_library(Alfa::tensor ALIAS alfa_tensor)
target_include_directories(alfa_tensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(alfa_tensor PUBLIC cxx_std_20)
alfa_set_compile_options(alfa_tensor)

if(ALFA_BUILD_TESTS)
    file(GLOB ALFA_TENSOR_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
    alfa_add_test(alfa_tensor_tests
        SOURCES ${ALFA_TENSOR_TESTS}
        LIBRARIES Alfa::tensor)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME alfa_aom_training_contracts
            COMMAND ${Python3_EXECUTABLE} -m unittest discover
                    -s training/tests -p "test_aom*.py" -v)
        set_tests_properties(alfa_aom_training_contracts PROPERTIES
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    endif()
endif()
