file(GLOB ALFA_DSP_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
alfa_add_test(alfa_dsp_tests
    SOURCES ${ALFA_DSP_TEST_SOURCES}
    LIBRARIES Alfa::dsp)
target_include_directories(alfa_dsp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# The generated tier limits must match the tier spec (docs/tiers/allocation_by_tier.json).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME alfa_dsp_tier_limits_fresh
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_tier_limits.py --check)
    set_tests_properties(alfa_dsp_tier_limits_fresh PROPERTIES SKIP_RETURN_CODE 77)
endif()
