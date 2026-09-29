file(GLOB ALFA_DSP_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
alfa_add_test(alfa_dsp_tests
    SOURCES ${ALFA_DSP_TEST_SOURCES}
    LIBRARIES Alfa::dsp)
target_include_directories(alfa_dsp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
