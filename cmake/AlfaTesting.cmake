# alfa_add_test(<name> SOURCES <files...> LIBRARIES <targets...>)
# Builds a Catch2 test executable and registers each TEST_CASE with CTest.
# Tag real-time or slow tests with [.slow] so they are skipped by default.
include(Catch)

function(alfa_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBRARIES" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain ${ARG_LIBRARIES})
    alfa_set_compile_options(${name})
    catch_discover_tests(${name}
        TEST_PREFIX "${name}:"
        DISCOVERY_MODE PRE_TEST)
endfunction()
