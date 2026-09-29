# alfa_set_compile_options(<target>)
# Applies the project warning set to a target. Every first-party target calls this.
function(alfa_set_compile_options target)
    get_target_property(type ${target} TYPE)
    if(type STREQUAL "INTERFACE_LIBRARY")
        set(scope INTERFACE)
    else()
        set(scope PRIVATE)
    endif()

    if(MSVC)
        # C4324: "structure was padded due to alignment specifier" is expected wherever
        # alignas() is used for SIMD or cache-line alignment.
        target_compile_options(${target} ${scope} /W4 /permissive- /utf-8 /Zc:__cplusplus /wd4324)
        if(ALFA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} ${scope} /WX)
        endif()
    else()
        target_compile_options(${target} ${scope}
            -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
            -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual)
        if(ALFA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} ${scope} -Werror)
        endif()
        if(ALFA_ENABLE_SANITIZERS)
            target_compile_options(${target} ${scope} -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} ${scope} -fsanitize=address,undefined)
        endif()
    endif()
endfunction()
