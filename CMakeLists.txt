cmake_minimum_required(VERSION 3.30)
project(NN C)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_STANDARD 11)

include_directories(/opt/homebrew/include)

add_executable(NN number_recognizer/UI.c)

find_library(RAYLIB_LIBRARY raylib PATHS /opt/homebrew/lib REQUIRED)

# Link libraries
target_link_libraries(NN
        ${RAYLIB_LIBRARY}
        ${COCOA_FRAMEWORK}
        ${IOKIT_FRAMEWORK}
        ${COREVIDEO_FRAMEWORK}
        ${OPENGL_FRAMEWORK}
)
