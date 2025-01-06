cmake_minimum_required(VERSION 3.30)
project(NN C)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_C_STANDARD 11)


include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/libs)

add_executable(${PROJECT_NAME} src/function_approximation.c)

find_package(raylib REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE raylib)

if(APPLE)
    find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
    find_library(IOKIT_FRAMEWORK IOKit REQUIRED)
    find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
    find_library(OPENGL_FRAMEWORK OpenGL REQUIRED)

    target_link_libraries(${PROJECT_NAME}
            PRIVATE ${COCOA_FRAMEWORK}
            ${IOKIT_FRAMEWORK}
            ${COREVIDEO_FRAMEWORK}
            ${OPENGL_FRAMEWORK})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
