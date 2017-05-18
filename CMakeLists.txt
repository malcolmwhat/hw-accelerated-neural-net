cmake_minimum_required(VERSION 2.8)
project(hardware_accelerator)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

set(SOURCE_FILES library.c library.h feed_forward_translation.c
        feed_forward_translation.h hardware_emulator.c hardware_emulator.h layer.h utilities.h)
add_library(hardware_accelerator SHARED ${SOURCE_FILES})
add_executable(runTests test.c ${SOURCE_FILES} test.h)