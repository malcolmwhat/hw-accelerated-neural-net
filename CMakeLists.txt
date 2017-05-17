cmake_minimum_required(VERSION 3.5)
project(hardware_accelerator)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES library.cpp library.h feed_forward_translation.c
        feed_forward_translation.h hardware_emulator.c hardware_emulator.h layer.h utilities.h)
add_library(hardware_accelerator SHARED ${SOURCE_FILES})