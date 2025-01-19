cmake_minimum_required(VERSION 3.20)

project(FilterProject LANGUAGES C CXX)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ox -Ofast -fPIC -fp-model=fast=2 -xhost -global-hoist -opt-prefetch=2")

add_executable(program main.cpp filter.F90)
set_source_files_properties(filter.F90 PROPERTIES LANGUAGE C)
