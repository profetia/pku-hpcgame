cmake_minimum_required(VERSION 3.20)

project(FilterProject LANGUAGES C CXX Fortran)

set(CMAKE_Fortran_FLAGS "-O3 -xHost -fp-model=fast=2")

add_executable(program main.cpp filter.F90)
set_source_files_properties(filter.F90 PROPERTIES LANGUAGE Fortran)
