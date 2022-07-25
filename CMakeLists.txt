
cmake_minimum_required(VERSION 3.19 FATAL_ERROR) #Specify the minimum cmake version to use
project(vectorAdd LANGUAGES CXX CUDA) #Project name and languages in use

find_package(CUDAToolkit) #find cuda library and compiler

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/) # Specify the output directory (Usually just bin or bin/Debug etc)

add_library(MML1 MM1.cu) # cpp
add_library(MML2 MM2.cu) # res major
add_library(MML3 MM3.cu) # res major + local output
add_library(MML4 MM4.cu) # res major + two axis + local output
add_library(MML5 MM5.cu) # res major + blocked + local output
add_library(MML6 MM6.cu) # res major + blocked + shared + local output
add_library(MML7 MM7.cu) # res major + blocked + shared + local output + coealocal 1
add_library(MML8 MM8.cu) # res major + blocked + shared + local output + coealocal 2
add_library(MML9 MM9.cu) # res major + blocked + shared + local output + coealocal 3
add_library(MML10 MM10.cu) # Strassen's
add_library(MML11 MM11.cu) # Winograd's

add_executable(MMP1 performance.cu)
target_link_libraries(MMP1 MML1)
add_executable(MMP2 performance.cu)
target_link_libraries(MMP2 MML2)
add_executable(MMP3 performance.cu)
target_link_libraries(MMP3 MML3)
add_executable(MMP4 performance.cu)
target_link_libraries(MMP4 MML4)
add_executable(MMP5 performance.cu)
target_link_libraries(MMP5 MML5)
add_executable(MMP6 performance.cu)
target_link_libraries(MMP6 MML6)
add_executable(MMP7 performance.cu)
target_link_libraries(MMP7 MML7)
add_executable(MMP8 performance.cu)
target_link_libraries(MMP8 MML8)
add_executable(MMP9 performance.cu)
target_link_libraries(MMP9 MML9)
add_executable(MMP10 performance.cu)
target_link_libraries(MMP10 MML10)
add_executable(MMP11 performance.cu)
target_link_libraries(MMP11 MML11)

add_executable(MMC1 correctness.cu)
target_link_libraries(MMC1 MML1)
add_executable(MMC2 correctness.cu)
target_link_libraries(MMC2 MML2)
add_executable(MMC3 correctness.cu)
target_link_libraries(MMC3 MML3)
add_executable(MMC4 correctness.cu)
target_link_libraries(MMC4 MML4)
add_executable(MMC5 correctness.cu)
target_link_libraries(MMC5 MML5)
add_executable(MMC6 correctness.cu)
target_link_libraries(MMC6 MML6)
add_executable(MMC7 correctness.cu)
target_link_libraries(MMC7 MML7)
add_executable(MMC8 correctness.cu)
target_link_libraries(MMC8 MML8)
add_executable(MMC9 correctness.cu)
target_link_libraries(MMC9 MML9)
add_executable(MMC10 correctness.cu)
target_link_libraries(MMC10 MML10)
add_executable(MMC11 correctness.cu)
target_link_libraries(MMC11 MML11)