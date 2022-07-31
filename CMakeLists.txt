
cmake_minimum_required(VERSION 3.19 FATAL_ERROR) #Specify the minimum cmake version to use
project(vectorAdd LANGUAGES CXX CUDA) #Project name and languages in use

find_package(CUDAToolkit) #find cuda library and compiler

include_directories(include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/) # Specify the output directory (Usually just bin or bin/Debug etc)

add_library(MML1  src/dmm/MM1.cu) # cpp
add_library(MML2  src/dmm/MM2.cu) # res major
add_library(MML3  src/dmm/MM3.cu) # res major + local output
add_library(MML4  src/dmm/MM4.cu) # res major + two axis + local output
add_library(MML5  src/dmm/MM5.cu) # res major + blocked + local output
add_library(MML6  src/dmm/MM6.cu) # res major + blocked + shared + local output
add_library(MML7  src/dmm/MM7.cu) # res major + blocked + shared + local output + coealocal 1
add_library(MML8  src/dmm/MM8.cu) # res major + blocked + shared + local output + coealocal 2
add_library(MML9  src/dmm/MM9.cu) # res major + blocked + shared + local output + coealocal 3
add_library(MML10 src/dmm/MM10.cu) # Strassen's
add_library(MML11 src/dmm/MM11.cu) # Winograd's

add_executable(MMP1  src/tests/performance.cu)
target_link_libraries(MMP1 MML1)
add_executable(MMP2  src/tests/performance.cu)
target_link_libraries(MMP2 MML2)
add_executable(MMP3  src/tests/performance.cu)
target_link_libraries(MMP3 MML3)
add_executable(MMP4  src/tests/performance.cu)
target_link_libraries(MMP4 MML4)
add_executable(MMP5  src/tests/performance.cu)
target_link_libraries(MMP5 MML5)
add_executable(MMP6  src/tests/performance.cu)
target_link_libraries(MMP6 MML6)
add_executable(MMP7  src/tests/performance.cu)
target_link_libraries(MMP7 MML7)
add_executable(MMP8  src/tests/performance.cu)
target_link_libraries(MMP8 MML8)
add_executable(MMP9  src/tests/performance.cu)
target_link_libraries(MMP9 MML9)
add_executable(MMP10 src/tests/performance.cu)
target_link_libraries(MMP10 MML10)
add_executable(MMP11 src/tests/performance.cu)
target_link_libraries(MMP11 MML11)

add_executable(MMC1  src/tests/correctness.cu)
target_link_libraries(MMC1 MML1)
add_executable(MMC2  src/tests/correctness.cu)
target_link_libraries(MMC2 MML2)
add_executable(MMC3  src/tests/correctness.cu)
target_link_libraries(MMC3 MML3)
add_executable(MMC4  src/tests/correctness.cu)
target_link_libraries(MMC4 MML4)
add_executable(MMC5  src/tests/correctness.cu)
target_link_libraries(MMC5 MML5)
add_executable(MMC6  src/tests/correctness.cu)
target_link_libraries(MMC6 MML6)
add_executable(MMC7  src/tests/correctness.cu)
target_link_libraries(MMC7 MML7)
add_executable(MMC8  src/tests/correctness.cu)
target_link_libraries(MMC8 MML8)
add_executable(MMC9  src/tests/correctness.cu)
target_link_libraries(MMC9 MML9)
add_executable(MMC10 src/tests/correctness.cu)
target_link_libraries(MMC10 MML10)
add_executable(MMC11 src/tests/correctness.cu)
target_link_libraries(MMC11 MML11)

add_executable(MMCGPU1  src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU1 MML1)
add_executable(MMCGPU2  src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU2 MML2)
add_executable(MMCGPU3  src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU3 MML3)
add_executable(MMCGPU4  src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU4 MML4)
add_executable(MMCGPU5  src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU5 MML5)
add_executable(MMCGPU6  src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU6 MML6)
add_executable(MMCGPU7  src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU7 MML7)
add_executable(MMCGPU8  src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU8 MML8)
add_executable(MMCGPU9  src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU9 MML9)
add_executable(MMCGPU10 src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU10 MML10)
add_executable(MMCGPU11 src/tests/correctnessGPU.cu)
target_link_libraries(MMCGPU11 MML11)

add_executable(get src/tests/get.cu)
target_link_libraries(get MML10)