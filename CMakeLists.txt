cmake_minimum_required(VERSION 3.6)

project(dysco)

# Casacore has a separate CMake file in this directory
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

set(CASACORE_MAKE_REQUIRED_EXTERNALS_OPTIONAL TRUE)
find_package(Casacore REQUIRED COMPONENTS casa tables)

find_package(GSL REQUIRED)

find_package(Threads REQUIRED)

include_directories(${CASACORE_INCLUDE_DIRS})
include_directories(${GSL_INCLUDE_DIRS})

add_compile_options(-O3 -Wall -DNDEBUG)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_PACKAGES "Build Debian packages" OFF)
option(ENABLE_CUDA "Build the CUDA GPU backend" OFF)

# User may optionally set `TARGET_CPU` if `PORTABLE=OFF`
option(PORTABLE "Build portable binaries (with slightly decreased performance)"
       OFF)
include(CMake/SetTargetCPU.cmake)

set(DYSCO_SOURCES
  aftimeblockencoder.cc
  dyscostman.cc
  dyscodatacolumn.cc
  dyscoweightcolumn.cc
  stochasticencoder.cc
  threadeddyscocolumn.cc
  rftimeblockencoder.cc
  rowtimeblockencoder.cc
  kernels/af_reference.cc)

set(DYSCO_GPU_SOURCES)
if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED YES)
  add_compile_definitions(DYSCO_HAS_CUDA)
  list(APPEND DYSCO_SOURCES gpu_af_time_block_encoder.cc)
  # GPU variants of the relevant unit tests (only exist in CUDA builds).
  set(DYSCO_TEST_GPU_SOURCES tests/testgpu.cc)
  list(
    APPEND
    DYSCO_GPU_SOURCES
    kernels/cuda/k1_gather_convert.cu
    kernels/cuda/k2_channel_rms.cu
    kernels/cuda/k3_antenna_rms_matrix.cu
    kernels/cuda/k4_antenna_gain_solve.cu
    kernels/cuda/k5_apply_norm.cu
    kernels/cuda/k6_fit_to_max.cu
    kernels/cuda/k7_quantize_dither.cu
    kernels/cuda/k8_bit_pack.cu
    kernels/cuda/d1_bit_unpack.cu
    kernels/cuda/d2_dequantize.cu
    kernels/cuda/gpu_af_codec.cu)
endif()

add_library(dyscostman-object OBJECT ${DYSCO_SOURCES} ${DYSCO_GPU_SOURCES})
set_property(TARGET dyscostman-object PROPERTY POSITION_INDEPENDENT_CODE 1)

# Note: casapy fails if Casa is linked in the storage manager, so we have to
# trust that casapy's version of casacore is binary compatible with this storage
# manager's casacore.
add_library(dyscostman SHARED $<TARGET_OBJECTS:dyscostman-object>)
set_target_properties(dyscostman PROPERTIES SOVERSION 0)
target_link_libraries(dyscostman ${GSL_LIBRARIES} ${CASACORE_LIBRARIES}
                      ${CMAKE_THREAD_LIBS_INIT})
if(ENABLE_CUDA)
  target_link_libraries(dyscostman CUDA::cudart)
endif()

add_executable(dscompress dscompress.cc stopwatch.cc)
target_link_libraries(dscompress dyscostman ${GSL_LIBRARIES}
                      ${CASACORE_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(decompress decompress.cc)
target_link_libraries(decompress dyscostman ${GSL_LIBRARIES}
                      ${CASACORE_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if(ENABLE_CUDA)
  add_executable(dsbench kernels/cuda/bench_af_codec.cu ${DYSCO_GPU_SOURCES}
                          kernels/af_reference.cc)
  target_include_directories(dsbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(dsbench CUDA::cudart)

  # Bit-compatibility test: CPU reference (Philox) vs GPU kernels.
  add_executable(dysco_gpu_test kernels/cuda/test_af_codec.cu
                                ${DYSCO_GPU_SOURCES} kernels/af_reference.cc)
  target_include_directories(dysco_gpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(dysco_gpu_test CUDA::cudart)
endif()

# add target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(
    doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
endif(DOXYGEN_FOUND)

find_package(Boost COMPONENTS filesystem unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIR})
  add_executable(
    runtests EXCLUDE_FROM_ALL
    $<TARGET_OBJECTS:dyscostman-object>
    tests/runtests.cc
    tests/encodeexample.cc
    tests/testbytepacking.cc
    tests/testdictionary.cc
    tests/testdithering.cc
    tests/testdyscostman.cc
    tests/testtimeblockencoder.cc
    ${DYSCO_TEST_GPU_SOURCES})
  target_link_libraries(
    runtests ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    ${GSL_LIBRARIES} ${CASACORE_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  if(ENABLE_CUDA)
    target_link_libraries(runtests CUDA::cudart)
  endif()
  add_test(runtests runtests)
  add_custom_target(
    check
    COMMAND runtests -l unit_scope
    DEPENDS runtests)
else()
  message("Boost testing framework not found.")
endif()

install(TARGETS dyscostman DESTINATION lib)
install(TARGETS dscompress DESTINATION bin)

# Add CPack directory if user wants to generate Debian packages
if(BUILD_PACKAGES)
  add_subdirectory(CPack)
endif()
