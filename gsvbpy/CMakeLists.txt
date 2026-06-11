cmake_minimum_required(VERSION 3.18)
project(gsvbpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Python + pybind11 (vendored) ---
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
add_subdirectory(third_party/pybind11)

# --- BLAS / LAPACK (OpenBLAS) ---
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# --- Vendored header-only Armadillo (no runtime wrapper; link BLAS/LAPACK directly) ---
add_library(armadillo_headeronly INTERFACE)
target_include_directories(armadillo_headeronly INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/armadillo/include)
target_compile_definitions(armadillo_headeronly INTERFACE
  ARMA_DONT_USE_WRAPPER
  ARMA_USE_LAPACK
  ARMA_USE_BLAS
  ARMA_DONT_USE_OPENMP)

# --- Vendored header-only Ensmallen ---
add_library(ensmallen_headeronly INTERFACE)
target_include_directories(ensmallen_headeronly INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ensmallen/include)

# --- Extension module ---
pybind11_add_module(_gsvb_core
  cpp/bindings.cpp
  cpp/rmath.cpp
  cpp/utils.cpp
  cpp/linear.cpp
  cpp/logistic.cpp
  cpp/poisson.cpp)

target_include_directories(_gsvb_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(_gsvb_core PRIVATE
  armadillo_headeronly
  ensmallen_headeronly
  ${LAPACK_LIBRARIES}
  ${BLAS_LIBRARIES})

install(TARGETS _gsvb_core DESTINATION gsvbpy)
