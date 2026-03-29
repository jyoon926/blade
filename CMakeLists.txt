cmake_minimum_required(VERSION 3.14)
project(blade CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# ---- pybind11 (from pip) ----------------------------------------------------
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS "${PYBIND11_CMAKEDIR}")

# ---- OpenMP -----------------------------------------------------------------
find_package(OpenMP REQUIRED)

# ---- C++ core library -------------------------------------------------------
set(BLADE_SOURCES
    src/tensor.cpp
    src/node.cpp
    src/ops.cpp
    src/nn/module.cpp
    src/nn/layers.cpp
    src/nn/activations.cpp
    src/nn/loss.cpp
    src/nn/pooling.cpp
    src/optim/optimizer.cpp
    src/optim/sgd.cpp
    src/optim/adam.cpp
    src/data/dataset.cpp
    src/data/mnist.cpp
)

add_library(blade_core STATIC ${BLADE_SOURCES})
target_include_directories(blade_core PUBLIC include)
target_link_libraries(blade_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(blade_core PRIVATE -O3 -Wall -Wextra -march=native)

# ---- Python extension -------------------------------------------------------
pybind11_add_module(blade MODULE python/bindings.cpp)
target_link_libraries(blade PRIVATE blade_core)
target_compile_options(blade PRIVATE -O3)
