cmake_minimum_required(VERSION 3.10)
project(fastor_test)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -mavx2 -mfma -march=native")

# Find the Eigen3 library
find_package(Eigen3 REQUIRED)

# Include FetchContent module
include(FetchContent)

# Fetch Fastor
FetchContent_Declare(
        Fastor
        GIT_REPOSITORY https://github.com/romeric/Fastor.git
        GIT_TAG        master
)

# Make Fastor available
FetchContent_MakeAvailable(Fastor)

# Add the executable
add_executable(fastor_test main.cpp)

# Link Eigen3 library
target_link_libraries(fastor_test Eigen3::Eigen Fastor)