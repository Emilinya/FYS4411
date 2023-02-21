cmake_minimum_required(VERSION 3.16)

project(project_1_fys4411)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add directories containing header-files
include_directories(
  .
  include
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cc"
    GLOB SOURCES "src/*.cc"
)

# Create executable called "main"
add_executable(main ${SOURCES})
