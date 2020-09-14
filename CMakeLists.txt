cmake_minimum_required (VERSION 3.12 FATAL_ERROR)

project (UnscentedKalmanFilterCA
  VERSION 0.1.0.0
  DESCRIPTION "Unscented Kalman Filter with uniform acceleration dynamic model."
  LANGUAGES CXX
)

add_executable (${PROJECT_NAME})

find_package (OpenCV 4 REQUIRED
  COMPONENTS
    highgui
    tracking
)

target_include_directories (${PROJECT_NAME}
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries (${PROJECT_NAME}
  PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options (${PROJECT_NAME}
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
      -Wall
      -Wextra
      -Wpedantic
      -pedantic-errors
    >
    $<$<CXX_COMPILER_ID:MSVC>:
      /Wall
      /EHsc
      /MP
    >
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:
      /Zi
      /MDd
    >
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:
      /MD
    >
)

set (SOURCE_FILES
  "./src/main.cpp"
  "./src/motion_model.cpp"
  "./src/motion_model.hpp"
  "./src/sliding_vector.hpp"
)

source_group (TREE "${CMAKE_CURRENT_SOURCE_DIR}"
  FILES
    ${SOURCE_FILES}
)

target_sources (${PROJECT_NAME}
  PRIVATE
    ${SOURCE_FILES}
)

set_target_properties (${PROJECT_NAME}
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    USE_FOLDERS ON
)
