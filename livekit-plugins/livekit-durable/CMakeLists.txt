cmake_minimum_required(VERSION 3.18)
project(lk_durable LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
add_library(lk_durable MODULE livekit/durable/frame.c)
target_link_libraries(lk_durable PRIVATE Python::Module)
target_include_directories(lk_durable PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/livekit/durable
)

target_compile_definitions(lk_durable PRIVATE Py_BUILD_CORE)

set_target_properties(lk_durable PROPERTIES PREFIX "" OUTPUT_NAME "lk_durable" C_STANDARD 99)

if(DEFINED Python_EXTENSION_SUFFIX)
  set_target_properties(lk_durable PROPERTIES SUFFIX "${Python_EXTENSION_SUFFIX}")
endif()

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(lk_durable PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
  )
endif()
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG)
  set_target_properties(lk_durable PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}"
  )
endif()
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE)
  set_target_properties(lk_durable PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}"
  )
endif()

