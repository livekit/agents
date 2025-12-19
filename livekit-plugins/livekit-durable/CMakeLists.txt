cmake_minimum_required(VERSION 3.18)
project(lk_durable LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
add_library(lk_durable MODULE livekit/durable/frame.c)
target_link_libraries(lk_durable PRIVATE Python::Module)
target_include_directories(lk_durable PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/livekit/durable
)

# target_compile_definitions(lk_durable PRIVATE Py_BUILD_CORE)

# from https://github.com/pybind/pybind11/blob/d4f9cfbc2866f2156e1b17cb478a67088c6063f6/tools/pybind11NewTools.cmake#L146
execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
    "import sys, importlib; s = importlib.import_module('distutils.sysconfig' if sys.version_info < (3,10) else 'sysconfig'); print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO') or '')"
  OUTPUT_VARIABLE _PY_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_PY_EXT_SUFFIX STREQUAL "")
  message(FATAL_ERROR "Could not determine Python EXT_SUFFIX/SO")
endif()

get_filename_component(_PY_DEBUG_POSTFIX "${_PY_EXT_SUFFIX}" NAME_WE)
get_filename_component(_PY_SUFFIX        "${_PY_EXT_SUFFIX}" EXT)

set_target_properties(lk_durable PROPERTIES
  PREFIX ""
  OUTPUT_NAME "lk_durable"
  DEBUG_POSTFIX "${_PY_DEBUG_POSTFIX}"
  SUFFIX "${_PY_SUFFIX}"
)

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

