cmake_minimum_required(VERSION 3.19)
project(lk_blingfire)

set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PythonInterp REQUIRED)
find_package(pybind11 REQUIRED)

message(STATUS "Using Python: ${PYTHON_EXECUTABLE}")

include(FetchContent)
FetchContent_Declare(blingfire
                     GIT_REPOSITORY https://github.com/microsoft/BlingFire)
FetchContent_MakeAvailable(blingfire)

include_directories(${blingfire_SOURCE_DIR})

pybind11_add_module(lk_blingfire src/main.cpp)
target_link_libraries(lk_blingfire PRIVATE blingfiretokdll_static fsaClient)

# remove hardcoded values from Blingfire builds
if(WIN32 AND MSVC)
  set(_no_debug_targets
      fsaClient
      fsaClientTiny
      fsaCompile
      blingfiretokdll_static
      lk_blingfire)

  foreach(t ${_no_debug_targets})
    get_target_property(_copts ${t} COMPILE_OPTIONS)
    if(_copts)
      list(REMOVE_ITEM _copts "/Zi" "/ZI" "/Z7" "/GL")
      set_target_properties(${t} PROPERTIES COMPILE_OPTIONS "${_copts}")
    endif()

    get_target_property(_lopts ${t} LINK_OPTIONS)
    if(_lopts)
      list(REMOVE_ITEM _lopts "/DEBUG" "/LTCG")
      set_target_properties(${t} PROPERTIES LINK_OPTIONS "${_lopts}")
    endif()

    target_link_options(${t} PRIVATE "/DEBUG:NONE" "/PDB:")
  endforeach()
endif()
