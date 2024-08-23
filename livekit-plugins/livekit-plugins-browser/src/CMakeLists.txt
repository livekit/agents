include(FetchContent)

set(FETCHCONTENT_QUIET off)

# I don't want to write a different code per platform for the dev mode.
# so use glfw and imgui like I do for my other side projects...
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui GIT_TAG origin/docking)
FetchContent_GetProperties(imgui)
FetchContent_MakeAvailable(imgui)
file(GLOB IMGUI_SOURCES ${imgui_SOURCE_DIR}/*.cpp)
add_library(imgui STATIC ${IMGUI_SOURCES}
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
)
set_target_properties(imgui PROPERTIES CXX_STANDARD 17)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/misc/cpp ${imgui_SOURCE_DIR}/backends ${GLFW_INCLUDE_DIR})
target_link_libraries(imgui PRIVATE glfw)


set(LKCEF_SRCS app.cpp app.hpp handler.hpp handler.cpp dev_renderer.hpp dev_renderer.cpp gleq.h browser_handle.hpp browser_handle.cpp)
set(LKCEF_SRCS_LINUX main_linux.cpp)
set(LKCEF_SRCS_MAC app_mac.mm)
set(LKCEF_SRCS_WINDOWS main_win.cpp )
append_platform_sources(LKCEF_SRCS)
source_group(lkcef FILES ${LKCEF_SRCS})

set(LKCEF_HELPER_SRCS )
set(LKCEF_HELPER_SRCS_LINUX helper_main_linux.cpp)
set(LKCEF_HELPER_SRCS_MAC helper_main_mac.mm)
set(LKCEF_HELPER_SRCS_WINDOWS helper_main_win.cpp)
append_platform_sources(LKCEF_HELPER_SRCS)
source_group(lkcef FILES ${LKCEF_HELPER_SRCS})

set(LKCEF_PYTHON_SRCS agents_python.hpp
               agents_python.cpp)

if(OS_LINUX OR OS_WINDOWS)
  # Logical target used to link the libcef library on Linux and Windows. On
  # macOS the CEF framework is loaded dynamically at startup.
  add_logical_target("libcef_lib" "${CEF_LIB_DEBUG}" "${CEF_LIB_RELEASE}")
endif()

set_cef_target_out_dir() # Determine the target output directory.

if(OS_LINUX)
  # Helper executable target.
  add_executable(lkcef_helper ${LKCEF_HELPER_SRCS})
  set_executable_target_properties(lkcef_helper)
  add_dependencies(lkcef_helper libcef_dll_wrapper)
  target_link_libraries(lkcef_helper libcef_lib libcef_dll_wrapper
                        ${CEF_STANDARD_LIBS})

  # Set rpath so that libraries can be placed next to the executable.
  set_target_properties(lkcef_helper PROPERTIES INSTALL_RPATH "$ORIGIN")
  set_target_properties(lkcef_helper PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

  # library target.
  add_library(lkcef SHARED ${LKCEF_SRCS})
  set_library_target_properties(lkcef)
  add_dependencies(lkcef libcef_dll_wrapper lkcef_helper)
  target_link_libraries(lkcef libcef_lib libcef_dll_wrapper
                        ${CEF_STANDARD_LIBS})

  # Set rpath so that libraries can be placed next to the library.
  set_target_properties(lkcef PROPERTIES INSTALL_RPATH "$ORIGIN")
  set_target_properties(lkcef PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

  # Copy binary and resource files to the target output directory.
  copy_files("lkcef" "${CEF_BINARY_FILES}" "${CEF_BINARY_DIR}"
             "${CEF_TARGET_OUT_DIR}")
  copy_files("lkcef" "${CEF_RESOURCE_FILES}" "${CEF_RESOURCE_DIR}"
             "${CEF_TARGET_OUT_DIR}")
endif()

if(OS_MAC)
  # Avoid CMP0042 policy errors.
  set(CMAKE_MACOSX_RPATH 1)

  # Avoid CMP0068 policy errors.
  if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
  endif()

  add_executable(lkcef_app MACOSX_BUNDLE dummy.cpp) # dummy app
  set_target_properties(lkcef_app PROPERTIES
          MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/resources/lkcefapp-Info.plist"
          OUTPUT_NAME "lkcef_app"
  )


  # library target.
  add_library(lkcef STATIC ${LKCEF_SRCS})
  set_library_target_properties(lkcef)
  add_dependencies(lkcef libcef_dll_wrapper)
  target_include_directories(lkcef PRIVATE ${GLFW_INCLUDE_DIR})
  target_link_libraries(lkcef libcef_dll_wrapper ${CEF_STANDARD_LIBS} glfw imgui)

  add_custom_command(
    TARGET lkcef
    POST_BUILD
    # Copy the CEF framework into the main app bundle.
    COMMAND
      ${CMAKE_COMMAND} -E copy_directory
      "${CEF_BINARY_DIR}/Chromium Embedded Framework.framework"
      "$<TARGET_BUNDLE_DIR:lkcef_app>/Contents/Frameworks/Chromium Embedded Framework.framework"
    VERBATIM)

  # Create the multiple Helper app bundle targets.
  foreach(_suffix_list ${CEF_HELPER_APP_SUFFIXES})
    # Convert to a list and extract the suffix values.
    string(REPLACE ":" ";" _suffix_list ${_suffix_list})
    list(GET _suffix_list 0 _name_suffix)
    list(GET _suffix_list 1 _target_suffix)
    list(GET _suffix_list 2 _plist_suffix)

    # Define Helper target and output names.
    set(_helper_target "lkcef_Helper${_target_suffix}")
    set(_helper_output_name "lkcef Helper${_name_suffix}")

    # Create Helper-specific variants of the helper-Info.plist file.
    set(_helper_info_plist
        "${CMAKE_CURRENT_BINARY_DIR}/lkcef-Info${_target_suffix}.plist")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/resources/lkcefhelper-Info.plist"
         _plist_contents)
    string(REPLACE "\${EXECUTABLE_NAME}" "${_helper_output_name}"
                   _plist_contents ${_plist_contents})
    string(REPLACE "\${PRODUCT_NAME}" "${_helper_output_name}" _plist_contents
                   ${_plist_contents})
    string(REPLACE "\${BUNDLE_ID_SUFFIX}" "${_plist_suffix}" _plist_contents
                   ${_plist_contents})
    file(WRITE ${_helper_info_plist} ${_plist_contents})

    # Create Helper executable target.
    add_executable(${_helper_target} MACOSX_BUNDLE ${LKCEF_HELPER_SRCS})
    set_executable_target_properties(${_helper_target})
    add_dependencies(${_helper_target} libcef_dll_wrapper)
    target_link_libraries(${_helper_target} libcef_dll_wrapper
                          ${CEF_STANDARD_LIBS})


    set_target_properties(
      ${_helper_target}
      PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${_helper_info_plist}
                 OUTPUT_NAME ${_helper_output_name})

    # Add the Helper as a dependency of the main executable target.
    add_dependencies(lkcef "${_helper_target}")

    # Copy the Helper app bundle into the Frameworks directory.
    add_custom_command(
      TARGET lkcef
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        "${CEF_TARGET_OUT_DIR}/${_helper_output_name}.app"
        "$<TARGET_BUNDLE_DIR:lkcef_app>/Contents/Frameworks/${_helper_output_name}.app"
      VERBATIM)
  endforeach()
endif()

if(OS_WINDOWS)
  # Helper executable target.
  add_executable(lkcef_helper WIN32 ${LKCEF_HELPER_SRCS})
  set_executable_target_properties(lkcef_helper)
  add_dependencies(lkcef_helper libcef_dll_wrapper)
  target_link_libraries(lkcef_helper libcef_lib libcef_dll_wrapper
                        ${CEF_STANDARD_LIBS})

  # library target.
  add_library(lkcef SHARED ${LKCEF_SRCS})
  set_library_target_properties(lkcef)
  add_dependencies(lkcef libcef_dll_wrapper lkcef_helper)
  target_link_libraries(lkcef libcef_lib libcef_dll_wrapper
                        ${CEF_STANDARD_LIBS})

  # Add the custom manifest files to the DLL and helper EXE.
  add_windows_manifest("${CMAKE_CURRENT_SOURCE_DIR}" "lkcef" "dll")
  add_windows_manifest("${CMAKE_CURRENT_SOURCE_DIR}" "lkcef_helper" "exe")

  # Copy binary and resource files to the target output directory.
  copy_files("lkcef" "${CEF_BINARY_FILES}" "${CEF_BINARY_DIR}"
             "${CEF_TARGET_OUT_DIR}")
  copy_files("lkcef" "${CEF_RESOURCE_FILES}" "${CEF_RESOURCE_DIR}"
             "${CEF_TARGET_OUT_DIR}")
endif()


# TODO(theomonnom): should be pretty similar for NodeJS
pybind11_add_module(lkcef_python ${LKCEF_PYTHON_SRCS})

set_target_properties(lkcef_python PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(lkcef_python PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

target_include_directories(lkcef_python PRIVATE ${CEF_INCLUDE_PATH})
target_link_libraries(lkcef_python PUBLIC lkcef)
target_link_libraries(lkcef_python PUBLIC libcef_dll_wrapper ${CEF_STANDARD_LIBS})
add_dependencies(lkcef_python libcef_dll_wrapper)
