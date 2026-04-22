cmake_minimum_required(VERSION 3.20)

project(melotts_english_onnx_cpp LANGUAGES CXX)

set(MELOTTS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/outputs")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MELOTTS_BUILD_DEMO "Build the MeloTTS ONNX CLI demo" ON)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to the ONNX Runtime installation root")

if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to your ONNX Runtime installation directory.")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS
    "${ONNXRUNTIME_ROOT}/include"
    "${ONNXRUNTIME_ROOT}/include/onnxruntime"
  REQUIRED)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS
    "${ONNXRUNTIME_ROOT}/lib"
    "${ONNXRUNTIME_ROOT}/lib64"
  REQUIRED)

set(ONNXRUNTIME_DLL_DIR "${ONNXRUNTIME_ROOT}/lib")
set(ONNXRUNTIME_DLLS
  "${ONNXRUNTIME_DLL_DIR}/onnxruntime.dll"
  "${ONNXRUNTIME_DLL_DIR}/onnxruntime_providers_shared.dll")

add_library(melotts_engine
  src/melotts_engine.cpp
)

target_include_directories(melotts_engine
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(melotts_engine
  PUBLIC
    "${ONNXRUNTIME_LIBRARY}"
)

target_compile_definitions(melotts_engine
  PRIVATE
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
)

if(MSVC)
  target_compile_options(melotts_engine PRIVATE /W4 /EHsc)
else()
  target_compile_options(melotts_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MELOTTS_BUILD_DEMO)
  add_executable(melotts_cli demo/melotts_cli.cpp)
  target_link_libraries(melotts_cli PRIVATE melotts_engine)
  add_custom_command(TARGET melotts_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_DLLS} $<TARGET_FILE_DIR:melotts_cli>)

  add_executable(melotts_stream_cli demo/melotts_stream_cli.cpp)
  target_link_libraries(melotts_stream_cli PRIVATE melotts_engine)
  add_custom_command(TARGET melotts_stream_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_DLLS} $<TARGET_FILE_DIR:melotts_stream_cli>)
endif()

add_custom_target(distclean
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${MELOTTS_OUTPUT_DIR}"
  COMMENT "Removes MeloTTS.C runtime output files"
)
