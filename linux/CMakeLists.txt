# PhoenixDB — Linux (Flutter desktop) plugin.
#
# Builds the Rust cdylib during the Flutter build and installs it next to the
# executable, where the bundle's rpath ($ORIGIN/lib) picks it up.

cmake_minimum_required(VERSION 3.13)
set(PROJECT_NAME "phoenixdb")
project(${PROJECT_NAME} LANGUAGES CXX)

set(PLUGIN_NAME "${PROJECT_NAME}_plugin")
set(RUST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rust")

# Flutter builds Debug/Profile/Release; only Debug maps to a cargo debug build.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CARGO_PROFILE_FLAG "")
  set(CARGO_OUT_DIR "debug")
else()
  set(CARGO_PROFILE_FLAG "--release")
  set(CARGO_OUT_DIR "release")
endif()

set(PHOENIX_LIB "${RUST_DIR}/target/${CARGO_OUT_DIR}/libphoenixdb.so")

add_custom_command(
  OUTPUT ${PHOENIX_LIB}
  COMMAND cargo build --lib ${CARGO_PROFILE_FLAG}
  WORKING_DIRECTORY ${RUST_DIR}
  COMMENT "Building PhoenixDB Rust library (${CARGO_OUT_DIR})"
  VERBATIM
)
add_custom_target(phoenixdb_rust ALL DEPENDS ${PHOENIX_LIB})

# An empty target satisfies Flutter's plugin contract; all behaviour is in the
# bundled shared library, reached through dart:ffi rather than a method channel.
add_library(${PLUGIN_NAME} INTERFACE)
add_dependencies(${PLUGIN_NAME} phoenixdb_rust)

set(phoenixdb_bundled_libraries
  "${PHOENIX_LIB}"
  PARENT_SCOPE
)
