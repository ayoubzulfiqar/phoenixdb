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

# Fail early: without this, a missing toolchain surfaces as an opaque
# "Error 1" from the custom command below, with no mention of Rust.
find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
  message(FATAL_ERROR
    "phoenixdb: `cargo` was not found on PATH.\n"
    "The Linux plugin builds its native library from Rust source.\n"
    "Install the Rust toolchain from https://rustup.rs and reopen your shell.")
endif()

set(PHOENIX_LIB "${RUST_DIR}/target/${CARGO_OUT_DIR}/libphoenixdb.so")

# `sql` is required, not optional: `phoenix_sql_query` and `phoenix_has_sql`
# are part of ABI v3, and the Dart loader rejects a library reporting a
# different version. The vector engine (`phoenix_vector_*`, also v3) needs no
# flag: it has no optional dependencies and is always compiled in. Building without it produces a library the app cannot load.
set(PHOENIX_FEATURES "--features" "sql")

add_custom_command(
  OUTPUT ${PHOENIX_LIB}
  COMMAND cargo build --lib ${CARGO_PROFILE_FLAG} ${PHOENIX_FEATURES}
  WORKING_DIRECTORY ${RUST_DIR}
  COMMENT "Building PhoenixDB Rust library (${CARGO_OUT_DIR}, features: sql)"
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
