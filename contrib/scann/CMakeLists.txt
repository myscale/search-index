cmake_minimum_required(VERSION 3.14)
project(scann)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(MACHINE_FLAGS "-mavx2 -mfma -fsized-deallocation -w")

set(COMMON_WARNING_FLAGS "-Wall -Werror -Wnon-virtual-dtor -Wno-sign-compare -Wno-unused-variable -Wno-unused-function -Wno-unused-result")

if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # options for standalone building
    message(STATUS "scann standalone mode")
    set(CMAKE_BUILD_COLOR_MAKEFILE  ON)
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo;Debug;Release;MinSizeRel" CACHE STRING "" FORCE)

    IF(NOT CMAKE_BUILD_TYPE)
        message(STATUS "CMAKE_BUILD_TYPE is not set, set to default = RELWITHDEBINFO")
        set(CMAKE_BUILD_TYPE            "RELWITHDEBINFO")
    ENDIF()
    MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )

    set(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O3 -g")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g -DDEBUG")

    set(CMAKE_EXE_LINKER_FLAGS              "-static-libgcc -static-libstdc++")

    set(CMAKE_CXX_FLAGS "-std=c++20 -D_GLIBCXX_USE_CXX11_ABI=1 ${COMMON_WARNING_FLAGS} ${MACHINE_FLAGS}")
else()
    # integrated in search-index
    message(STATUS "scann search-index integration mode")
    # TODO how to solve CXX11_ABI issue?
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_WARNING_FLAGS} ${MACHINE_FLAGS}")
    set(add_compile_definitions SEARCH_INDEX_INTEGRATION_MODE)
endif()


message(STATUS "CXX_FLAGS:=${CMAKE_CXX_FLAGS}")

message(STATUS "CMAKE_MODULE_PATH:=${CMAKE_MODULE_PATH}")

OPTION(SCANN_PYBIND "Build ScaNN with python bindings" ON)
if (SCANN_PYBIND)
    message(STATUS "Compile ScaNN with -fPIC")
    # -fPIC is required for building python extensions
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif()

# System Checking
find_package(Threads)

set(SCANN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "SCANN Include Dir: ${SCANN_INCLUDE_DIRS}")

# add C++ macro for building with cmake
add_definitions(-DSCANN_CMAKE_MODE=1)

file(GLOB_RECURSE SCANN_SOURCE_FILES
     "${CMAKE_CURRENT_SOURCE_DIR}/scann/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/scann/*.cc"
     "${CMAKE_CURRENT_SOURCE_DIR}/tpl-out/*.cc"
    )
file(GLOB_RECURSE REMOVE_SOURCE_FILES
     "${CMAKE_CURRENT_SOURCE_DIR}/scann/hashes/internal/*.tpl.cc" # exclude tpl files
     "${CMAKE_CURRENT_SOURCE_DIR}/scann/scann_ops/cc/*" # pybind11 module
    )
list(REMOVE_ITEM SCANN_SOURCE_FILES ${REMOVE_SOURCE_FILES})

file(GLOB_RECURSE TSL_SOURCE_FILES
     "${CMAKE_CURRENT_SOURCE_DIR}/contrib/tensorflow/src/*.cc"
    )

add_library(scann_s ${SCANN_SOURCE_FILES} ${TSL_SOURCE_FILES})

include_directories(${SCANN_INCLUDE_DIRS})
target_include_directories(scann_s PUBLIC ${SCANN_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib/tensorflow/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib/tensorflow/include/src)

# use ClickHouse abseil in MYSCALE_MODE
if (NOT MYSCALE_MODE)
    set(ABSL_PROPAGATE_CXX_STD ON)
    add_subdirectory(contrib/abseil-cpp)
endif()

# nsync
add_subdirectory(contrib/nsync)
target_include_directories(scann_s PUBLIC contrib/nsync/public)

# double-conversion
if (NOT MYSCALE_MODE)
    message(STATUS "scann link to local double-conversion")
    add_subdirectory(contrib/double-conversion)
    target_include_directories(scann_s PUBLIC contrib/double-conversion)
    set(DoubleConversion_Libs double-conversion)
else()
    message(STATUS "sann link to ClickHouse double-conversion")
    target_include_directories(scann_s PUBLIC ${CMAKE_SOURCE_DIR}/contrib/double-conversion)
    set(DoubleConversion_Libs _double-conversion)
endif()

# protobuf
if (NOT MYSCALE_MODE)
    message(STATUS "scann link to local protobuf")
    set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")
    add_subdirectory(contrib/protobuf/cmake contrib/protobuf)
    set(Protobuf_Libs protobuf::libprotobuf)
else()
    message(STATUS "scann link to ClickHouse protobuf")
    set(Protobuf_Libs ch_contrib::protobuf)
endif()

target_link_libraries(scann_s PUBLIC
    nsync_cpp ${DoubleConversion_Libs}
    ${Protobuf_Libs}
    ${CMAKE_THREAD_LIBS_INIT}
)

target_link_libraries(scann_s PUBLIC
    absl::core_headers
    absl::flags
    absl::base
    absl::memory
    absl::flat_hash_set
    absl::flat_hash_map
    absl::status
    absl::time
    absl::random_random
    absl::random_distributions
)

if (MYSCALE_MODE)
    add_library(ch_contrib::scann ALIAS scann_s)
endif()

# build python extension module
if(SCANN_PYBIND)
    message(STATUS "Building pybind module")
    add_subdirectory(scann/scann_ops)
endif()
