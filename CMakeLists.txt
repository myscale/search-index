cmake_minimum_required(VERSION 3.14)
project(myscale_search_index LANGUAGES C CXX)
set(SearchIndex_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# C++20 is required for MyScale compilation
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_EXTENSIONS ON)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# RelWithDebInfo mode by default (similar to MyScale)
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UC)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast")
option(SI_BUILD_WITH_LIBCXX "Build with libc++ instead of libstdc++" OFF)
if (SI_BUILD_WITH_LIBCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Set to Debug mode")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -ggdb -DDEBUG")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -ggdb")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    # release mode options
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
    add_compile_options(-DNDEBUG)
endif()

option(MYSCALE_MODE "Perform search in MyScale Database" OFF)
if(MYSCALE_MODE)
    # for MyScale Mode, exception handling, logging relies on CK mechanism
    message(STATUS "Build in MyScale Mode")
    option (NSYNC_ENABLE_TESTS "Disable for nsync tests in MYSCALE_MODE" OFF)
else()
    message(STATUS "Build in Standalone Mode")
    # -DSANITIZE=address
    include(cmake/ccache.cmake)
    include (cmake/sanitize.cmake)
    include (cmake/arch.cmake)
    set (OS_LINUX (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    add_compile_options(-fcolor-diagnostics)
endif()

## option to dump memory usage
OPTION(DUMP_MEMORY_USAGE "dump memory usage" OFF)
if(DUMP_MEMORY_USAGE)
    add_compile_definitions(USE_MEMORY_RECORDER=1)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/SearchIndex)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib/faiss)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib/simde)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

if (MYSCALE_MODE)
    # include MyScale DB related headers for logging
    include_directories(${CMAKE_SOURCE_DIR}/base)
    include_directories(${CMAKE_SOURCE_DIR}/src)
    include_directories(${CMAKE_SOURCE_DIR}/contrib/boost)
    include_directories(${CMAKE_SOURCE_DIR}/contrib/fmtlib/include)
    include_directories(${CMAKE_SOURCE_DIR}/base/poco/Foundation/include)
    include_directories(${CMAKE_SOURCE_DIR}/base/poco/Util/include)
    include_directories(${CMAKE_SOURCE_DIR}/base/poco/XML/include)
    include_directories(${CMAKE_SOURCE_DIR}/contrib/magic_enum/include)
    include_directories(${CMAKE_SOURCE_DIR}/contrib/cityhash102/include)
else()
    add_compile_options(-Werror)
endif()

file(GLOB SOURCE_FILES "SearchIndex/*.h" "SearchIndex/*.cpp" "SearchIndex/Common/*.cpp")
file(GLOB MAIN_FILES "SearchIndex/main_*.cpp" "SearchIndex/test_*.cpp"
        "SearchIndex/lib_*.cpp" "SearchIndex/demo_*.cpp")
foreach(f ${MAIN_FILES})
    list(REMOVE_ITEM SOURCE_FILES ${f})
endforeach()

find_package(MKL CONFIG)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# list utility libraries for linking
set(VECTOR_INDEX_UTIL_LIBS OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

SET(VECTOR_INDEX_LINK_LIBS ${VECTOR_INDEX_UTIL_LIBS})

# faiss submodule
option(ENABLE_FAISS "Enable Faiss for MyScaleSearchIndex" ON)
if(ENABLE_FAISS)
    message(STATUS "Enable Faiss")
    set(FAISS_OPT_LEVEL avx2)
    set(FAISS_ENABLE_GPU OFF)
    if(MYSCALE_MODE)
        set(FAISS_ENABLE_PYTHON OFF)
    else()
        set(FAISS_ENABLE_PYTHON ON)
    endif()
    set(BUILD_TESTING OFF)
    add_subdirectory(contrib/faiss)
    link_directories(${CMAKE_CURRENT_BINARY_DIR}/contrib/faiss/build/faiss)
    list(APPEND VECTOR_INDEX_LINK_LIBS faiss)
endif()

# ScaNN module
option(ENABLE_SCANN "Enable ScaNN for MyScaleSearchIndex" ON)
if(ENABLE_SCANN)
    message(STATUS "Enable ScaNN")
    set(SCANN_PBYIND OFF)
    add_subdirectory(contrib/scann)
    include_directories(contrib/scann)
    link_directories(${CMAKE_CURRENT_BINARY_DIR}/contrib/scann/build)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib/scann/contrib/tensorflow/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib/scann/contrib/tensorflow/include/src)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib/scann/contrib/usr/local/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/contrib/scann/contrib/abseil-cpp)
    list(APPEND VECTOR_INDEX_LINK_LIBS scann_s)
else()
    # Remove SCANN related source files
    list(FILTER SOURCE_FILES EXCLUDE REGEX "SearchIndex/ScaNNIndex.*")
endif()
message(STATUS "SOURCE_FILES: ${SOURCE_FILES}")

message(STATUS "VECTOR_INDEX_LINK_LIBS: ${VECTOR_INDEX_LINK_LIBS}")

add_library(search_index OBJECT ${SOURCE_FILES})
set_target_properties(search_index PROPERTIES LINKER_LANGUAGE CXX)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/SearchIndex/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/SearchIndex/Config.h)

# build shared & static libraries
add_library(${PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:search_index>)
target_link_libraries(${PROJECT_NAME}_static PUBLIC ${VECTOR_INDEX_LINK_LIBS})
target_include_directories(${PROJECT_NAME}_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME}_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/contrib)
target_include_directories(${PROJECT_NAME}_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/contrib/faiss)
target_include_directories(${PROJECT_NAME}_static PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)

if(MYSCALE_MODE)
    add_library(ch_contrib::search_index ALIAS ${PROJECT_NAME}_static)
endif()

## boost-ext/ut unit tests
include(cmake/AddCustomCommandOrTest.cmake)

function(add_ut_test file target)
    add_executable(boost_ut_${target} ${file})
    ut_add_custom_command_or_test(TARGET boost_ut_${target} COMMAND boost_ut_${target} ${ARGN})
endfunction()

add_ut_test(tests/VectorIndexTest.cpp test_vector_index)
target_link_libraries(boost_ut_test_vector_index PUBLIC ${PROJECT_NAME}_static)
