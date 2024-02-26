# project name inherits from parent project
cmake_minimum_required(VERSION 3.14)

execute_process(COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

execute_process(COMMAND python -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE _numpy_include
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)

# ignore the tensorflow kernel/ops bindings
file(GLOB PYSCANN_SOURCE_FILES
     "${CMAKE_CURRENT_SOURCE_DIR}/cc/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/cc/*.cc"
     "${CMAKE_CURRENT_SOURCE_DIR}/cc/python/scann_pybind.cc"
    )
message(STATUS "scann_pybind source_files: ${PYSCANN_SOURCE_FILES}")

message(STATUS "scann_pybind links protobuf: ${Protobuf_LIBRARIES}")
message(STATUS "scann_pybind links abseil: ${Abseil_LIBS}")

# _GLIBCXX_USE_CXX11_ABI=1 is required for using protobuf CXX11 ABI
# set(CMAKE_CXX_FLAGS "-std=c++20 -D_GLIBCXX_USE_CXX11_ABI=1 ${COMMON_WARNING_FLAGS} ${MACHINE_FLAGS}")
add_library(scann_pybind MODULE ${PYSCANN_SOURCE_FILES})
target_link_libraries(
        scann_pybind
        PRIVATE
        pybind11::module
        pybind11::lto
        pybind11::windows_extras
        scann_s
        ${Protobuf_LIBRARIES}
        ${Abseil_LIBS}
)

pybind11_extension(scann_pybind)

if(CMAKE_BUILD_TYPE MATCHES Release)
    message("Strip pybind module debug info")
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(scann_pybind)
    set_target_properties(scann_pybind PROPERTIES CXX_VISIBILITY_PRESET "hidden"
            CUDA_VISIBILITY_PRESET "hidden")
endif()

# generally, the VERSION_INFO flag is set by pyproject.toml, by way of setup.py.
# attempts to locate the version within CMake fail because the version has to be available
# to pyproject.toml for the sdist to work after we build it.

if(NOT "${VERSION_INFO}")
    set(VERSION_INFO "0.7.0dev")
endif()
target_compile_definitions(scann_pybind PRIVATE VERSION_INFO="${VERSION_INFO}")
