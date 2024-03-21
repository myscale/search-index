include(FindPackageHandleStandardArgs)
find_path(Abseil_INCLUDE_DIR NAMES absl)

foreach(LIB absl_base
            absl_debugging_internal
            absl_flags
            absl_flags_internal
            absl_flags_commandlineflag
            absl_flags_commandlineflag_internal
            absl_flags_marshalling
            absl_flags_parse
            absl_flags_config
            absl_flags_program_name
            absl_flags_reflection
            absl_flags_usage
            absl_flags_usage_internal
            absl_flags_private_handle_accessor
            absl_hash
            absl_malloc_internal
            absl_strings
            absl_strings_internal
            absl_time
            absl_synchronization
            absl_random_distributions
            absl_bad_any_cast_impl
            absl_bad_variant_access
            absl_bad_optional_access)
    find_library(Abseil_${LIB} NAMES ${LIB})
    list(APPEND Abseil_LIBS ${Abseil_${LIB}})
endforeach()

find_package_handle_standard_args(Abseil DEFAULT_MSG
    Abseil_LIBS
    Abseil_INCLUDE_DIR)

if (NOT Abseil_FOUND)
  message(STATUS "Can not find Abseil")
else()
  message(STATUS "Found Abseil: ${Abseil_INCLUDE_DIR} ${Abseil_LIBS}")
endif (NOT Abseil_FOUND)

mark_as_advanced(Abseil_INCLUDE_DIR Abseil_LIBS)
