option(BOOST_UT_ENABLE_RUN_AFTER_BUILD "Automatically run built artifacts. If disabled, the tests can be run with ctest instead" OFF)

function(ut_add_custom_command_or_test)
  # Define the supported set of keywords
  set(prefix "PARSE")
  set(noValues "")
  set(singleValues TARGET)
  set(multiValues COMMAND)

  # Process the arguments passed in
  include(CMakeParseArguments)
  cmake_parse_arguments("${prefix}" "${noValues}" "${singleValues}" "${multiValues}" ${ARGN})

  if(BOOST_UT_ENABLE_RUN_AFTER_BUILD)
    add_custom_command(TARGET ${PARSE_TARGET} COMMAND ${PARSE_COMMAND})
  else()
    add_test(NAME ${PARSE_TARGET} COMMAND ${PARSE_COMMAND})
  endif()
endfunction()
