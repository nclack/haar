cmake_minimum_required(VERSION 2.8)
project(ndHaar)
set(CMAKE_MAJOR_VERSION 0)
set(CMAKE_MINOR_VERSION 0)
set(CMAKE_PATCH_VERSION 1a)


macro(show v)
  message("${v} is ${${v}}")
endmacro()

set(ROOT_3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty)

add_subdirectory(3rdParty/mylib)
include_directories(3rdParty/mylib)

include_directories(src)
file(GLOB SRCS src/*.c)
file(GLOB HDRS src/*.h)

add_executable(hw app/hw.c)
target_link_libraries(hw mylib)

add_executable(haar app/haar.c
  ${HDRS}
  ${SRCS})
target_link_libraries(haar mylib)

add_executable(zorder app/zorder.c
  ${HDRS}
  ${SRCS})
target_link_libraries(zorder mylib)

###############################################################################
#  Testing
###############################################################################
  enable_testing(true)
  add_subdirectory(test)

###############################################################################
#  Docs
###############################################################################
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
                 @ONLY
                 )
  add_custom_target(apidocs
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT           "Generating API Docs"
    VERBATIM
    )
endif()
