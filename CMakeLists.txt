cmake_minimum_required(VERSION 2.8)
project(pyramid)

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

add_executable(ihaar app/ihaar.c
  ${HDRS}
  ${SRCS})
target_link_libraries(ihaar mylib)

add_executable(zorder app/zorder.c
  ${HDRS}
  ${SRCS})
target_link_libraries(zorder mylib)

###############################################################################
#  Testing
###############################################################################
  enable_testing(true)
  add_subdirectory(test)
