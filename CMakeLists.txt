cmake_minimum_required(VERSION 2.8)
project(pyramid)

add_subdirectory(3rdParty/mylib)
include_directories(3rdParty/mylib)

add_executable(hw app/hw.c)
target_link_libraries(hw mylib)

###############################################################################
#  Testing
###############################################################################
  enable_testing(true)
  add_subdirectory(test)
