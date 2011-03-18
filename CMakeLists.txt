cmake_minimum_required(VERSION 2.8)

add_subdirectory(3rdParty/mylib)
include_directories(3rdParty/mylib)

add_executable(hw app/hw.c)
target_link_libraries(hw mylib)
