cmake_minimum_required(VERSION 2.8.3)
project(voice_command_listener)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs actionlib)

set( CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

## Declare a catkin package
catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(voice_command_listener voice_command_listener.cpp)
target_link_libraries(voice_command_listener ${catkin_LIBRARIES})

add_executable(nao_audio nao_audio.cpp)
target_link_libraries(nao_audio ${catkin_LIBRARIES})