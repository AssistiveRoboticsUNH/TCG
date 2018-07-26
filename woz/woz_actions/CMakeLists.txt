cmake_minimum_required(VERSION 2.8.3)
project(woz_actions)

find_package(catkin REQUIRED COMPONENTS
  nao_interaction_msgs
  nao_msgs
  naoqi_bridge_msgs
  roscpp
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES node_controllers
#  CATKIN_DEPENDS nao_interaction_msgs nao_msgs naoqi_bridge_msgs roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(woz_actions_point src/point.cpp)
add_dependencies(woz_actions_point ${catkin_EXPORTED_TARGETS})
target_link_libraries(woz_actions_point ${catkin_LIBRARIES})