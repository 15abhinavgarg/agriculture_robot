cmake_minimum_required(VERSION 3.8)
project(agriculture_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(xacro REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ros2_control REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(example_interfaces REQUIRED)

# this one is for custom interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Num.msg"
  "msg/Sphere.msg"
  "srv/AddThreeInts.srv"
  DEPENDENCIES geometry_msgs 
)

install(DIRECTORY
urdf
meshes
launch
worlds
config
srv
msg
DESTINATION share/${PROJECT_NAME}/
)

# # for service 
# add_executable(server src/service_member_function.py)
# set_target_properties(server PROPERTIES
#   LANGUAGE PYTHON
# )
# ament_target_dependencies(server rclpy example_interfaces)

# add_executable(client src/client_member_function.py)
# ament_target_dependencies(client rclpy example_interfaces)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  src/pick_and_place.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
    DIRECTORY include/agriculture_robot
    DESTINATION include
    )

install(
  PROGRAMS
  launch/robot_description_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# install(TARGETS
#     server
#     client
#   DESTINATION lib/${PROJECT_NAME})

ament_package()
