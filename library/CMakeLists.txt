add_library(slammer_lib 
  src/slammer.cpp src/events.cpp src/camera.cpp
  src/backend.cpp src/pipeline.cpp
  src/frontend.cpp src/map.cpp 
  src/descriptor.cpp
  src/flow.cpp
  src/image.cpp
  src/imu.cpp 
  src/math.cpp
  src/keyframe_index.cpp
  src/optimizer.cpp
  src/orb.cpp
  src/pnp.cpp
  src/loop_pose_optimizer.cpp
  src/pose_graph_optimizer.cpp
  src/poses_locations_optimizer.cpp
  src/utility.cpp
  src/loris/driver.cpp src/loris/opencv_utils.cpp
)


target_include_directories(slammer_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(slammer_lib
  fmt::fmt png
)

add_executable(test_slammer_lib 
  tests/slammer_test.cpp tests/events_test.cpp
  tests/backend_test.cpp tests/pipeline_test.cpp
  tests/frontend_test.cpp tests/map_test.cpp 
  tests/descriptor_test.cpp
  tests/flow_test.cpp
  tests/image_test.cpp
  tests/imu_test.cpp 
  tests/math_test.cpp
  tests/keyframe_index_test.cpp
  tests/optimizer_test.cpp
  tests/loop_pose_optimizer_test.cpp
  tests/pose_graph_optimizer_test.cpp
  tests/orb_test.cpp
  tests/utility_test.cpp
  tests/loris/driver_test.cpp tests/loris/opencv_utils_test.cpp
)

add_dependencies(test_slammer_lib slammer_lib)

target_link_libraries(test_slammer_lib 
  slammer_lib ${OpenCV_LIBS} GTest::gtest GTest::gtest_main
  ${Boost_LIBRARIES}
  Eigen3::Eigen
  png
  fmt::fmt
  sciplot::sciplot
)

add_test(NAME SlammerLibTests COMMAND test_slammer_lib WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})