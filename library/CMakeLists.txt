add_library(slammer_lib 
  src/slammer.cpp src/events.cpp src/camera.cpp
  src/backend.cpp src/frontend.cpp src/map.cpp 
  src/imu.cpp src/landmark.cpp
  src/math.cpp
  src/loris/arrow_utils.cpp src/loris/driver.cpp src/loris/opencv_utils.cpp src/loris/schema.cpp
)

target_include_directories(slammer_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(test_slammer_lib 
  tests/slammer_test.cpp tests/events_test.cpp
  tests/backend_test.cpp tests/frontend_test.cpp tests/map_test.cpp
  tests/imu_test.cpp tests/landmark_test.cpp
  tests/math_test.cpp
  tests/loris/arrow_utils_test.cpp tests/loris/driver_test.cpp tests/loris/opencv_utils_test.cpp
)

add_dependencies(test_slammer_lib slammer_lib)

target_link_libraries(test_slammer_lib 
  slammer_lib arrow_shared ${OpenCV_LIBS} GTest::gtest GTest::gtest_main
)

add_test(NAME SlammerLibTests COMMAND test_slammer_lib WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})