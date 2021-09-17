add_library(slammer_lib src/slammer.cpp src/loris/schema.cpp)
target_link_libraries(slammer_lib arrow_shared)

target_include_directories(slammer_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(test_slammer_lib tests/slammer_test.cpp)
add_dependencies(test_slammer_lib slammer_lib)
target_link_libraries(test_slammer_lib slammer_lib arrow_shared GTest::gtest GTest::gtest_main)

add_test(SlammerLibTests test_slammer_lib)