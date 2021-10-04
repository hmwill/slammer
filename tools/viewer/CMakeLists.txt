add_executable(viewer src/main.cpp)

target_include_directories(viewer PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(viewer slammer_lib arrow_shared ${OpenCV_LIBS})

add_dependencies(viewer slammer_lib)
