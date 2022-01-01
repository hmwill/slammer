add_executable(frontend_gui main.cpp)

target_include_directories(frontend_gui PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(frontend_gui slammer_lib arrow_shared ${OpenCV_LIBS})

add_dependencies(frontend_gui slammer_lib)
