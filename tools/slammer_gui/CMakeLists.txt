add_executable(slammer_gui src/main.cpp)

target_include_directories(slammer_gui PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(slammer_gui slammer_lib arrow_shared ${OpenCV_LIBS})

add_dependencies(slammer_gui slammer_lib)
