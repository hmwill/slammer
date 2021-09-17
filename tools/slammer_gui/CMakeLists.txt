add_executable(slammer_gui src/main.cpp)
target_link_libraries(slammer_gui slammer_lib)

add_dependencies(slammer_gui slammer_lib)