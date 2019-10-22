cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

find_package(OpenCV REQUIRED)
# find_package(Qt5Core)
# find_package(Qt5Gui)
# find_package(Qt5Widgets)
# find_package(Qt5Test)
# find_package(Qt5Concurrent)

include_directories(
  include)

add_executable(ORB_extractor_and_brute_force_matcher
  src/ORB_extractor_and_brute_force_matcher_main.cpp
  )

target_link_libraries(ORB_extractor_and_brute_force_matcher
  ${OpenCV_LIBS}
  )

