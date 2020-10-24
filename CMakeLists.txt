cmake_minimum_required(VERSION 3.18)

project(gpuTest)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package( OpenCV 4.0.0 REQUIRED )

# mur_stereo_video
add_executable( gpuTest 
    src/main.cpp
)

target_include_directories( gpuTest
    PUBLIC ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries( gpuTest 
    ${OPENCV_LIBRARIES}
    ${OpenCV_LIBS}
)