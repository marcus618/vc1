cmake_minimum_required(VERSION 3.16)
project(ASSINGMENT1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
	CACHE STRING "Vcpkg toolchain file")
set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv")
find_package(OpenCV REQUIRED)
add_executable(MyProject main.cpp histogram.cpp matching.cpp featureDetection.cpp homography.cpp stitching.cpp)
target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS})
