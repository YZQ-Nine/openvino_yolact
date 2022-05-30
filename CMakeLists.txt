cmake_minimum_required(VERSION 3.19)
project(yolact_ncs2_openvino_2021_4)

set(CMAKE_CXX_STANDARD 11)

Add_Definitions(-DUN_BLOOD_IMAGE -DWINDOWS -DWIN32_LEAN_AND_MEAN)

set(HEADERS
        detector.h
)

set(SOURCE
        detector.cpp
        main.cpp
        )

#/FORCE:MULTIPLE 
set(OpenCV_DIR "C:\\Program Files (x86)\\Intel\\openvino_2021.4.752\\opencv")
set(OpenCV_INCLUDE_DIRS "C:\\Program Files (x86)\\Intel\\openvino_2021.4.752\\opencv\\include")
set(OpenCV_LIBS "E:\\Dev\\TCTDetect\\libs\\opencv452\\build\\x64\\vc15\\lib\\opencv_world452.lib")

set(InferenceEngine "C:\\Program Files (x86)\\Intel\\openvino_2021.4.752\\inference_engine\\include")

set(ngraph_DIR "C:\\Program Files (x86)\\Intel\\openvino_2021.4.752\\deployment_tools\\ngraph")
set(ngraph_INCLUDE_DIRS "C:\\Program Files (x86)\\Intel\\openvino_2021.4.752\\deployment_tools\\ngraph\\include")
set(ngraph_LIBS "C:\\Program Files (x86)\\Intel\\openvino_2021.4.752\\deployment_tools\\ngraph\\lib\\ngraph.lib")


#set(OpenCV_FOUND TRUE)
#find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS}\\opencv2) #6

include_directories(${InferenceEngine})
include_directories(${ngraph_INCLUDE_DIRS})


include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(yolact_ncs2_openvino_2021_4 ${HEADERS} ${SOURCE})

set(
        LINKLIB
        ${OpenCV_LIBS}
        ${ngraph_LIBS}
)

target_link_libraries(yolact_ncs2_openvino_2021_4 ${LINKLIB})