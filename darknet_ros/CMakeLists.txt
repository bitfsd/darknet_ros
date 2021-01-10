cmake_minimum_required(VERSION 2.8.12)
project(darknet_ros)

# Set c++11 cmake flags
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable -Wfatal-errors -fPIC ${CMAKE_C_FLAGS}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/" ${CMAKE_MODULE_PATH})

# Define path of darknet folder here.
find_path(DARKNET_PATH
  NAMES "README.md"
  HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../darknet/")
message(STATUS "Darknet path dir = ${DARKNET_PATH}")
add_definitions(-DDARKNET_FILE_PATH="${DARKNET_PATH}")

# Find CUDA
find_package(CUDA QUIET)
if (CUDA_FOUND)
  find_package(CUDA REQUIRED)
  find_package(CUDNN)
  message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
  message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
  )
  add_definitions(-DGPU)
  add_definitions(-DCUDNN)
  add_definitions(-DCUDNN_HALF)
  add_definitions(-DOPENCV)
else()
  list(APPEND LIBRARIES "m")
endif()

# Find X11
message ( STATUS "Searching for X11..." )
find_package ( X11 REQUIRED )
if ( X11_FOUND )
  include_directories ( ${X11_INCLUDE_DIR} )
  link_libraries ( ${X11_LIBRARIES} )
  message ( STATUS " X11_INCLUDE_DIR: " ${X11_INCLUDE_DIR} )
  message ( STATUS " X11_LIBRARIES: " ${X11_LIBRARIES} )
endif ( X11_FOUND )

# Find rquired packeges
find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(catkin REQUIRED
  COMPONENTS
    cv_bridge
    roscpp
    rospy
    std_msgs
    actionlib
    darknet_ros_msgs
    image_transport
)

# Enable OPENCV in darknet
add_definitions(-DOPENCV)
add_definitions(-O4 -g)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}_lib
  CATKIN_DEPENDS
    cv_bridge
    roscpp
    actionlib
    rospy
    std_msgs
    darknet_ros_msgs
    image_transport
  DEPENDS
    Boost
)


include_directories(
  ${DARKNET_PATH}/src
  ${DARKNET_PATH}/include
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${DARKNET_PATH}/3rdparty/stb/include
  ${DARKNET_PATH}/3rdparty/pthreads
  ${CUDNN_INCLUDE_DIR}
)



if (CUDA_FOUND)

  link_directories(
    /usr/local/cuda/lib64
  )

  cuda_add_library(${PROJECT_NAME}_lib
    src/YoloObjectDetector.cpp
    src/image_interface.c
    ${DARKNET_PATH}/src/activation_layer.c        ${DARKNET_PATH}/src/activations.c    
    ${DARKNET_PATH}/src/art.c                     ${DARKNET_PATH}/src/avgpool_layer.c           
    ${DARKNET_PATH}/src/batchnorm_layer.c         ${DARKNET_PATH}/src/blas.c                    
    ${DARKNET_PATH}/src/box.c                     ${DARKNET_PATH}/src/captcha.c                     
    ${DARKNET_PATH}/src/cifar.c                   ${DARKNET_PATH}/src/classifier.c         
    ${DARKNET_PATH}/src/coco.c                    ${DARKNET_PATH}/src/col2im.c                  
    ${DARKNET_PATH}/src/compare.c                 ${DARKNET_PATH}/src/connected_layer.c
    ${DARKNET_PATH}/src/conv_lstm_layer.c         ${DARKNET_PATH}/src/convolutional_layer.c
    ${DARKNET_PATH}/src/cost_layer.c              ${DARKNET_PATH}/src/cpu_gemm.c              
    ${DARKNET_PATH}/src/crnn_layer.c              ${DARKNET_PATH}/src/crop_layer.c              
    ${DARKNET_PATH}/src/dark_cuda.c               ${DARKNET_PATH}/src/darknet.c                
    ${DARKNET_PATH}/src/data.c                    ${DARKNET_PATH}/src/deconvolutional_layer.c   
    ${DARKNET_PATH}/src/demo.c                    ${DARKNET_PATH}/src/detection_layer.c         
    ${DARKNET_PATH}/src/detector.c                ${DARKNET_PATH}/src/dice.c           
    ${DARKNET_PATH}/src/dropout_layer.c           ${DARKNET_PATH}/src/gaussian_yolo_layer.c
    ${DARKNET_PATH}/src/gemm.c                    ${DARKNET_PATH}/src/getopt.c                    
    ${DARKNET_PATH}/src/gettimeofday.c            ${DARKNET_PATH}/src/go.c
    ${DARKNET_PATH}/src/gru_layer.c               ${DARKNET_PATH}/src/im2col.c
    ${DARKNET_PATH}/src/image.c                   ${DARKNET_PATH}/src/layer.c
    ${DARKNET_PATH}/src/list.c                    ${DARKNET_PATH}/src/local_layer.c
    ${DARKNET_PATH}/src/lstm_layer.c              ${DARKNET_PATH}/src/matrix.c
    ${DARKNET_PATH}/src/maxpool_layer.c           ${DARKNET_PATH}/src/network.c
    ${DARKNET_PATH}/src/nightmare.c               ${DARKNET_PATH}/src/normalization_layer.c
    ${DARKNET_PATH}/src/option_list.c             ${DARKNET_PATH}/src/parser.c
    ${DARKNET_PATH}/src/region_layer.c            ${DARKNET_PATH}/src/reorg_layer.c
    ${DARKNET_PATH}/src/reorg_old_layer.c         ${DARKNET_PATH}/src/rnn.c
    ${DARKNET_PATH}/src/rnn_layer.c               ${DARKNET_PATH}/src/rnn_vid.c
    ${DARKNET_PATH}/src/route_layer.c             ${DARKNET_PATH}/src/sam_layer.c
    ${DARKNET_PATH}/src/scale_channels_layer.c    ${DARKNET_PATH}/src/shortcut_layer.c
    ${DARKNET_PATH}/src/softmax_layer.c           ${DARKNET_PATH}/src/super.c
    ${DARKNET_PATH}/src/swag.c                    ${DARKNET_PATH}/src/tag.c
    ${DARKNET_PATH}/src/tree.c                    ${DARKNET_PATH}/src/upsample_layer.c
    ${DARKNET_PATH}/src/utils.c                   ${DARKNET_PATH}/src/voxel.c
    ${DARKNET_PATH}/src/writing.c                 ${DARKNET_PATH}/src/yolo.c
    ${DARKNET_PATH}/src/yolo_layer.c

    ${DARKNET_PATH}/src/http_stream.cpp           ${DARKNET_PATH}/src/image_opencv.cpp
#    ${DARKNET_PATH}/src/yolo_v2_class.cpp         ${DARKNET_PATH}/src/yolo_console_dll.cpp


    ${DARKNET_PATH}/src/activation_kernels.cu     ${DARKNET_PATH}/src/avgpool_layer_kernels.cu
    ${DARKNET_PATH}/src/blas_kernels.cu           ${DARKNET_PATH}/src/col2im_kernels.cu         
    ${DARKNET_PATH}/src/convolutional_kernels.cu  ${DARKNET_PATH}/src/crop_layer_kernels.cu
    ${DARKNET_PATH}/src/deconvolutional_kernels.cu ${DARKNET_PATH}/src/dropout_layer_kernels.cu
    ${DARKNET_PATH}/src/im2col_kernels.cu          ${DARKNET_PATH}/src/maxpool_layer_kernels.cu
    ${DARKNET_PATH}/src/network_kernels.cu

  )

  target_link_libraries(${PROJECT_NAME}_lib
    m
    pthread
    stdc++
    cuda
    cudart
    cublas
    curand
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDNN_LIBRARIES}
    
  )
  
  #target_link_libraries(${PROJECT_NAME}_lib PRIVATE CuDNN::CuDNN)
  #target_link_libraries(${PROJECT_NAME}_lib PRIVATE CuDNN::CuDNN)
  target_compile_definitions(${PROJECT_NAME}_lib PRIVATE -DCUDNN)
  target_compile_definitions(${PROJECT_NAME}_lib PUBLIC -DCUDNN)
  
  target_compile_definitions(${PROJECT_NAME}_lib PRIVATE -DCUDNN_HALF)
  target_compile_definitions(${PROJECT_NAME}_lib PUBLIC -DCUDNN_HALF)
  
  
  

  cuda_add_executable(${PROJECT_NAME}
    src/yolo_object_detector_node.cpp
  )

else()

  add_library(${PROJECT_NAME}_lib
    src/YoloObjectDetector.cpp
    src/image_interface.c

    ${DARKNET_PATH}/src/activation_layer.c        ${DARKNET_PATH}/src/activations.c    
    ${DARKNET_PATH}/src/art.c                     ${DARKNET_PATH}/src/avgpool_layer.c           
    ${DARKNET_PATH}/src/batchnorm_layer.c         ${DARKNET_PATH}/src/blas.c                    
    ${DARKNET_PATH}/src/box.c                     ${DARKNET_PATH}/src/captcha.c                     
    ${DARKNET_PATH}/src/cifar.c                   ${DARKNET_PATH}/src/classifier.c         
    ${DARKNET_PATH}/src/coco.c                    ${DARKNET_PATH}/src/col2im.c                  
    ${DARKNET_PATH}/src/compare.c                 ${DARKNET_PATH}/src/connected_layer.c
    ${DARKNET_PATH}/src/conv_lstm_layer.c         ${DARKNET_PATH}/src/convolutional_layer.c
    ${DARKNET_PATH}/src/cost_layer.c              ${DARKNET_PATH}/src/cpu_gemm.c              
    ${DARKNET_PATH}/src/crnn_layer.c              ${DARKNET_PATH}/src/crop_layer.c              
    ${DARKNET_PATH}/src/dark_cuda.c               ${DARKNET_PATH}/src/darknet.c                
    ${DARKNET_PATH}/src/data.c                    ${DARKNET_PATH}/src/deconvolutional_layer.c   
    ${DARKNET_PATH}/src/demo.c                    ${DARKNET_PATH}/src/detection_layer.c         
    ${DARKNET_PATH}/src/detector.c                ${DARKNET_PATH}/src/dice.c           
    ${DARKNET_PATH}/src/dropout_layer.c           ${DARKNET_PATH}/src/gaussian_yolo_layer.c
    ${DARKNET_PATH}/src/gemm.c                    ${DARKNET_PATH}/src/getopt.c                    
    ${DARKNET_PATH}/src/gettimeofday.c            ${DARKNET_PATH}/src/go.c
    ${DARKNET_PATH}/src/gru_layer.c               ${DARKNET_PATH}/src/im2col.c
    ${DARKNET_PATH}/src/image.c                   ${DARKNET_PATH}/src/layer.c
    ${DARKNET_PATH}/src/list.c                    ${DARKNET_PATH}/src/local_layer.c
    ${DARKNET_PATH}/src/lstm_layer.c              ${DARKNET_PATH}/src/matrix.c
    ${DARKNET_PATH}/src/maxpool_layer.c           ${DARKNET_PATH}/src/network.c
    ${DARKNET_PATH}/src/nightmare.c               ${DARKNET_PATH}/src/normalization_layer.c
    ${DARKNET_PATH}/src/option_list.c             ${DARKNET_PATH}/src/parser.c
    ${DARKNET_PATH}/src/region_layer.c            ${DARKNET_PATH}/src/reorg_layer.c
    ${DARKNET_PATH}/src/reorg_old_layer.c         ${DARKNET_PATH}/src/rnn.c
    ${DARKNET_PATH}/src/rnn_layer.c               ${DARKNET_PATH}/src/rnn_vid.c
    ${DARKNET_PATH}/src/route_layer.c             ${DARKNET_PATH}/src/sam_layer.c
    ${DARKNET_PATH}/src/scale_channels_layer.c    ${DARKNET_PATH}/src/shortcut_layer.c
    ${DARKNET_PATH}/src/softmax_layer.c           ${DARKNET_PATH}/src/super.c
    ${DARKNET_PATH}/src/swag.c                    ${DARKNET_PATH}/src/tag.c
    ${DARKNET_PATH}/src/tree.c                    ${DARKNET_PATH}/src/upsample_layer.c
    ${DARKNET_PATH}/src/utils.c                   ${DARKNET_PATH}/src/voxel.c
    ${DARKNET_PATH}/src/writing.c                 ${DARKNET_PATH}/src/yolo.c
    ${DARKNET_PATH}/src/yolo_layer.c

    ${DARKNET_PATH}/src/http_stream.cpp           ${DARKNET_PATH}/src/image_opencv.cpp
  #  ${DARKNET_PATH}/src/yolo_v2_class.cpp         ${DARKNET_PATH}/src/yolo_console_dll.cpp

  )

  target_link_libraries(${PROJECT_NAME}_lib
    m
    pthread
    stdc++
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
  )

  add_executable(${PROJECT_NAME}
    src/yolo_object_detector_node.cpp
  )

endif()

target_link_libraries(${PROJECT_NAME}
  ${PROJECT_NAME}_lib
)

add_dependencies(${PROJECT_NAME}_lib
  darknet_ros_msgs_generate_messages_cpp
)

install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
