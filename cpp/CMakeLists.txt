cmake_minimum_required(VERSION 2.8.12)
project( qr_code )
find_package( OpenCV REQUIRED )
find_package( dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})



set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

add_definitions(-std=c++11 -DGAZR_VERSION=${VERSION})

find_package(dlib REQUIRED)
# Adhere to GNU filesystem layout conventions


macro(add_example name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib )
endmacro()


# if an example requires GUI, call this macro to check DLIB_NO_GUI_SUPPORT to include or exclude
macro(add_gui_example name)
   if (DLIB_NO_GUI_SUPPORT)
      message("No GUI support, so we won't build the ${name} example.")
   else()
      add_example(${name})
   endif()
endmacro()

add_executable( qr_image image.cpp )
add_executable( qr_video video.cpp )
#add_executable( video_test video_test.cpp )
add_executable(video_dlib_correlation video_dlib_correlation.cpp)
target_link_libraries( qr_image ${OpenCV_LIBS} )
target_link_libraries( qr_video ${OpenCV_LIBS} )
#target_link_libraries( video_test ${OpenCV_LIBS} )
find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      
      target_link_libraries( video_dlib_correlation ${dlib_LIBRARIES} ${OpenCV_LIBS}  )
   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()

