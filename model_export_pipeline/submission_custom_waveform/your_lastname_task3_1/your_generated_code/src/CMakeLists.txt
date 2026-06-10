cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 20)

set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/managed_components/esp-tflite-micro"
    "${CMAKE_CURRENT_LIST_DIR}/managed_components/espressif__esp-dsp"
    "${CMAKE_CURRENT_LIST_DIR}/managed_components/espressif__esp-nn")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

add_compile_definitions(TCB_SPAN_NAMESPACE_NAME=std)  # remove if switching to C++20
project(model_benchmark)
