include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB CUDA_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.cu)

add_library(maths SHARED ${CUDA_SRC})
message("Current Path:${CMAKE_CURRENT_LIST_DIR} line: ${CMAKE_CURRENT_FUNCTION_LIST_LINE} ${CUDA_SRC}")
