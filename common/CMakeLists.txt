include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB CUDA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
file(GLOB CPP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(common_path ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)

add_library(utils SHARED ${CUDA_SRC})
add_Library(tools SHARED ${CPP_SRC})