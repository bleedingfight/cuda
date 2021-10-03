enable_testing()
find_package(GTest REQUIRED)
set(sub_module "")
# 包含项目下的所有头文件
PROJECT_SUBMODULES(submodule ${CMAKE_SOURCE_DIR})
FILENAME(${CMAKE_SOURCE_DIR}/test)
# CREATE_TEST_CASE(test_vector_add ${CMAKE_SOURCE_DIR}/test)
