# 获取目录下的子目录名
# subdirnamelist 子目录名称列表，输出变量，调用处定义，填充时不能加${}
# targetdir      目标路径，为全路径
macro(SUBDIRNAMELIST_MACRO all_subdirectories targetdir)
    # message(STATUS "macro all_subdirectories = ${all_subdirectories}")
    # message(STATUS "macro targetdir = ${targetdir}")
    file(GLOB children ${targetdir}/*) # 获取目标路径下的内容，深度为1
    # message(STATUS "macro children = ${children}")
    set(dirlist "")
    foreach(child ${children})
        file(RELATIVE_PATH child_name ${targetdir} ${child}) # 通过相对路径计算获取名称
        if(IS_DIRECTORY ${targetdir}/${child_name})
            # message(STATUS "Child Path include  ${child_name}")
            list(APPEND dirlist ${child_name})
        endif()
    endforeach()
    list(APPEND ${all_subdirectories} ${dirlist})
    # message(STATUS "macro dirlist = ${dirlist}")
    # message(STATUS "macro subdirnamelist = ${subdirnamelist}")
endmacro()
 
# 获取目录下是否有 CMakeLists
# hascmakelist 是否含有CMakeLists.txt，输出变量，调用处定义，填充时不能加${}
# targetdir    目标路径，为全路径


macro(CHECK_DIR_HAS_CMAKELIST hascmakelist targetdir)
    # message(STATUS "macro check has cmakelist targetdir = ${targetdir}")
    set(${hascmakelist} FALSE)
    get_filename_component(abs_path ${targetdir} REALPATH BASE_DIR ${CMAKE_SOURCE_DIR})
    # message(STATUS "<==> ${abs_path} is absolute path!")
    if(IS_DIRECTORY ${abs_path})
        if(EXISTS ${abs_path}/CMakeLists.txt)
            set(${hascmakelist} TRUE)
            # message(STATUS "macro check has cmakelist is dir, targetdir = ${targetdir}")
        endif()
    endif()
endmacro()

# 查找根目录下包含CMakeLists.txt文件的子目录
macro(ALL_HEAD_PATH sub_module_name rootpath)
    # message(STATUS "== ${sub_module_name} ${rootpath}")
    SUBDIRNAMELIST_MACRO(all_subdirectories ${rootpath})
    # message(STATUS "macro add_subdir all_subdirectories = ${all_subdirectories}  ${rootpath}")
    foreach(subdir ${all_subdirectories})
        # message(STATUS "macro add_subdir subdir = ${subdir}")
        CHECK_DIR_HAS_CMAKELIST(hascmakelist ${subdir})
        # message(STATUS "subdirectory ${subdir} = ${hascmakelisttemp}")
        if(${hascmakelist})
            # message(STATUS "macro add_subdir go to add_subdir = ${subdir}")
            list(APPEND sub_module_name ${subdir})
        endif()
    endforeach()
endmacro()

macro(INCLUDE_HEAD_ALL all_paths sub_module_name)
    set(head_paths "")
    # message(STATUS " <+++++> ${sub_module_name}")
    foreach(subpath ${sub_module_name})
        file(GLOB children ${subpath}/*)
        foreach(filename ${children})
            if(IS_DIRECTORY ${filename})
                # message(STATUS "PATH ${filename}")
                file(GLOB head_filename ${filename}/*.h)
                # message(STATUS "**** ${head_filename}")
                foreach(heads ${head_filename})
                    if(EXISTS ${heads})
                        # message(STATUS "==> ${heads}")
                        list(APPEND all_paths ${filename})
                    endif()
                endforeach()
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES all_paths)
endmacro()

# 获取workspace下的子模块(包含有CMakeLists.txt的目录)
macro(PROJECT_SUBMODULES submodule workspace)
  ALL_HEAD_PATH(sub_module_name ${workspace})
  list(REMOVE_DUPLICATES sub_module_name)
  foreach(module ${sub_module_name})
      get_filename_component(abs_path ${module} REALPATH BASE_DIR ${CMAKE_SOURCE_DIR})
      include_directories(${abs_path}/include)
  endforeach()
endmacro()

# 通过文件名称创建测试用例你可以CREATE_TEST_CASE(arg0 arg1) 或者 CREATE_TEST_CASE(arg0)
function(CREATE_TEST_CASE)
if(${ARGC} EQUAL 1)
    set(test_name ${ARGV0})
    set(tests_path "${CMAKE_SOURCE_DIR}/test")
    message(STATUS "Current params only one,so test path use default: ${tests_path}")
elseif(${ARGC} EQUAL 2)
    set(test_name ${ARGV0})
    set(tests_path "${ARGV1}")
else()
   message(FATAL_ERROR "Function param nums must be 1 or 2")
endif()
message("Test case:${test_name} test path:${tests_path}")
if(NOT IS_DIRECTORY ${tests_path})
    message(FATAL_ERROR "${tests_path} not exist!")
else()
    message(STATUS "==> ${tests_path}/src/${test_name}.cu")
    if(EXISTS ${tests_path}/src/${test_name}.cu)
        add_executable(${test_name} ${tests_path}/src/test_vector_add.cu)
        target_link_libraries(${test_name} GTest::gtest GTest::gtest_main pthread)
        add_test(NAME ${test_name} COMMAND ${test_name})
    elseif(EXISTS ${tests_path}/src/${test_name}.cc)
        add_executable(${test_name} ${tests_path}/src/test_vector_add.cu)
        target_link_libraries(${test_name} GTest::gtest GTest::gtest_main pthread)
        add_test(NAME ${test_name} COMMAND ${test_name})
    else() 
        message(FATAL_ERROR "Can't find ${test_name} source")
    endif()
endif()
endfunction()

# add test
function(FILENAME test_paths)
    if(${ARGC} EQUAL 1)
        file(GLOB CUDA_SOURCE ${CMAKE_SOURCE_DIR}/test/src/*.cu)
        # message(STATUS "MATCHED filename:${CMAKE_SOURCE_DIR}/test ${CUDA_SOURCE}")
        foreach(filename ${CUDA_SOURCE})
            if(EXISTS ${filename})
                STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" test_name ${filename})
                message(STATUS "MATCHED filename:${test_name}")
                CREATE_TEST_CASE(${test_name} ${CMAKE_SOURCE_DIR}/test)
            else()
                message(FATAL_ERROR "${test_name} not exists!")
            endif()
        endforeach()
    else()
        message(STATUS "FILENAME params nums must be 2")
    endif()   
    # CREATE_TEST_CASE(${filename} ${CMAKE_SOURCE_DIR}/test)

endfunction()