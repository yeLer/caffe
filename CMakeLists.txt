# 最小支持版本设定，一般工程都是从cmake_minimum_required开始的，表示系统cmake的最低版本号，如果版本号
# 比这个2.8.7小，那么就无法cmake 通过
cmake_minimum_required(VERSION 2.8.7)
# 这个cmake_policy没什么实际作用的，只是做版本向后兼容的提示
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
# caffe工程名，每一个项目都会有一个工程名字，用project（）来定义
project(Caffe C CXX)

# ---[ Caffe version  Caffe version SET这个命令，就是设置变量的意思，简单用法 SET(<VAR> value)
# caffe版本号
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")
# add_definitions 定义cmake define定义，类似于 Ｃ++ 里的#define　CAFFE_VERSION
# 其实就是定义了个宏CAFFE_VERSION, 定义这个变量，后面会用到if(CAFFE_VERSION) 来判断这个变量是否被定义
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# ---[ Using cmake scripts and modules 
# list命令是定义字符串数组，用于保存是一个或者多个变量的
# CMAKE_MODULE_PATH　这个变量是cmake内部预定义的环境变量，表示包含cmake 模块的路径，
# cmake 模块是一个cmake 脚本，用于查询安装在系统内或者本地的库　里面都是一些Findxxx.cmake，
# 后面会出现用find_pakage()调用cmake模块；一般cmake 模块都会返回(这只是个约定)
# xx_FOUND 用于判断是否找到该xx库,xx_INCLUDE_DIR:include路径,xx__LIBRARY：libxxx.so等路径与名字
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# #include()用于包含cmake脚本文件，类似于调用了该段宏展开
include(ExternalProject) # ExternalProject 外部工程，还没有编译出库来的，需要现在下载编译，其脚本在External目录里
include(GNUInstallDirs)  # GNUInstallDirs 这个变量跟系统的ＧＮＵ界面有关系

# cmake/Utils.cmake : 这个脚本里定义了很多Cmake 函数function,每个函数或者每一个宏都有英文对应的解释
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
# 这里全部调用cmake/Utils.cmake　里自定义的函数caffe_option（），其实里头就是调用了cmake命令option()
# 为工程添加可选开关，有默认值OFF 或者 ON 表示该定义开关,在工程编译前cmake 命令可以　cmake -DCPU_ONLY=ON 打开
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
# OpenMP是一种用于共享内存并行系统的多线程程序设计方案,支持的编程语言包括C、C++和Fortran
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)

# This code is taken from https://github.com/sh1r0/caffe-android-lib
caffe_option(USE_HDF5 "Build with hdf5" ON)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  # CMAKE_CXX_FLAGS 这个变量也是cmake 预定义的变量，用于设定C++编译器的属性
  # -fPIC则表明使用地址无关代码。PIC：Position Independent Code
  # -Wall选项意思是编译后显示所有警告
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

# 调用自定义函数 :这个函数里最关键是定义了Caffe_LINK这个变量，用于收集caffe　相关的库
caffe_set_caffe_link()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
# #configure_file(<input> <output>[COPYONLY] [ESCAPE_QUOTES] [@ONLY])
# 该文件设置配置文件信息，配置的方式是：以<input>(必须是个文件)中定义的变量替换<output>中的以${VAR}格式或@VAR@格式引用的任意变量，
# 如同它们的值是由CMake确定的一样。 如果一个变量还未定义，它会被替换为空。
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
# Caffe_INCLUDE_DIR :定义工程的include路径变量
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
# Caffe_SRC_DIR :定义工程中的源码路径的变量
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
# include_directories:该命令为工程添加include路径，就是头文件的查询路径(就是告诉编译器头文件在哪)
include_directories(${PROJECT_BINARY_DIR})

# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
  list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
  list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)
  add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ uninstall target
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake)

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()

