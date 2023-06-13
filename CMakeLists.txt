cmake_minimum_required(VERSION 3.5)

project(pcd_nricp)

# if want to debug with vscode && gdb
set(CMAKE_BUILD_TYPE Debug)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(Trimesh2_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/trimesh2/include/)
set(Trimesh2_LIB_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/trimesh2/lib.Linux64/libtrimesh.a)

# set cmake_modules path
# set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message ("OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Eigen3
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message ("Eigen3 found")
else()
    message (FATAL_ERROR "Cannot find Eigen3")
endif()

# SuiteSparse
# find_package(SuiteSparse REQUIRED)
# if(SuiteSparse_FOUND)
#     message ("SuiteSparse found")
# else()
#     message (FATAL_ERROR "Cannot find SuiteSparse")
# endif()



find_package(SuiteSparse QUIET NO_MODULE)  # 1st: Try to locate the *config.cmake file.
if(NOT SuiteSparse_FOUND)
        #set(SuiteSparse_VERBOSE ON)
        find_package(SuiteSparse REQUIRED) # 2nd: Use FindSuiteSparse.cmake module
else()
        message(STATUS "Find SuiteSparse : include(${USE_SuiteSparse})")
        include(${USE_SuiteSparse})
endif()
MESSAGE(STATUS "SuiteSparse_LIBS: ${SuiteSparse_LIBRARIES}")

# nricp executable
add_executable(nricp main.cpp)

target_include_directories(nricp 
    PUBLIC
    ${Trimesh2_INCLUDE_DIRS}
    # PUBLIC
    # ${EIGEN3_INCLUDE_DIR}
    PUBLIC
    ${SuiteSparse_INCLUDE_DIRS}
    PUBLIC 
    ${Boost_INCLUDE_DIRS})

target_link_libraries(nricp
    Eigen3::Eigen
    ${Trimesh2_LIB_DIRS} 
    # Eigen/CholmodSupport needs SuiteSparse_LIBRARIES
    ${SuiteSparse_LIBRARIES} 
    # flann bug:
    # won't automatically link liblz4 
    /usr/lib/x86_64-linux-gnu/liblz4.so)

# def_trans executable
add_executable(dfm_transf deformation_transfer.cpp)

target_include_directories(dfm_transf
    PUBLIC
    ${Trimesh2_INCLUDE_DIRS}
    PUBLIC
    ${SuiteSparse_INCLUDE_DIRS})

target_link_libraries(dfm_transf
    Eigen3::Eigen
    ${Trimesh2_LIB_DIRS}
    # Eigen/CholmodSupport needs SuiteSparse_LIBRARIES
    ${SuiteSparse_LIBRARIES} 
    # flann bug:
    # won't automatically link liblz4 
    /usr/lib/x86_64-linux-gnu/liblz4.so) 