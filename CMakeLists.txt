cmake_minimum_required(VERSION 2.8)
project(ft_blas)

enable_language(Fortran)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

# PLEASE modify this lines to provide link libraries
set(BLAS_LIBRARIES /usr/lib/atlas/libblas.a)
set(GSL_LIBRARIES gsl)
# END
#

find_package(Threads REQUIRED)

include_directories(${GSL_INCLUDE_DIR} )
add_definitions(-O2 -Wall )
set(LIBS ${LIBS} ${GSL_LIBRARIES}   ${BLAS_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} m)
foreach(f ${LIBS})
    message(${f})
endforeach(f)

include_directories("${PROJECT_SOURCE_DIR}/src")
add_subdirectory(src)
add_subdirectory(test)


