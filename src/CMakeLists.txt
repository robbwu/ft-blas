add_executable(mmbs mmbs.c)

target_link_libraries(mmbs ${LIBS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mmbs.h
    DEPENDS mmbs
    COMMAND mmbs > ${CMAKE_CURRENT_BINARY_DIR}/mmbs.h
)

add_library(ft_blas dgemm.c mmbs.h)
