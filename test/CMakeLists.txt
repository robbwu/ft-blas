
set(LIBS ${LIBS} ft_blas) 
link_libraries(${LIBS})
add_executable(
    dgemm_test
    dgemm_test.c
)

