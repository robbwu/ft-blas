test: ft_blas.c Makefile
		#cc -Wall -framework Accelerate -lgsl ft_blas.c -o test
		#cc  -framework Accelerate -lgsl ft_blas.c -o test -O2
		gcc -lgsl  -lpthread -O2   -o test ft_blas.c /usr/lib/atlas/libblas.a
		#./test

clean:
		rm test stock *.o

