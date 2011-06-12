test: ft_blas.c Makefile
		#cc -Wall -framework Accelerate -lgsl ft_blas.c -o test
		cc  -framework Accelerate -lgsl ft_blas.c -o test -O2
		#./test

clean:
		rm test stock *.o

