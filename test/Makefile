include ../make.inc

all: dgemm_test


dgemm_test: dgemm_test.c
	$(CC) $(OPTS) dgemm_test.c ../src/dgemm.o   -lm $(GSLLIB) $(BLASLIB) $(PTHREADLIB)  -o dgemm_test
