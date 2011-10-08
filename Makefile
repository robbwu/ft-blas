include ./make.inc


all: mm_opt_rank ft_blaslib tests
mm_opt_rank:
	

ft_blaslib:
	( cd src; make )

tests:
	( cd test; make )

clean:
		rm src/*.o lib/* test/*test

