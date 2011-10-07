include ./make.inc


all: ft_blaslib tests

ft_blaslib:
	( cd src; make )

tests:
	( cd test; make )

clean:
		rm src/*.o lib/* test/*test

