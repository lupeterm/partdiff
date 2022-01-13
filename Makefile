CC=mpicc
LDLIBS=-lm -fopenmp
FLAGS= -std=c11 -Wall -Wextra -Wpedantic -O3 -g
all: clean partdiff
partdiff: partdiff.c
clean: 
	$(RM) partdiff 