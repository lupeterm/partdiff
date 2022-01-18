#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>
#include <mpi.h>

/* ************* */
/* Some defines. */
/* ************* */
#define M_PI 3.141592653589793
#define TWO_PI_SQUARE (2 *M_PI*M_PI)
#define MAX_INTERLINES 12000
#define MAX_ITERATION 200000
#define MAX_THREADS 1024
#define METH_GAUSS_SEIDEL 1
#define METH_JACOBI 2
#define FUNC_F0 1
#define FUNC_FPISIN 2
#define TERM_PREC 1
#define TERM_ITER 2

struct options
{
	uint64_t number;		 /* Number of threads                              */
	uint64_t method;		 /* Gauss Seidel or Jacobi method of iteration     */
	uint64_t interlines;	 /* matrix size = interlines*8+9                   */
	uint64_t inf_func;		 /* inference function                             */
	uint64_t termination;	 /* termination condition                          */
	uint64_t term_iteration; /* terminate if iteration number reached          */
	double term_precision;	 /* terminate if precision reached                 */
};

struct calculation_arguments
{
	uint64_t N;			   /* number of spaces between lines (lines=N+1)     */
	uint64_t num_matrices; /* number of matrices                             */
	double h;			   /* length of a space between two lines            */
	// double*** Matrix;	   /* index matrix used for addressing M             */
	double* Matrix;
	uint64_t chunk_size_general;
	uint64_t chunk_size_process;
	uint64_t i_start;
	uint64_t i_end;
	int rank;
	int num_processes;
};

struct calculation_results
{
	uint64_t m;
	uint64_t stat_iteration; /* number of current iteration                    */
	double stat_precision;	 /* actual precision of all slaves in iteration    */
};

enum message_types
{
	MSG_INTERMEDIATE_CALCULATED_ROW
};

/* ************************************************************************ */
/* Global variables                                                         */
/* ************************************************************************ */

/* time measurement variables */
struct timeval start_time; /* time when program started                      */
struct timeval comp_time;  /* time when calculation completed                */

/* ************************************************************************ */
/* initVariables: Initializes some global variables                         */
/* ************************************************************************ */
static void
initVariables(struct calculation_arguments *arguments, struct calculation_results *results, struct options const *options)
{
	arguments->N = (options->interlines * 8) + 9 - 1;
	arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
	arguments->h = 1.0 / arguments->N;

	arguments->chunk_size_general = (arguments->N - 1) / arguments->num_processes;

	/* calculate range of rows the current thread has to do */
	if (arguments->rank + 1 == arguments->num_processes) /* last thread? then do the missing rows for uneven numbers */
		arguments->chunk_size_process = (arguments->N - 1) - ((arguments->num_processes - 1) * arguments->chunk_size_general);
	else
		arguments->chunk_size_process = arguments->chunk_size_general;

	/*calculate start and endpoint for thread*/
	arguments->i_start = arguments->rank * arguments->chunk_size_general + 1 /* because first row is not written to */;
	arguments->i_end = ((arguments->rank + 1) * arguments->chunk_size_general) + 1 + (arguments->chunk_size_process - arguments->chunk_size_general);

	printf("\nN = %ld\n(general) chunk_size = %ld\n", arguments->N, arguments->chunk_size_general);
	printf(
		"(thread-specific) chunk_size = %ld\n\ti_start = %ld\n\ti_end = %ld\n\n",
		arguments->chunk_size_process, arguments->i_start, arguments->i_end);
	MPI_Barrier(MPI_COMM_WORLD);
	results->m = 0;
	results->stat_iteration = 0;
	results->stat_precision = 0;
}

/* ************************************************************************ */
/* freeMatrices: frees memory for matrices                                  */
/* ************************************************************************ */
static void
freeMatrices(struct calculation_arguments *arguments)
{
	// uint64_t i;

	// for (i = 0; i < arguments->num_matrices; i++)
	// {
	// 	free(arguments->Matrix[i]);
	// }

	free(arguments->Matrix);
}

static void
usage(char *name)
{
	printf("Usage: %s [num] [method] [lines] [func] [term] [prec/iter]\n", name);
	printf("\n");
	printf("  - num:       number of threads (1 .. %d)\n", MAX_THREADS);
	printf("  - method:    calculation method (1 .. 2)\n");
	printf("                 %1d: Gauß-Seidel\n", METH_GAUSS_SEIDEL);
	printf("                 %1d: Jacobi\n", METH_JACOBI);
	printf("  - lines:     number of interlines (0 .. %d)\n", MAX_INTERLINES);
	printf("                 matrixsize = (interlines * 8) + 9\n");
	printf("  - func:      interference function (1 .. 2)\n");
	printf("                 %1d: f(x,y) = 0\n", FUNC_F0);
	printf("                 %1d: f(x,y) = 2 * pi^2 * sin(pi * x) * sin(pi * y)\n", FUNC_FPISIN);
	printf("  - term:      termination condition ( 1.. 2)\n");
	printf("                 %1d: sufficient precision\n", TERM_PREC);
	printf("                 %1d: number of iterations\n", TERM_ITER);
	printf("  - prec/iter: depending on term:\n");
	printf("                 precision:  1e-4 .. 1e-20\n");
	printf("                 iterations:    1 .. %d\n", MAX_ITERATION);
	printf("\n");
	printf("Example: %s 1 2 100 1 2 100 \n", name);
}

static void
askParams(struct options *options, int argc, char **argv)
{
	int ret;

	if (argc < 7 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "-?") == 0)
	{
		usage(argv[0]);
		exit(0);
	}

	ret = sscanf(argv[1], "%" SCNu64, &(options->number));

	if (ret != 1 || !(options->number >= 1 && options->number <= MAX_THREADS))
	{
		usage(argv[0]);
		exit(1);
	}

	ret = sscanf(argv[2], "%" SCNu64, &(options->method));

	if (ret != 1 || !(options->method == METH_GAUSS_SEIDEL || options->method == METH_JACOBI))
	{
		usage(argv[0]);
		exit(1);
	}

	ret = sscanf(argv[3], "%" SCNu64, &(options->interlines));

	if (ret != 1 || !(options->interlines <= MAX_INTERLINES))
	{
		usage(argv[0]);
		exit(1);
	}

	ret = sscanf(argv[4], "%" SCNu64, &(options->inf_func));

	if (ret != 1 || !(options->inf_func == FUNC_F0 || options->inf_func == FUNC_FPISIN))
	{
		usage(argv[0]);
		exit(1);
	}

	ret = sscanf(argv[5], "%" SCNu64, &(options->termination));

	if (ret != 1 || !(options->termination == TERM_PREC || options->termination == TERM_ITER))
	{
		usage(argv[0]);
		exit(1);
	}

	if (options->termination == TERM_PREC)
	{
		ret = sscanf(argv[6], "%lf", &(options->term_precision));

		options->term_iteration = MAX_ITERATION;

		if (ret != 1 || !(options->term_precision >= 1e-20 && options->term_precision <= 1e-4))
		{
			usage(argv[0]);
			exit(1);
		}
	}
	else
	{
		ret = sscanf(argv[6], "%" SCNu64, &(options->term_iteration));

		options->term_precision = 0;

		if (ret != 1 || !(options->term_iteration >= 1 && options->term_iteration <= MAX_ITERATION))
		{
			usage(argv[0]);
			exit(1);
		}
	}
}

/* ************************************************************************ */
/* allocateMemory ()                                                        */
/* allocates memory and quits if there was a memory allocation problem      */
/* ************************************************************************ */
static void *
allocateMemory(size_t size)
{
	void *p;

	if ((p = malloc(size)) == NULL)
	{
		printf("Speicherprobleme! (%" PRIu64 " Bytes angefordert)\n", size);
		exit(1);
	}

	return p;
}

/* ************************************************************************ */
/* allocateMatrices: allocates memory for matrices                          */
/* ************************************************************************ */
static void
allocateMatrices(struct calculation_arguments *arguments)
{
	uint64_t const N = arguments->N;
	arguments->Matrix = allocateMemory(arguments->num_matrices * (arguments->chunk_size_process+2) * (N + 1) * sizeof(double));
}
/* ************************************************************************ */
/* initMatrices: Initialize matrix/matrices and some global variables       */
/* ************************************************************************ */
static void
initMatrices(struct calculation_arguments *arguments, struct options const *options)
{
	uint64_t g, i, j; /* local variables for loops */

	uint64_t const N = arguments->N;
	double const h = arguments->h;
	typedef double(*matrix)[arguments->chunk_size_process+2][N + 1];
	matrix Matrix = (matrix)arguments->Matrix;

	/* initialize matrix/matrices with zeros */
	for (g = 0; g < arguments->num_matrices; g++)
	{
		for (i = 0; i < (arguments->chunk_size_process + 2); i++)
		{
			for (j = 0; j <= N; j++)
			{
				Matrix[g][i][j] = 0.0;
			}
		}
	}
	int rank = arguments->rank;
	int size = arguments->num_processes;

	if (options->inf_func == FUNC_F0)
	{
		for (g = 0; g < arguments->num_matrices; g++)
		{
			for (i = 0; i < arguments->chunk_size_process + 2; i++){
				Matrix[g][i][0] = 1.0 - (h * (i + arguments->i_start - 1));
				Matrix[g][i][N] = h * (i + arguments->i_start - 1);
			}
			for (i = 0; i <= N; i++){
				if (rank == 0){			
					// printf("g1%d ded %ld\n", arguments->rank, i);
					Matrix[g][0][i] = 1.0 - (h * i);
				}
				if (rank == size - 1){
					// printf("g%d ded 1%ld\n", arguments->rank, i);
					Matrix[g][arguments->chunk_size_process + 1][i] = h * i;
				}
			}   
			if (rank == size - 1) 
			{
				Matrix[g][arguments->chunk_size_process+1][0] = 0.0;  // ? auch ohne die zeile das gewünschte ergebnis
			}
			if (rank == 0){
				Matrix[g][0][N] = 0.0;
			}                     
		}
	}
}
/* ************************************************************************ */
/* calculate: solves the equation                                           */
/* ************************************************************************ */
static void
calculate_jacobi(struct calculation_arguments const *arguments, struct calculation_results *results, struct options const *options)
{
	if(arguments->rank == 0){
		printf("Distributed Memory calculation!\n");
	}
	uint64_t i, j; /* local variables for loops */
	int m1, m2;			/* used as indices for old and new matrices */
	double star;		/* four times center value minus 4 neigh.b values */
	double residuum;	/* residuum of current iteration */
	double maxresiduum; /* maximum residuum value of a slave in iteration */

	uint64_t const N = arguments->N;
	double const h = arguments->h;

	double pih = 0.0;
	double fpisin = 0.0;

	typedef double(*matrix)[arguments->chunk_size_process+2][N + 1];
	matrix Matrix = (matrix)arguments->Matrix;	

	int term_iteration = options->term_iteration;
	m1 = 0;
	m2 = 1;
	if (options->inf_func == FUNC_FPISIN)
	{
		pih =M_PI* h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

	while (term_iteration > 0)
	{

		maxresiduum = 0;

		// if first process, there will be no previous process to receive from
		if (arguments->rank != 0)
		{
			// receive last self-calculated row from previous process which is very first row of this process
			// MPI_Recv(Matrix_In[0], N + 1, MPI_DOUBLE, arguments->rank - 1, MSG_INTERMEDIATE_CALCULATED_ROW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// send first self-calculated row to previous process
			// MPI_Send(Matrix_Out[1], N + 1, MPI_DOUBLE, arguments->rank - 1, MSG_INTERMEDIATE_CALCULATED_ROW, MPI_COMM_WORLD);

			MPI_Sendrecv(Matrix[m2][1], N + 1, MPI_DOUBLE, arguments->rank - 1, 0, Matrix[m2][0], N + 1, MPI_DOUBLE, arguments->rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		if ((arguments->rank + 1) != arguments->num_processes)
		{
			// send last self-calculated row to next process
			// MPI_Send(Matrix_Out[arguments->chunk_size_process], N + 1, MPI_DOUBLE, arguments->rank + 1, MSG_INTERMEDIATE_CALCULATED_ROW, MPI_COMM_WORLD);

			// receive first self-calculated row from next process which is very last row of this process
			// MPI_Recv(Matrix_In[arguments->chunk_size_process + 1], N + 1, MPI_DOUBLE, arguments->rank + 1, MSG_INTERMEDIATE_CALCULATED_ROW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Sendrecv(Matrix[m2][arguments->chunk_size_process], N + 1, MPI_DOUBLE, arguments->rank + 1, 1, Matrix[m2][arguments->chunk_size_process + 1], N + 1, MPI_DOUBLE, arguments->rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		/* over all rows */
		for (i = 1; i < arguments->chunk_size_process+1; i++)
		{
			double fpisin_i = 0.0;

			if (options->inf_func == FUNC_FPISIN)
			{
				fpisin_i = fpisin * sin(pih * (double)(i + arguments->i_start - 1));
			}

			/* over all columns */
			for (j = 1; j < N; j++)
			{
				star = 0.25 * (Matrix[m2][i - 1][j] + Matrix[m2][i][j - 1] + Matrix[m2][i][j + 1] + Matrix[m2][i + 1][j]);

				if (options->inf_func == FUNC_FPISIN)
				{
					star += fpisin_i * sin(pih * (double)j);
				}

				if (options->termination == TERM_PREC || term_iteration == 1)
				{
					residuum = Matrix[m2][i][j] - star;
					residuum = (residuum < 0) ? -residuum : residuum;
					maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
				}
				// printf("working here for rank=%d, i=%ld, j=%ld\n",arguments->rank,i,j);

				Matrix[m1][i][j] = star;
			}
		}
		// printf("rank %d, maxres: %f\n", arguments->rank, maxresiduum);
		/* like MPI_Reduce but the result is stored on all processes */
		MPI_Allreduce(&maxresiduum, &(results->stat_precision), 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

		results->stat_iteration++;

		/* exchange m1 and m2 */
		i = m1;
		m1 = m2;
		m2 = i;

		/* check for stopping calculation depending on termination method */
		if (options->termination == TERM_PREC)
		{
			if (results->stat_precision < options->term_precision)
			{
				term_iteration = 0;
			}
		}
		else if (options->termination == TERM_ITER)
		{
			term_iteration--;
		}

		// printf("\n\nIteration %d\n\n", results->stat_iteration);
	}
	/*
	if (arguments->rank == 0)
	{
		printf("\n\nCalculation ended!\n\n");
	}
	*/

	results->m = m2;
}

static void
calculate_gauss(struct calculation_arguments const *arguments, struct calculation_results *results, struct options const *options)
{
	int rank = arguments->rank;  /* current rank of process */
	int size = arguments->num_processes; /* total amount of processes */

	uint64_t i, j, iter, chunksize; /* local variables  */
	int m1, m2;			/* used as indices for old and new matrices */
	double star;		/* four times center value minus 4 neigh.b values */
	double residuum;	/* residuum of current iteration */
	double maxresiduum; /* maximum residuum value of a slave in iteration */
	uint64_t const N = arguments->N;
	double const h = arguments->h;
	double pih = 0.0;
	double fpisin = 0.0;

	/*  typedef of matrix, +2 because of overlapping rows */
	typedef double(*matrix)[arguments->chunk_size_process+2][N + 1];
	matrix Matrix = (matrix)arguments->Matrix;
	
	MPI_Request req; /* needed for MPI_Isend  */
	chunksize= arguments->chunk_size_process;
	iter = 0;

	int term_iteration = options->term_iteration;
	m1 = 0;
	m2 = 0;
	if (options->inf_func == FUNC_FPISIN)
	{
		pih =M_PI* h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

	/*
		termination because of iteration works.
		termination because of precision does not (because it hasnt been implemented).
		MPI_Allreduce completely halts the program.
		A valid approach might be to ISend the maxresiduums around, and perform a manual reduction if you will. 
	*/


	while (term_iteration > 0)
	{
		maxresiduum = 0;
		
		if(size > 1){
			if(rank == 0 && iter > 0){
				// printf("rank 0 receiving with iter %ld \n", iter);
				MPI_Recv(Matrix[0][chunksize+1],N+1, MPI_DOUBLE, 1, iter-1,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			if(rank >0 && rank < size -1 ){
				// printf("rank %d receiving from up with iter %ld \n",rank,  iter);
				MPI_Recv(Matrix[0][0],N+1,MPI_DOUBLE, rank-1, iter+1,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if(iter > 0){
					// printf("rank %d receiving from down with iter %ld \n",rank,  iter);
					MPI_Recv(Matrix[0][chunksize+1],N+1,MPI_DOUBLE, rank+1, iter-1,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
			if(rank == size -1){
				// printf("rank %d receiving with iter %ld \n",rank, iter);
				MPI_Recv(Matrix[0][0],N+1,MPI_DOUBLE, rank-1, iter+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}

		/* over all rows */
		for (i = 1; i <= arguments->chunk_size_process; i++)
		{
			double fpisin_i = 0.0;

			if (options->inf_func == FUNC_FPISIN)
			{
				fpisin_i = fpisin * sin(pih * (double)(i + arguments->i_start - 1));
			}
			/* over all columns */
			for (j = 1; j < N; j++)
			{
				star = 0.25 * (Matrix[m2][i - 1][j] + Matrix[m2][i][j - 1] + Matrix[m2][i][j + 1] + Matrix[m2][i + 1][j]);

				if (options->inf_func == FUNC_FPISIN)
				{
					star += fpisin_i * sin(pih * (double)j);
				}

				if (options->termination == TERM_PREC || term_iteration == 1)
				{
					residuum = Matrix[m2][i][j] - star;
					residuum = (residuum < 0) ? -residuum : residuum;
					maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
				}
				// printf("working here for rank=%d, i=%ld, j=%ld\n",arguments->rank,i,j);

				Matrix[m1][i][j] = star;
			}
			if(i == 1 && rank > 0){
				// printf("rank %d sending up with iter %ld \n", rank, iter);
				MPI_Isend(Matrix[0][1],N+1,MPI_DOUBLE, rank-1, iter,MPI_COMM_WORLD, &req);
			}
		}
		iter++;
		results->stat_iteration=iter;
		if(size > 1){
			if(rank == 0){
				// printf("0 sending with iter %ld \n", iter);
				MPI_Send(Matrix[0][chunksize], N+1, MPI_DOUBLE, 1, iter,MPI_COMM_WORLD);
			}
			if(rank >0 && rank < size -1 ){
				// printf("rank %d sending down with iter %ld \n", rank, iter);
				MPI_Send(Matrix[0][chunksize], N+1, MPI_DOUBLE, rank+1, iter,MPI_COMM_WORLD);
			}
		}

		/* exchange m1 and m2 */
		i = m1;
		m1 = m2;
		m2 = i;


		/* check for stopping calculation depending on termination method */
		if (options->termination == TERM_PREC)
		{
			if (results->stat_precision < options->term_precision)
			{
				term_iteration = 0;
			}
		}
		else if (options->termination == TERM_ITER)
		{
			term_iteration--;
		}

		// printf("\n\nIteration %d\n\n", results->stat_iteration);
	}
	/*
	if (arguments->rank == 0)
	{
		printf("\n\nCalculation ended!\n\n");
	}
	*/

	results->m = m2;
}

/* ************************************************************************ */
/*  displayStatistics: displays some statistics about the calculation       */
/* ************************************************************************ */
static void
displayStatistics(struct calculation_arguments const *arguments, struct calculation_results const *results, struct options const *options)
{
	if (arguments->rank != 0)
	{
		return;
	}
	int N = arguments->N;
	double time = (comp_time.tv_sec - start_time.tv_sec) + (comp_time.tv_usec - start_time.tv_usec) * 1e-6;

	printf("Berechnungszeit:    %f s\n", time);
	printf("Speicherbedarf:     %f MiB\n", (N + 1) * (N + 1) * sizeof(double) * arguments->num_matrices / 1024.0 / 1024.0);
	printf("Berechnungsmethode: ");

	if (options->method == METH_GAUSS_SEIDEL)
	{
		printf("Gauß-Seidel");
	}
	else if (options->method == METH_JACOBI)
	{
		printf("Jacobi");
	}

	printf("\n");
	printf("Interlines:         %" PRIu64 "\n", options->interlines);
	printf("Stoerfunktion:      ");

	if (options->inf_func == FUNC_F0)
	{
		printf("f(x,y) = 0");
	}
	else if (options->inf_func == FUNC_FPISIN)
	{
		printf("f(x,y) = 2pi^2*sin(pi*x)sin(pi*y)");
	}

	printf("\n");
	printf("Terminierung:       ");

	if (options->termination == TERM_PREC)
	{
		printf("Hinreichende Genaugkeit");
	}
	else if (options->termination == TERM_ITER)
	{
		printf("Anzahl der Iterationen");
	}

	printf("\n");
	printf("Anzahl Iterationen: %" PRIu64 "\n", results->stat_iteration);
	printf("Norm des Fehlers:   %e\n", results->stat_precision);
	printf("\n");
}

/****************************************************************************/
/** Beschreibung der Funktion displayMatrix:                               **/
/**                                                                        **/
/** Die Funktion displayMatrix gibt eine Matrix                            **/
/** in einer "ubersichtlichen Art und Weise auf die Standardausgabe aus.   **/
/**                                                                        **/
/** Die "Ubersichtlichkeit wird erreicht, indem nur ein Teil der Matrix    **/
/** ausgegeben wird. Aus der Matrix werden die Randzeilen/-spalten sowie   **/
/** sieben Zwischenzeilen ausgegeben.                                      **/
/****************************************************************************/

/*
 * rank und size sind der MPI-Rang und die Größe des Kommunikators
 * from und to stehen für den globalen(!) Bereich von Zeilen für die dieser Prozess zuständig ist
 *
 * Beispiel mit 9 Matrixzeilen und 4 Prozessen:
 * - Rang 0 is verantwortlich für Zeilen 1-2, Rang 1 für 3-4, Rang 2 für 5-6 und Rang 3 für 7
 * - Zeilen 0 und 8 sind nicht inkludiert, weil sie nicht berechnet werden
 * - Jeder Prozess speichert zwei Randzeilen in seiner Matrix
 * - Zum Beispiel: Rang 2 hat vier Zeilen 0-3 aber berechnet nur 1-2 weil 0 und 3 beide Randzeilen für andere Prozesse sind;
 *   Rang 2 ist daher verantwortlich für die globalen Zeilen 5-6
 */
static void
displayMatrixMpi(struct calculation_arguments *arguments, struct calculation_results *results, struct options *options, int rank, int size, int from, int to)
{
  int const elements = 8 * options->interlines + 9;

  int x, y;

  typedef double(*matrix)[to - from + 3][arguments->N + 1];
  matrix Matrix = (matrix)arguments->Matrix;
  int m = results->m;

  MPI_Status status;

  // Die erste Zeile gehört zu Rang 0
  if (rank == 0)
  {
    from--;
  }

  // Die letzte Zeile gehört zu Rang (size - 1)
  if (rank == size - 1)
  {
    to++;
  }

  if (rank == 0)
  {
    printf("Matrix:\n");
  }
  

  for (y = 0; y < 9; y++)
  {
    int line = y * (options->interlines + 1);

    if (rank == 0)
    {
      // Prüfen, ob die Zeile zu Rang 0 gehört
      if (line < from || line > to)
      {
        // Der Tag wird genutzt, um Zeilen in der richtigen Reihenfolge zu empfangen
        // Matrix[m][0] wird überschrieben, da die Werte nicht mehr benötigt werden
        MPI_Recv(Matrix[m][0], elements, MPI_DOUBLE, MPI_ANY_SOURCE, 42 + y, MPI_COMM_WORLD, &status);
      }
    }
    else
    {
      if (line >= from && line <= to)
      {
        // Zeile an Rang 0 senden, wenn sie dem aktuellen Prozess gehört
        // (line - from + 1) wird genutzt, um die lokale Zeile zu berechnen
        MPI_Send(Matrix[m][line - from + 1], elements, MPI_DOUBLE, 0, 42 + y, MPI_COMM_WORLD);
      }
    }

    if (rank == 0)
    {
      for (x = 0; x < 9; x++)
      {
        int col = x * (options->interlines + 1);

        if (line >= from && line <= to)
        {
          // Diese Zeile gehört zu Rang 0
          printf("%7.4f", Matrix[m][line][col]);
        }
        else
        {
          // Diese Zeile gehört zu einem anderen Rang und wurde weiter oben empfangen
          printf("%7.4f", Matrix[m][0][col]);
        }
      }

      printf("\n");
    }
  }
  fflush(stdout);
}


/* ************************************************************************ */
/*  main                                                                    */
/* ************************************************************************ */
int main(int argc, char **argv)
{
	struct options options;
	struct calculation_arguments arguments;
	struct calculation_results results;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &(arguments.num_processes));
	MPI_Comm_rank(MPI_COMM_WORLD, &(arguments.rank));

	askParams(&options, argc, argv);

	initVariables(&arguments, &results, &options);

	allocateMatrices(&arguments);
	initMatrices(&arguments, &options);

	gettimeofday(&start_time, NULL);
	/*  Seperated the methods as I would not like to see both cramped into one, I think the fs can handle couple more byte of text. */ 
	if(options.method == METH_JACOBI){
		calculate_jacobi(&arguments, &results, &options);
	}
	else{
		calculate_gauss(&arguments, &results, &options);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&comp_time, NULL);



	displayStatistics(&arguments, &results, &options);
	displayMatrixMpi(&arguments, &results, &options, arguments.rank, arguments.num_processes, arguments.i_start, arguments.i_end - 1);

	freeMatrices(&arguments);
	MPI_Finalize();
	return 0;
}
