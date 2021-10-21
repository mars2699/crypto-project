#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS     idk yet

// **AUTO REGRESSION MODEL** 

int AutoRegression(
   double   *inputseries,
   int      length,
   int      degree,
   double   *coefficients,
   int      method)