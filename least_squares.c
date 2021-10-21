#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS     idk yet
#include "linreg.h"
#include <math.h> 
#define REAL float
#define REAL double


// **LEAST SQUARES REGRESSION MODEL** 
                          


//n = number of data points
//x,y  = arrays of data
//*b = y intercept
//*m  = slope
//*V = volatility/correlation coefficient 

// REAL returns the real part of a complex number

// Parse in data from csv file?

int linreg(int n, const REAL x[], const REAL y[], REAL* m, REAL* b, REAL* V){
    REAL   sumx = 0.0;                      /* sum of x     */
    REAL   sumx2 = 0.0;                     /* sum of x**2  */
    REAL   sumxy = 0.0;                     /* sum of x * y */
    REAL   sumy = 0.0;                      /* sum of y     */
    REAL   sumy2 = 0.0;                     /* sum of y**2  */

    for (int i = 0; i < n; i++){ 
        sumx  += x[i];       
        sumx2 += sqr(x[i]);  
        sumxy += x[i] * y[i];
        sumy  += y[i];      
        sumy2 += sqr(y[i]); 
    } 

    
	
// From here, we could plot with gnu plot?