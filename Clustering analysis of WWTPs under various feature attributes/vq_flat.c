#include <math.h>
#include "mex.h"
#include "vq_util.h"

#define GET_TOTAL_ELEMENTS(a)  (mxGetM(a) * mxGetN(a)) 

/* Input Arguments */

#define	VEC	prhs[0]
#define NC	prhs[1]
#define CVTYPE	prhs[2]
#define MC	prhs[3]
#define MITER	prhs[4]

/* Output Arguments */

#define COST	plhs[0]
#define CMEAN	plhs[1]
#define CVAR	plhs[2]
#define CCOST	plhs[3]
#define CSIZE	plhs[4]
#define	CMAP	plhs[5]


void 
mexFunction(
	    int nlhs,       mxArray *plhs[],
	    int nrhs, const mxArray *prhs[]
	    )
{
  double	*Vec, *Cost, *CMean, *CVar, *CMap, *CSize, *CCost, *tmp;
  int		i, N, L, Nc, CVType = DIAGC, maxIter = 100, minClustSize = 1;
  int		*cmap, *csize, dims[3];
  
  /* Check for proper number of arguments */
  
  if (nrhs < 2)  {
    mexErrMsgTxt("Requires at least two arguments.");
  }
  
  N = mxGetM( VEC );
  L = mxGetN( VEC );
  
  /* Get input pointers */
  Vec = mxGetPr( VEC );
  tmp = mxGetPr( NC ); Nc = (int) tmp[0];
  
  if (Nc>L)
    mexErrMsgTxt("Too many clusters for too few data vectors.");

  if (nrhs>=3) {
    tmp = mxGetPr( CVTYPE );
    CVType = (int) tmp[0];
    if (CVType<NULLC || CVType>INVFULLC)
      mexErrMsgTxt("Unsupported covariance type.");     
  }

  if (nrhs>=4) {
    tmp = mxGetPr( MC );
    minClustSize = (int) tmp[0];
  }

  if (nrhs>=5) {
    tmp = mxGetPr( MITER );
    maxIter = (int) tmp[0];
  }

  /* Create outputs */
  COST = mxCreateDoubleMatrix(1,1, mxREAL);
  Cost = mxGetPr( COST );

  CMEAN = mxCreateDoubleMatrix(N,Nc, mxREAL);
  CMean = mxGetPr( CMEAN );

  if (CVType==FULLC || CVType==INVFULLC) {
    dims[0] = N; dims[1] = N; dims[2] = Nc;
    CVAR = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
  } else
    CVAR = mxCreateDoubleMatrix(N,Nc, mxREAL);   
  CVar = mxGetPr( CVAR );

  CCOST = mxCreateDoubleMatrix(1,Nc, mxREAL);
  CCost = mxGetPr( CCOST );
  
  CSIZE = mxCreateDoubleMatrix(1,Nc, mxREAL);
  CSize = mxGetPr( CSIZE );
  
  CMAP = mxCreateDoubleMatrix(1,L, mxREAL);
  CMap = mxGetPr( CMAP );

  /* Allocate tmp data storage */
  cmap  = mxCalloc(L,sizeof(int));
  csize = mxCalloc(Nc,sizeof(int));

  /* Call FlatCluster function */
  FlatCluster( Vec, N, L,
	       CMean, CVar, CCost, csize, cmap, Nc,
	       CVType, minClustSize, maxIter,
	       Cost);

  /* Copy tmp to out */
  for (i=0;i<L;i++)
    CMap[i] = cmap[i]+1;
  for (i=0;i<Nc;i++)
    CSize[i] = csize[i];

  /* If covariance is INVFULLC find inverses */
  if (CVType==INVFULLC) {

    double *mat, *imat, dmat;
    int j;

    imat = mxCalloc(N*N,sizeof(double));

      for (i=0;i<Nc;i++) {
      mat = CVar+i*N*N;
      /*sinv(mat, N, imat, &dmat);*/
      for (j=0;j<N*N;j++)
	mat[j] = imat[j];
    }
    
    mxFree(imat);
  }

  /* Free tmp data storage */
  mxFree(cmap);
  mxFree(csize);

  return;
}
