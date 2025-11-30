
#include "vq_util.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

static double *covMat = NULL;
static int covKind = NULLC;
static int covOff = 0;
static double *vTmp = NULL;
static int dimTmp = 0;

void
printMat(double *mat, int N, int M)
{
  int n,m;
  for (n=0;n<N;n++){
    for (m=0;m<M;m++)
      printf("%f ", mat[n+m*N]);
    printf("\n");
  }
}

void
printIntMat(double *mat, int N, int M)
{
  int n,m;
  for (n=0;n<N;n++){
    for (m=0;m<M;m++)
      printf("%d ", mat[n+m*N]);
    printf("\n");
  }
}

void
setCov( double *cov, int kind, int dim )
{
  covMat = cov;
  covKind = kind;
  switch (kind) {
  case NULLC:
    covOff = 0;
    break;
  case DIAGC:
  case INVDIAGC:
    covOff = dim;
    break;
  case FULLC:
  case INVFULLC:
    covOff = dim*dim;
    break;
  default:
    fprintf( stderr, "Cannot find covariance type [%d].\n",kind);
  }
}

void
printCov(int dim)
{
  switch(covKind) {
  case NULLC:
    break;
  case DIAGC:
  case INVDIAGC:
    printMat(covMat,dim,1);
    break;
  case FULLC:
  case INVFULLC:
    printMat(covMat,dim,dim);
    break;
  default:
    fprintf( stderr, "Cannot find covariance type [%d].\n", covKind);
  }
    
}

double
Distance(double *v1, double *v2, int dim)
{
  double *iv, *crow;
  double *ic;
  double sum,x;
  int i,j;

  /* printf("before\n"); */
  if (vTmp==NULL) {
    vTmp = calloc( dim, sizeof(double) );
    dimTmp = dim;
  }
  
  if (dimTmp<dim) {
    free(vTmp);
    vTmp = calloc( dim, sizeof(double) );
    dimTmp = dim;
  }
  /* printf("after\n"); */

  switch(covKind){
  case NULLC:
    sum = 0.0;
    for (i=0; i<dim; i++){
      x = v1[i]-v2[i]; sum += x*x;
    }
    break;
  case DIAGC:
    iv = covMat;  /* covKind == DIAGC */
    sum = 0.0;
    for (i=0; i<dim; i++){
      x = v1[i]-v2[i]; sum += x*x/iv[i];
    }
    break;
  case INVDIAGC:
    iv = covMat;  /* covKind == INVDIAGC */
    sum = 0.0;
    for (i=0; i<dim; i++){
      x = v1[i]-v2[i]; sum += x*x*iv[i];
    }
    break;
  case INVFULLC:
    ic = covMat; /* covKind == INVFULLC */
    for (i=0;i<dim;i++)
      vTmp[i] = v1[i] - v2[i];
    sum = 0.0;
    for (i=0;i<dim;i++) {
      crow = ic+i*dim;
      for (j=0; j<i; j++)
	sum += vTmp[i]*vTmp[j]*crow[j];
    }
    sum *= 2;
    for (i=0;i<dim;i++)
      sum += vTmp[i] * vTmp[i] * ic[i+i*dim];
    break;
  case FULLC:
    fprintf( stderr, "Distance with covariance type %d not supported.\n",
	     covKind);
    break;
  default:
    fprintf( stderr, "Cannot find covariance type [%d].\n",covKind);
  }
  return sqrt(sum);
}


int 
AllocateVectors( double *vectList, int dim, int len,
		 double *clusterMean,
		 double *clusterCost, int *clusterSize,
		 int *clusterMap,
		 int nc, int minClustSize,
		 double *totalCost)
{
  int n, i, bestn, cs;
  double d,min;
  double *v, *mat = covMat;

  /* zero all clusters */
  *totalCost=0.0;
  for (n=0; n<nc; n++){
    clusterCost[n] = 0.0;
    clusterSize[n] = 0;
  }
  
  /* scan pool of vectors */
  for (i=0; i<len; i++) {

    v = vectList+i*dim;

    /* find center nearest to i'th vector */
    min=Distance(v,clusterMean,dim); bestn = 0;
    if (covMat) covMat += covOff;

    for (n=1; n<nc; n++) {
      d = Distance(v,clusterMean+n*dim,dim);
      if (covMat) covMat += covOff;
      if (d < min) {
	min = d; bestn = n;
      }
    }

/*     printf(" Vector %d is closest to %d with distance %f\n", */
/* 	   i, bestn, min); */
    

    /* increment costs and allocate vector to bestn */
    *totalCost += min;   
    clusterCost[bestn] += min;
    clusterSize[bestn] ++;
    clusterMap[i] = bestn;

    covMat = mat;
  }

  /* Check for any empty clusters and average costs */
  for (n=0; n<nc; n++) {
    cs = clusterSize[n];
    if (cs < minClustSize) {
      return n;
    }
    clusterCost[n] /= cs;
  }
  return -1;
}

/* SplitVectors: distribute all pool vectors in cluster n between
   clusters n1 and n2.  The centers of these clusters have already
   been set.  Each vector is placed in cluster with nearest center.   */
void 
SplitVectors( double *vectList, int dim, int len,
	      double *clusterMean, int *clusterMap, int *clusterSize,
	      int n, int n1, int n2)
{
  int i, bestn;
  double d1,d2;
  double *v;
  int c,c1,c2;

  for (i=0; i<len; i++)

    if (clusterMap[i]==n) {

      v = vectList+i*dim;

      /* find center nearest to i'th vector */ 
      d1=Distance(v,clusterMean+n1*dim,dim); 
      d2=Distance(v,clusterMean+n2*dim,dim); 
      bestn = (d1<d2)?n1:n2;

      /* allocate vector to bestn */
      clusterMap[i] = bestn; 
      clusterSize[bestn] ++;

    }

  /* Check for any empty clusters and average costs */
  c = clusterSize[n];
  c1 = clusterSize[n1];
  c2 = clusterSize[n2];
  if (c1 == 0 || c2 == 0)
    fprintf(stderr, "SplitVectors: empty cluster %d[%d] ->%d[%d] + %d[%d]",
	   n,c,n1,c1,n2,c2);
}

void 
FindCenters( double *vectList, int dim, int len,
	     double *clusterMean, int *clusterMap, int *clusterSize,
	     int a, int b)
{
  int n,i,j,cidx;
  double *v, *ctr;
  double cs;
   
  for (n=a; n<=b; n++) {
    v = clusterMean + n*dim;
    for (i=0;i<dim;i++)
      v[i] = 0.0;
  }

  for (i=0; i<len; i++) {
    cidx = clusterMap[i];
    if (cidx>=a && cidx<=b){
      v = vectList + i*dim;
      ctr = clusterMean + cidx*dim;
      for (j=0; j<dim; j++)
	ctr[j] += v[j];
    }
  }

  for (n=a; n<=b; n++){
    ctr = clusterMean + n*dim;
    cs = clusterSize[n];
    for (j=0; j<dim; j++) 
      ctr[j] /= cs;
  }
}

/* FindCovariance: of cluster n */
void 
FindCovariance( double *vectList, int dim, int len,
		double *clusterMean, int *clusterMap, int *clusterSize,
		int n, int kind,
		double *clusterCov )
{
  double *v, *mean;
  double *t;
  int i,j,k;
  double nx,x,y;
  double *sqsum;
  double *xsum;
  double *c;
   
  nx = clusterSize[n];
  mean = clusterMean + n*dim;

  switch(kind){
  case NULLC:
    
    break;

  case DIAGC:       /* diagonal covariance matrix */
  case INVDIAGC:    /* inverse diag covariance matrix */

    sqsum = calloc( dim, sizeof(double) );

    for (i=0;i<dim;i++)
      sqsum[i] = 0.0;

    for (i=0; i<len; i++) {
      if (clusterMap[i] == n) {
	v = vectList + i*dim;
	for (j=0; j<dim; j++){
	  x = v[j]-mean[j];
	  sqsum[j] += x*x;
	}
      }
    }

    v = clusterCov + n*dim;
    for (j=0; j<dim; j++)
      v[j] = (kind==DIAGC)?sqsum[j]/nx:nx/sqsum[j];

    free(sqsum);

    break;

  case FULLC:    /* full covariance matrix */
  case INVFULLC:

    xsum = calloc( dim*dim, sizeof(double) );

    for (i=0;i<dim*dim;i++)
      xsum[i] = 0.0;

    for (i=0; i<len; i++) {
      if (clusterMap[i] == n) {
	v = vectList + i*dim;
	for (j=0; j<dim; j++)
	  for (k=0; k<=j; k++){
	    x = v[j]-mean[j];
	    y = v[k]-mean[k];
	    xsum[j+ dim*k] += x*y;
	  }
      }
    }

    t = clusterCov + n*dim*dim;
    for (j=0; j<dim; j++) {
      t[j+j*dim] = xsum[j+j*dim]/nx;
      for (k=0; k<j; k++)
	t[j+k*dim] = t[k+j*dim] = xsum[j+k*dim]/nx;
    }

    if (kind==INVFULLC) {
      /* Invert t  here - not done for now */
    }
    
    free(xsum);

    break;

  default:
   fprintf(stderr, "FindCovariance: unsupported cov kind [%d]", kind);
   
  }
  
}

/* BiggestCluster: return index of cluster with highest average cost */
int 
BiggestCluster( double *clusterCost, int *clusterSize, int nc, 
		int minClustSize )
{
  double maxCost;
  int n,biggest;
   
  maxCost = clusterCost[0]; biggest = 0;
  for (n=1; n<nc; n++)
    if (clusterCost[n] > maxCost) {
      maxCost=clusterCost[n];
      if(clusterSize[n] >= minClustSize )
	biggest=n;
    }
  return biggest;
}

/* FullestCluster: return index of cluster with most elements */
int 
FullestCluster( int *clusterSize, int nc)
{
  int max,n,fullest;
   
  max=clusterSize[0]; fullest = 0;
  for (n=1; n<nc; n++)
    if (clusterSize[n] > max) {
      max=clusterSize[n]; fullest=n;
    }
  return fullest;
}

/* Perturb: copy perturbed versions of cluster n into n1 and n2. */
void 
Perturb( double *clusterMean, int dim,
	 int n, int n1, int n2)
{
  int i;
  double *v,*v1,*v2;
  double x;
   
  v  = clusterMean+n*dim;
  v1 = clusterMean+n1*dim;
  v2 = clusterMean+n2*dim;
  if (n!=n1) 
    for (i=0;i<dim;i++)
      v1[i] = v[i];
  if (n!=n2) 
    for (i=0;i<dim;i++)
      v2[i] = v[i];
  
  for (i=0; i<dim; i++) {
    x = fabs(v[i]*0.01);
    if (x<0.0001) x=0.0001;
    v1[i] += x; v2[i] -= x;
  }

}


/* InitClustering: create the cluster set data structure */
void 
InitClustering( double *vectList, int dim, int len,
		double *clusterMean,
		double *clusterCost, int *clusterSize,
		int *clusterMap,
		int nc )
{
  int i;

  if (len < nc) {
    fprintf(stderr, "InitClustering: only %d items for %d clusters",
	    len,nc);
  }

  vTmp = NULL;
  dimTmp = 0;
  covKind = NULLC;

  for (i=0; i<len; i++)  /* put all vecs in cluster 0 */
    clusterMap[i] = 0;
  
  clusterSize[0] = len;
  clusterCost[0] = 1.0;

  FindCenters( vectList, dim, len, clusterMean, clusterMap, clusterSize, 0,0);

}


void
FlatCluster( double *vectList, int dim, int len,
	     double *clusterMean, double *clusterCov,
	     double *clusterCost, int *clusterSize,
	     int *clusterMap,
	     int nc,
	     int kind, int minClustSize, int maxIter,
	     double *cost )
{
  int i,c,cc,ce,iter,repairCount;
  double oldCost,newCost;
  int converged;
  int curNumCl = 1;
   
  InitClustering( vectList, dim, len,
		  clusterMean, clusterCost, clusterSize, clusterMap,
		  nc );


  for (c=1; c<nc; c++) {

    /* increase num clusters by splitting biggest */
    cc = BiggestCluster( clusterCost, clusterSize, c, minClustSize);
    Perturb( clusterMean, dim, cc,cc,c ); curNumCl = c+1;
    oldCost = 1e10;

    /* reallocate vectors until cost stabilizes */
    for (iter=0,converged=0; !converged; iter++){

      repairCount = 0; /* try to fill empty clusters by splitting fullest */

      while ( (ce=AllocateVectors( vectList, dim, len,
				   clusterMean, clusterCost, clusterSize,
				   clusterMap, curNumCl, minClustSize,
				   &newCost)) >= 0  && ++repairCount <c) {
	cc = FullestCluster( clusterSize, curNumCl );
	Perturb( clusterMean, dim, cc,cc,ce );      /* ce = empty cluster */
      }

      if (ce >= 0)
	fprintf(stderr, "FlatCluster: Failed to make %d clusters at iter %d\n",c,iter);
      
      converged = (iter>=maxIter) || ((oldCost-newCost) / oldCost < 0.001);
 
      FindCenters(vectList, dim, len, clusterMean, clusterMap, clusterSize, 
		  0,curNumCl-1); 
      oldCost = newCost;

    }

  }

  for (i=0; i<nc; i++)
    FindCovariance( vectList, dim, len, 
		    clusterMean, clusterMap, clusterSize,
		    i, kind, 
		    clusterCov );

  *cost = newCost;
}  
