
 
#ifndef _VQ_UTIL_H_
#define _VQ_UTIL_H_

#define NULLC		0
#define DIAGC		1
#define INVDIAGC	2
#define FULLC		3
#define INVFULLC	4

#ifdef __cplusplus
extern "C" {
#endif
  
double Distance(double *v1, double *v2, int dim);
int AllocateVectors( double *vectList, int dim, int len,
		    double *clusterMean,
		    double *clusterCost, int *clusterSize,
		    int *clusterMap,
		    int nc, int minClustSize,
		    double *totalCost);
void SplitVectors( double *vectList, int dim, int len,
		   double *clusterMean, int *clusterMap, int *clusterSize,
		   int n, int n1, int n2);
void FindCenters( double *vectList, int dim, int len,
		  double *clusterMean, int *clusterMap, int *clusterSize,
		  int a, int b);
void FindCovariance( double *vectList, int dim, int len,
		     double *clusterMean, int *clusterMap, int *clusterSize,
		     int n, int kind,
		     double *clusterCov );
int BiggestCluster( double *clusterCost, int *clusterSize, int nc, 
		    int minClustSize );
int FullestCluster( int *clusterSize, int nc);
void Perturb( double *clusterMean, int dim,
	      int n, int n1, int n2);
void InitClustering( double *vectList, int dim, int len,
		     double *clusterMean,
		     double *clusterCost, int *clusterSize,
		     int *clusterMap,
		     int nc );

void FlatCluster( double *vectList, int dim, int len,
		  double *clusterMean, double *clusterCov,
		  double *clusterCost, int *clusterSize,
		  int *clusterMap,
		  int nc,
		  int kind, int minClustSize, int maxIter,
		  double *cost);

void setCov( double *mat, int kind, int dim );

#ifdef __cplusplus
	       }
#endif

#endif /* _VQ_UTIL_H_ */

  
