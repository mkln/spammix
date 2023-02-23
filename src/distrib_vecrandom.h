#ifndef MSP_VECRND
#define MSP_VECRND

#include "RcppArmadillo.h"
#include "R.h"

using namespace std;


inline arma::mat mrstdnorm(int r, int c){
  arma::mat result = arma::zeros(r, c);
  for(int i=0; i<r; i++){
    for(int j=0; j<c; j++){
      result(i,j) = R::rnorm(0,1);
    }
  }
  return result;
}

inline arma::vec vrpois(const arma::vec& lambdas){
  arma::vec y = arma::zeros(lambdas.n_elem);
  for(unsigned int i=0; i<lambdas.n_elem; i++){
    y(i) = R::rpois(lambdas(i));
  }
  return y;
}

inline arma::vec vrunif(int n){
  arma::vec result = arma::zeros(n);
  for(int i=0; i<n; i++){
    result(i) = R::runif(0, 1);
  }
  return result;
}


#endif
