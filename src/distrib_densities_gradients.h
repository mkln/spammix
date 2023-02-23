#ifndef UTILS_DENS_GRAD 
#define UTILS_DENS_GRAD

#include <RcppArmadillo.h>

using namespace std;

const double TOL_LOG_LOW=exp(-10);
const double TOL_HIGH=exp(10);
const double TOL_LOG_HIGH=10;

inline double poisson_logpmf(const double& x, double lambda){
  if(lambda < TOL_LOG_LOW){
    lambda = TOL_LOG_LOW;
  } else {
    if(lambda > TOL_HIGH){
      lambda = TOL_HIGH;
    }
  }
  return x * log(lambda) - lambda - lgamma(x+1);
}

inline double poisnmix_loggradient(const double& y, const double& offset, const double& prob, const double& w){
  // llik: y * log(lambda) - lambda - lgamma(y+1);
  // lambda = exp(o + w);
  // llik: y * (o + w) - exp(o+w);
  // grad: y - exp(o+w)
  if(offset + w > TOL_LOG_HIGH){
    return y - TOL_HIGH;
  }
  return y - prob * exp(offset + w);
}

inline double poisnmix_neghess_mult_sqrt(const double& mu){
  return pow(mu, 0.5);
}

inline double get_mult(const double& y, //const double& tausq, 
                       const double& offset, 
                       const double& prob,
                       const double& xij){
  // if m is the output from this function, then
  // m^2 X'X is the negative hessian of a glm model in which X*x is the linear term
  double mult=1;
  double mu = prob * exp(offset + xij);
  mult = poisnmix_neghess_mult_sqrt(mu);
  return mult;
}

inline double poisnmix_logdens_loggrad(double& loglike,
                                          const double& y, 
                                          const double& offset, const double& prob, 
                                          const double& xij, 
                                          bool do_grad=true){
  
  double gradloc;
  
  double lambda = prob * exp(offset + xij);//xz);//x(i));
  loglike += poisson_logpmf(y, lambda);
  if(do_grad){ gradloc = poisnmix_loggradient(y, offset, prob, xij); } //xz) * z(i);
  
  
  return gradloc;
}

#endif
