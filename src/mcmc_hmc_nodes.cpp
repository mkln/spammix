#include "mcmc_hmc_nodes.h"

NodeData::NodeData(){
  n=-1;
}

double NodeData::logfullcondit(const arma::vec& x){
  return 0;
}

arma::vec NodeData::gradient_logfullcondit(const arma::vec& x){
  return 0;
}


NodeDataW::NodeDataW(){
  n=-1;
}

NodeDataW::NodeDataW(const arma::mat& y_all, //const arma::mat& Z_in,
                                    const arma::umat& na_mat_all, const arma::mat& offset_all, 
                                    const arma::uvec& indexing_target_in,
                                    int k){
  
  indexing_target = indexing_target_in;
  y = y_all.rows(indexing_target);
  offset = offset_all.rows(indexing_target);
  na_mat = na_mat_all.rows(indexing_target);
  
  // ----
  
  n = y.n_rows;
  
}


void NodeDataW::update_mv(const arma::mat& offset_all, 
                          const arma::mat& prob_offset_all,
                          const arma::mat& Lambda_lmc_in){
  // arma::mat tausqmat = arma::zeros<arma::umat>(arma::size(new_offset));
  // for(unsigned int i=0; i<tausqmat.n_cols; i++){
  //   tausqmat.col(i).fill(tausq(i));
  // }
  // tausq_long = arma::vectorise(tausqmat);
  Lambda_lmc = Lambda_lmc_in;
  //tausq = tausq_in;
  offset = offset_all.rows(indexing_target);
  prob_offset = prob_offset_all.rows(indexing_target);
}


double NodeDataW::fwdcond_dmvn(const arma::mat& x, 
                    const arma::cube* Ri,
                    const arma::mat& Kcxpar){
  // conditional of x | parents
  
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(Kcxpar.n_cols > 0){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    numer += arma::conv_to<double>::from( xcentered.t() * (*Ri).slice(j) * xcentered );
  }
  return -.5 * numer;//result;
}

arma::vec NodeDataW::grad_fwdcond_dmvn(const arma::mat& x){
  
  // gradient of conditional of x | parents
  arma::mat norm_grad = arma::zeros(arma::size(x));
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(Kcxpar.n_cols > 0){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    norm_grad.col(j) = -(*Ri).slice(j) * xcentered;
  }
  return arma::vectorise(norm_grad);//result;
}

void NodeDataW::fwdconditional_mvn(double& logtarget, arma::vec& gradient, 
                        const arma::mat& x){
  arma::mat norm_grad = arma::zeros(arma::size(x));
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(Kcxpar.n_cols > 0){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    arma::vec Rix = (*Ri).slice(j) * xcentered;
    numer += arma::conv_to<double>::from( xcentered.t() * Rix );
    norm_grad.col(j) = - Rix;
  }
  logtarget = -.5 * numer;//result;
  gradient = arma::vectorise(norm_grad);//result;
}



double NodeDataW::bwdcond_dmvn(const arma::mat& x, int c){
  // conditional of Y | x, others
  
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = w_child(c).col(j) - Kcx_x(c).slice(j)*x.col(j);
    if(Kco_wo(c).n_cols > 0){
      xcentered -= Kco_wo(c).col(j);
    } 
    numer += arma::conv_to<double>::from(xcentered.t() * (*Ri_of_child(c)).slice(j) * xcentered);
  }
  
  return -0.5*numer;
}

arma::vec NodeDataW::grad_bwdcond_dmvn(const arma::mat& x, int c){
  // gradient of conditional of Y | x, others
  arma::mat result = arma::zeros(arma::size(x));
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::mat wccenter = w_child(c).col(j) - Kcx_x(c).slice(j) * x.col(j);
    if(Kco_wo(c).n_cols > 0){
      wccenter -= Kco_wo(c).col(j);
    } 
    result.col(j) = Kcx_x(c).slice(j).t() * (*Ri_of_child(c)).slice(j) * wccenter;
  }
  return arma::vectorise(result);
}

void NodeDataW::bwdconditional_mvn(double& xtarget, arma::vec& gradient, const arma::mat& x, int c){
  
  arma::mat result = arma::zeros(arma::size(x));
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = w_child(c).col(j) - Kcx_x(c).slice(j)*x.col(j);
    if(Kco_wo(c).n_cols > 0){
      xcentered -= Kco_wo(c).col(j);
    } 
    arma::vec Rix = (*Ri_of_child(c)).slice(j) * xcentered;
    numer += arma::conv_to<double>::from(xcentered.t() * Rix);
    result.col(j) = Kcx_x(c).slice(j).t() * Rix;
  }
  xtarget -= 0.5*numer;
  gradient += arma::vectorise(result);
}

void NodeDataW::neghess_fwdcond_dmvn(arma::mat& result, const arma::mat& x){
  
  int k = (*Ri).n_slices;
  int nr = (*Ri).n_rows;
  int nc = (*Ri).n_cols;
  
  for(int j=0; j<k; j++){
    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) += (*Ri).slice(j);
  }
}

void NodeDataW::neghess_bwdcond_dmvn(arma::mat& result,
                          const arma::mat& x, int c){
  
  int k = (*Ri_of_child(c)).n_slices;
  int nr = Kcx_x(c).n_cols; //(*Ri_of_child).n_rows;
  int nc = Kcx_x(c).n_cols; //(*Ri_of_child).n_cols;
  
  //arma::mat result = arma::zeros(nr * k, nc * k);
  for(int j=0; j<k; j++){
    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) += Kcx_x(c).slice(j).t() * (*Ri_of_child(c)).slice(j) * Kcx_x(c).slice(j);
  }
  
  //return result;
}

void NodeDataW::mvn_dens_grad_neghess(double& xtarget, arma::vec& gradient, arma::mat& neghess,
                           const arma::mat& x, int c){
  
  
  int k = (*Ri_of_child(c)).n_slices;
  int nr = Kcx_x(c).n_cols; //(*Ri_of_child).n_rows;
  int nc = Kcx_x(c).n_cols; //(*Ri_of_child).n_cols;
  
  arma::mat result = arma::zeros(arma::size(x));
  double numer = 0;
  for(int j=0; j<k; j++){
    arma::vec xcentered = w_child(c).col(j) - Kcx_x(c).slice(j)*x.col(j);
    if(Kco_wo(c).n_cols > 0){
      xcentered -= Kco_wo(c).col(j);
    } 
    arma::mat KRichild = Kcx_x(c).slice(j).t() * (*Ri_of_child(c)).slice(j);
    numer += arma::conv_to<double>::from(xcentered.t() * (*Ri_of_child(c)).slice(j) * xcentered);
    result.col(j) = KRichild * xcentered;
    neghess.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) += KRichild * Kcx_x(c).slice(j);
  }
  xtarget -= 0.5*numer;
  gradient += arma::vectorise(result);
  
}


arma::vec NodeDataW::compute_dens_and_grad(double& xtarget, const arma::mat& x){
  unsigned int nr = y.n_rows;
  unsigned int q = y.n_cols;
  unsigned int k = x.n_cols;
  
  arma::vec grad_loglike = arma::zeros(x.n_rows * x.n_cols);
  
  int indxsize = x.n_rows;
  
  double loglike = 0;
  for(int i=0; i<nr; i++){
    arma::mat LambdaH, Hloc;
    arma::mat wloc = x.row(i);
    
    for(unsigned int j=0; j<q; j++){
      if(na_mat(i, j) > 0){
        
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        double gradloc = poisnmix_logdens_loggrad(loglike, y(i,j), offset(i, j), prob_offset(i, j), xij); 

        arma::mat LambdaHt = Lambda_lmc.row(j).t();
        arma::vec Lgrad = LambdaHt * gradloc;
        for(unsigned int s1=0; s1<k; s1++){
          grad_loglike(s1 * indxsize + i) += Lgrad(s1);   
        }  
      
      }
    }
  }
  
  // GP prior
  double logprior = 0;
  arma::vec grad_logprior_par;
  
  //double logprior = fwdcond_dmvn(x, Ri, Kcxpar);
  //arma::vec grad_logprior_par = grad_fwdcond_dmvn(x, Ri, Kcxpar);
  fwdconditional_mvn(logprior, grad_logprior_par, x);
  
  double logprior_chi = 0;
  arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
  for(unsigned int c=0; c<num_children; c++ ){
    bwdconditional_mvn(logprior_chi, grad_logprior_chi, x, c);
  }
  
  xtarget = logprior + logprior_chi + loglike;
  
  return grad_loglike + 
    grad_logprior_par + 
    grad_logprior_chi;

}

// log posterior 
double NodeDataW::logfullcondit(const arma::mat& x){
  double loglike = 0;
  
  for(unsigned int i=0; i<y.n_rows; i++){
    arma::mat wloc = x.row(i);
    
    for(unsigned int j=0; j<y.n_cols; j++){
      //Rcpp::Rcout << i << " - " << j << endl;
      if(na_mat(i, j) > 0){
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        double notused = poisnmix_logdens_loggrad(loglike, y(i,j), offset(i, j), prob_offset(i,j), xij, false);
      }
    }
  }
  
  // GP prior
  double logprior = fwdcond_dmvn(x, Ri, Kcxpar);
  for(unsigned int c=0; c<num_children; c++ ){
    logprior += bwdcond_dmvn(x, c);
    
  }
  return ( loglike + logprior );
}

// log posterior 
double NodeDataW::loglike(const arma::mat& x){
  double loglike = 0;
  
  for(unsigned int i=0; i<y.n_rows; i++){
    arma::mat wloc = x.row(i);
    
    for(unsigned int j=0; j<y.n_cols; j++){
      //Rcpp::Rcout << i << " - " << j << endl;
      if(na_mat(i, j) > 0){
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        double notused = poisnmix_logdens_loggrad(loglike, y(i,j), offset(i, j), prob_offset(i, j), xij, false);
      }
    }
  }
  
  return ( loglike );
}

// Gradient of the log posterior
arma::vec NodeDataW::gradient_logfullcondit(const arma::mat& x){
  int q = y.n_cols;
  int k = x.n_cols;
  
  arma::vec grad_loglike = arma::zeros(x.n_rows * x.n_cols);
  
  int nr = y.n_rows;
  int indxsize = x.n_rows;

  for(int i=0; i<nr; i++){
    arma::mat wloc = x.row(i);
    
    for(unsigned int j=0; j<y.n_cols; j++){
      if(na_mat(i, j) > 0){
        double loglike = 0;
        arma::mat LambdaHt = Lambda_lmc.row(j).t();
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        arma::vec gradloc = LambdaHt * poisnmix_logdens_loggrad(loglike, y(i,j), offset(i, j), prob_offset(i, j), xij);
        
        for(int s=0; s<k; s++){
          grad_loglike(s * indxsize + i) += gradloc(s);   
        }
      } 
    }
  }
  
  arma::vec grad_logprior_par = grad_fwdcond_dmvn(x);
  arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
  for(unsigned int c=0; c<num_children; c++ ){
    grad_logprior_chi += grad_bwdcond_dmvn(x, c);
  }
  
  return grad_loglike + 
    grad_logprior_par + 
    grad_logprior_chi;
}


arma::mat NodeDataW::compute_dens_grad_neghess(double& xtarget, arma::vec& xgrad, const arma::mat& x){
  int nr = y.n_rows;
  int q = y.n_cols;
  int k = x.n_cols;
  
  arma::vec grad_loglike = arma::zeros(x.n_rows * x.n_cols);
  arma::mat neghess_logtarg = arma::zeros(x.n_rows * x.n_cols,
                                          x.n_rows * x.n_cols);
  
  int indxsize = x.n_rows;
  
  double loglike = 0;
  
  for(int i=0; i<nr; i++){
    arma::mat Hloc;
    arma::mat LambdaH;
    arma::mat wloc = x.row(i);
    
    // get mult
    arma::vec mult = arma::zeros(q);
    for(int j=0; j<q; j++){
      if(na_mat(i, j) > 0){
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        mult(j) = get_mult(y(i,j), offset(i,j),  prob_offset(i, j), xij);
      }
    }
    
    for(int j=0; j<q; j++){
      if(na_mat(i, j) > 0){
        
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        double gradloc = poisnmix_logdens_loggrad(loglike, y(i,j), offset(i, j), prob_offset(i, j), xij);

        arma::mat LambdaHt = Lambda_lmc.row(j).t();
        arma::vec Lgrad = LambdaHt * gradloc;
        
        arma::mat hess_LambdaH = Lambda_lmc.row(j).t() * mult(j);
        arma::mat neghessloc = hess_LambdaH * hess_LambdaH.t();
        
        for(int s1=0; s1<k; s1++){
          grad_loglike(s1 * indxsize + i) += Lgrad(s1);   
          for(int s2=0; s2<k; s2++){
            neghess_logtarg(s1 * indxsize + i, s2*indxsize + i) += neghessloc(s1, s2);
          }
        }  
      }
    }
  }
  
  //Rcpp::Rcout << "compute_dens_grad_neghess finishing " << endl;
  // GP prior
  double logprior = 0;
  arma::vec grad_logprior_par = arma::zeros(x.n_elem);
  
  //double logprior = fwdcond_dmvn(x, Ri, Kcxpar);
  //arma::vec grad_logprior_par = grad_fwdcond_dmvn(x, Ri, Kcxpar);
  
  fwdconditional_mvn(logprior, grad_logprior_par, x); // ***
  neghess_fwdcond_dmvn(neghess_logtarg, x);
  
  double logprior_chi = 0;
  arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
  
  
  for(unsigned int c=0; c<num_children; c++ ){
    //bwdconditional_mvn(logprior_chi, grad_logprior_chi, x, w_child(c), Ri_of_child(c), 
    //                   Kcx_x(c), Kco_wo(c));
    //neghess_bwdcond_dmvn(neghess_logtarg, x, w_child(c), Ri_of_child(c), Kcx_x(c));
    
    // with pointer to cube -- memory leak?
    mvn_dens_grad_neghess(logprior_chi, grad_logprior_chi, neghess_logtarg, x, c);
  }
  
  xtarget = logprior + logprior_chi + loglike;
  
  xgrad = grad_loglike + 
    grad_logprior_par + 
    grad_logprior_chi;
  
  return neghess_logtarg;
  
}



// Gradient of the log posterior
arma::mat NodeDataW::neghess_logfullcondit(const arma::mat& x){
  int q = y.n_cols;
  int k = x.n_cols;
  
  arma::mat neghess_logtarg = arma::zeros(x.n_rows * x.n_cols,
                                  x.n_rows * x.n_cols);
  
  int nr = y.n_rows;
  int indxsize = x.n_rows;

  for(int i=0; i<nr; i++){
    arma::mat wloc = x.row(i);
    for(unsigned int j=0; j<y.n_cols; j++){
      if(na_mat(i, j) > 0){
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        double mult = get_mult(y(i,j), offset(i,j), prob_offset(i,j), xij);
        
        arma::mat LambdaHt = Lambda_lmc.row(j).t() * mult;
        arma::mat neghessloc = LambdaHt * LambdaHt.t();
        
        for(int s1=0; s1<k; s1++){
          for(int s2=0; s2<k; s2++){
            neghess_logtarg(s1 * indxsize + i, s2*indxsize + i) += neghessloc(s1, s2);
          }
        }
      }
    }
  }
  
  neghess_fwdcond_dmvn(neghess_logtarg, x);
  
  //arma::mat neghess_logprior_chi = arma::zeros(arma::size(neghess_logprior_par));
  for(unsigned int c=0; c<num_children; c++ ){
    // adds to neghess_logprior_par
    neghess_bwdcond_dmvn(neghess_logtarg, x, c);
  }
  
  return neghess_logtarg;// + 
    //neghess_logprior_par; // + 
    //neghess_logprior_chi;
}


// Neghess of the log cond prior
arma::mat NodeDataW::neghess_prior(const arma::mat& x){
  
  arma::mat neghess_logtarg = arma::zeros(x.n_rows * x.n_cols,
                                          x.n_rows * x.n_cols);
  
  neghess_fwdcond_dmvn(neghess_logtarg, x);
  
  //arma::mat neghess_logprior_chi = arma::zeros(arma::size(neghess_logprior_par));
  for(unsigned int c=0; c<num_children; c++ ){
    // adds to neghess_logprior_par
    neghess_bwdcond_dmvn(neghess_logtarg, x, c);
  }
  
  return neghess_logtarg;// + 
  //neghess_logprior_par; // + 
  //neghess_logprior_chi;
}


NodeDataBLG::NodeDataBLG(){
  n=-1;
}

NodeDataBLG::NodeDataBLG(const arma::vec& y_in, const arma::mat& X_in, const arma::mat Z_in){
  y = y_in;
  n = y.n_elem;
  X = X_in;
  Z = Z_in;
  p = X.n_cols;
  pg = Z.n_cols;
  
}

void NodeDataBLG::update_mv(const arma::vec& Smu_tot, const arma::mat& Sigi_tot){
  mstar = Smu_tot;
  Vw_i = Sigi_tot;
}
void NodeDataBLG::update_X(const arma::mat& X_in){
  X = X_in;
  p = X.n_cols;
}
// log posterior 
double NodeDataBLG::logfullcondit(const arma::vec& x){
  double loglike = 0;
  
  // x is a vector of size p+pg storing beta and gamma 
  // here, beta includes the loadings of the latent factors
  beta = x.head(p);
  gamma = x.tail(pg);
  
  //Rcpp::Rcout << arma::size(X) << " " << arma::size(Z) << " " << arma::size(beta) << " " << arma::size(gamma) << endl;
  
  lambda = exp(X * beta);
  prob = 1.0 / ( 1 + exp(- Z * gamma));
  
  loglike = 0;
  for(int i=0; i<n; i++){
    loglike += y(i) * (log(lambda(i)) + log(prob(i))) - lambda(i) * prob(i);
  }
  
  double logprior = arma::conv_to<double>::from(
    x.t() * mstar - .5 * x.t() * Vw_i * x);
  
  //Rcpp::Rcout << "ll: " << loglike << " lp: " << logprior << endl;
  //Rcpp::Rcout << "X head \n" << X.head_rows(5) << endl; 
  
  return ( loglike + logprior );
  
}

// Gradient of the log posterior
arma::vec NodeDataBLG::gradient_logfullcondit(const arma::vec& x){
  arma::vec grad_loglike = arma::zeros(x.n_elem);
  
  arma::vec y_lambdaprob = y - lambda % prob;
  grad_loglike.head(p) = X.t() * y_lambdaprob;
  grad_loglike.tail(pg) = Z.t() * ( y_lambdaprob % (1 - prob) );
  
  arma::vec grad_logprior = mstar - Vw_i * x;
  //Rcpp::Rcout << "gll: " << grad_loglike.t() << " glp: " << grad_logprior.t() << endl;
  return grad_loglike + grad_logprior;
}

void NodeDataBLG::set_XtDX(const arma::vec& x){
  arma::vec p1mp = prob % (1 - prob);
  arma::vec mult_b2 = lambda % prob;
  arma::vec mult_bg = lambda % p1mp;
  arma::vec mult_g2 = p1mp % (y - lambda % (2 * prob - 1));
  
  XZm = arma::zeros(p + pg, p + pg);
  
  arma::mat Xtemp_b = X;
  arma::mat Xtemp_bg = X;
  arma::mat Ztemp_g = Z;
  for(unsigned int i=0; i<n; i++){
    Xtemp_b.row(i) *= mult_b2(i);
    Xtemp_bg.row(i) *= mult_bg(i);
    Ztemp_g.row(i) *= mult_g2(i);
  }
  
  XZm.submat(0, 0, p-1, p-1) = Xtemp_b.t() * X;
  XZm.submat(0, p, p-1, p+pg-1) = Xtemp_bg.t() * Z;
  XZm.submat(p, 0, p+pg-1, p-1) = arma::trans(XZm.submat(0, p, p-1, p+pg-1));
  XZm.submat(p, p, p+pg-1, p+pg-1) = Ztemp_g.t() * Z;
  
  //Rcpp::Rcout << "nh: \n" << XZm << endl;
  //if(XZm.has_nan()){
  //  Rcpp::stop("XZm has nans!\n");
  //}
}

arma::mat NodeDataBLG::neghess_logfullcondit(const arma::vec& x){
  set_XtDX(x);
  return XZm + Vw_i;
}

arma::vec NodeDataBLG::compute_dens_and_grad(double& xtarget, const arma::mat& x){
  xtarget = logfullcondit(x);
  return gradient_logfullcondit(x);
}

arma::mat NodeDataBLG::compute_dens_grad_neghess(double& xtarget, arma::vec& xgrad, const arma::mat& x){
  xtarget = logfullcondit(x);
  xgrad = gradient_logfullcondit(x);
  return neghess_logfullcondit(x);
}
