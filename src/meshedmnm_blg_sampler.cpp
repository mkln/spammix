#include "meshedmnm.h"

using namespace std;

void MeshedMNM::deal_with_BetaLambdaGamma(bool sample_beta, bool sample_lambda, bool sample_gamma){
  sample_hmc_BetaLambdaGamma(sample_beta, sample_lambda, sample_gamma);
}


void MeshedMNM::sample_hmc_BetaLambdaGamma(bool sample_beta, bool sample_lambda, bool sample_gamma){
  if(verbose & debug){
    Rcpp::Rcout << "[sample_hmc_BetaLambdaGamma] starting\n";
  }
  start = std::chrono::steady_clock::now();
  
  Rcpp::RNGScope scope;
  arma::mat rnorm_precalc = mrstdnorm(q, k+p+pg);
  arma::vec blg_runif = vrunif(q);
  arma::vec blg_runif2 = vrunif(q);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int j=0; j<q; j++){
    
    ///
    /// ** Beta & Lambda update **
    ///
    arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
    arma::vec y_obs = y(ix_by_q_a(j), oneuv * j);
    //arma::vec offsets_obs = offsets(ix_by_q_a(j), oneuv * j);
    
    // build W
    // filter: choose value of spatial processes at locations of Yj that are available
    arma::mat WWj = w.submat(ix_by_q_a(j), subcols); // acts as X //*********
    int pl = WWj.n_cols;
    arma::mat XW = arma::join_horiz(X.rows(ix_by_q_a(j)), WWj);

    arma::mat BL_Vi = arma::eye( XW.n_cols + pg, XW.n_cols + pg);
    BL_Vi.submat(0, 0, p-1, p-1) = Vi; // prior precision for beta
    BL_Vi.submat(p+pl, p+pl, p+pl+pg-1, p+pl+pg-1) = Gi;
    
    arma::vec BL_Vim = arma::zeros(XW.n_cols + pg);
    
    blg_node.at(j).update_mv(BL_Vim, BL_Vi);
    blg_node.at(j).update_X(XW);
    
    arma::vec curLrow = arma::join_vert( Bcoeff.col(j),
      arma::trans(Lambda.submat(oneuv*j, subcols)) );
    curLrow = arma::join_vert(curLrow, gamma.col(j) );
    
    arma::mat rnorm_row = arma::trans(rnorm_precalc.row(j).head(curLrow.n_elem));

    // not used for now
    arma::vec sample_mask = arma::ones(p + pl + pg);
    if(!sample_beta){
      sample_mask.subvec(0, p-1).fill(0);
    }
    if(!sample_lambda){
      sample_mask.subvec(p, p+pl-1).fill(0);
    }
    if(!sample_gamma){
      sample_mask.subvec(p+pl, p+pl+pg-1).fill(0);
    }
    // --
    
    arma::vec sampled;
    // nongaussian
    //Rcpp::Rcout << "step " << endl;
    blg_hmc_adapt.at(j).step();
    if((blg_hmc_started(j) == 0) && (blg_hmc_adapt.at(j).i == 10)){
      // wait a few iterations before starting adaptation
      //Rcpp::Rcout << "reasonable stepsize " << endl;
      
      double blg_eps = find_reasonable_stepsize(curLrow, blg_node.at(j), rnorm_row);
      
      int n_params = curLrow.n_elem;
      AdaptE new_adapting_scheme;
      new_adapting_scheme.init(blg_eps, n_params, w_hmc_srm, w_hmc_nuts, 1e4);
      blg_hmc_adapt.at(j) = new_adapting_scheme;
      blg_hmc_started(j) = 1;
      //Rcpp::Rcout << "done initiating adapting scheme" << endl;
    }
    if(which_hmc == 0){
      // some form of manifold mala
      sampled = simpa_cpp(curLrow, blg_node.at(j), blg_hmc_adapt.at(j), 
                              rnorm_row, blg_runif(j), blg_runif2(j), 
                              //sample_mask, sample_beta+sample_lambda+sample_gamma, 
                              true, debug);
    }
    if(which_hmc == 1){
      // mala
      sampled = mala_cpp(curLrow, blg_node.at(j), blg_hmc_adapt.at(j), 
                         rnorm_row, blg_runif(j), true, debug);
    }
    if(which_hmc == 2){
      // nuts
      sampled = nuts_cpp(curLrow, blg_node.at(j), blg_hmc_adapt.at(j)); 
    }
    
    if((which_hmc == 3) || (which_hmc == 4)){
      // some form of manifold mala
      sampled = manifmala_cpp(curLrow, blg_node.at(j), blg_hmc_adapt.at(j), 
                              rnorm_row, blg_runif(j), blg_runif2(j), 
                              true, debug);
    }
    if(which_hmc == 6){
      sampled = hmc_cpp(curLrow, blg_node.at(j), blg_hmc_adapt.at(j), 
                        rnorm_row, blg_runif(j), 0.1, true, debug);
    }
    
      
    if(sample_beta){
      
    } 
    if(sample_lambda){
      
    }
    if(sample_gamma){
      
    }
    
    Bcoeff.col(j) = sampled.head(p);
    Lambda.submat(oneuv*j, subcols) = arma::trans(sampled.subvec(p, p+pl-1));
    gamma.col(j) = sampled.tail(pg);
  
    XB.col(j) = X * Bcoeff.col(j);
    LambdaHw.col(j) = w * arma::trans(Lambda.row(j));
    Zg.col(j) = Z * gamma.col(j);
  }
  // refreshing density happens in the 'logpost_refresh_after_gibbs' function
  if(verbose & debug){
    Rcpp::Rcout << "[sample_hmc_BetaLambdaGamma] done\n";
  }
  
}
