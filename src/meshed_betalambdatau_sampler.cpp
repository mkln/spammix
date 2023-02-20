#include "meshed.h"

using namespace std;

void MeshedMNM::deal_with_BetaLambda(bool sample_beta, bool sample_lambda){
  sample_hmc_BetaLambda(sample_beta, sample_lambda);
}


void MeshedMNM::sample_hmc_BetaLambda(bool sample_beta, bool sample_lambda){
  if(verbose & debug){
    Rcpp::Rcout << "[sample_hmc_BetaLambdaTau] starting\n";
  }
  start = std::chrono::steady_clock::now();
  
  Rcpp::RNGScope scope;
  arma::mat rnorm_precalc = mrstdnorm(q, k+p);
  arma::vec lambda_runif = vrunif(q);
  arma::vec lambda_runif2 = vrunif(q);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int j=0; j<q; j++){
    
    ///
    /// ** Beta & Lambda update **
    ///
    arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
    arma::vec y_for_pois = Npop(ix_by_q_a(j), oneuv * j);
    arma::vec offsets_obs = offsets(ix_by_q_a(j), oneuv * j);
    
    // build W
    // filter: choose value of spatial processes at locations of Yj that are available
    arma::mat WWj = w.submat(ix_by_q_a(j), subcols); // acts as X //*********
    arma::mat XW = arma::join_horiz(X.rows(ix_by_q_a(j)), WWj);

    arma::mat BL_Vi = arma::eye( XW.n_cols, XW.n_cols );
    BL_Vi.submat(0, 0, p-1, p-1) = Vi; // prior precision for beta
    arma::vec BL_Vim = arma::zeros(XW.n_cols);
    
    lambda_node.at(j).update_mv(offsets_obs, BL_Vim, BL_Vi);
    lambda_node.at(j).update_y( y_for_pois );
    lambda_node.at(j).X = XW;
    
    arma::vec curLrow = arma::join_vert( Bcoeff.col(j),
      arma::trans(Lambda.submat(oneuv*j, subcols)));
    arma::mat rnorm_row = arma::trans(rnorm_precalc.row(j).head(curLrow.n_elem));
    
    arma::vec sampled;
  
    // nongaussian
    //Rcpp::Rcout << "step " << endl;
    lambda_hmc_adapt.at(j).step();
    if((lambda_hmc_started(j) == 0) && (lambda_hmc_adapt.at(j).i == 10)){
      // wait a few iterations before starting adaptation
      //Rcpp::Rcout << "reasonable stepsize " << endl;
      
      double lambda_eps = find_reasonable_stepsize(curLrow, lambda_node.at(j), rnorm_row);
      
      int n_params = curLrow.n_elem;
      AdaptE new_adapting_scheme;
      new_adapting_scheme.init(lambda_eps, n_params, w_hmc_srm, w_hmc_nuts, 1e4);
      lambda_hmc_adapt.at(j) = new_adapting_scheme;
      lambda_hmc_started(j) = 1;
      //Rcpp::Rcout << "done initiating adapting scheme" << endl;
    }
    if(which_hmc == 0){
      // some form of manifold mala
      sampled = simpa_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j), 
                              rnorm_row, lambda_runif(j), lambda_runif2(j), 
                              true, debug);
    }
    if(which_hmc == 1){
      // mala
      sampled = mala_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j), 
                         rnorm_row, lambda_runif(j), true, debug);
    }
    if(which_hmc == 2){
      // nuts
      sampled = nuts_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j)); 
    }
    
    if((which_hmc == 3) || (which_hmc == 4)){
      // some form of manifold mala
      sampled = manifmala_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j), 
                              rnorm_row, lambda_runif(j), lambda_runif2(j), 
                              true, debug);
    }
    if(which_hmc == 6){
      sampled = hmc_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j), 
                        rnorm_row, lambda_runif(j), 0.1, true, debug);
    }
    
      
    if(sample_beta){
      Bcoeff.col(j) = sampled.head(p);
    } 
    if(sample_lambda){
      Lambda.submat(oneuv*j, subcols) = arma::trans(sampled.tail(subcols.n_elem));
    }
  
    XB.col(j) = X * Bcoeff.col(j);
    LambdaHw.col(j) = w * arma::trans(Lambda.row(j));
  }
  // refreshing density happens in the 'logpost_refresh_after_gibbs' function
  if(verbose & debug){
    Rcpp::Rcout << "[sample_hmc_Lambda] done\n";
  }
  
}
