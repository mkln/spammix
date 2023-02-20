#include "meshed.h"

using namespace std;

void MeshedMNM::deal_with_gamma(){
  sample_hmc_gamma();
}

void MeshedMNM::sample_hmc_gamma(){
  if(verbose & debug){
    Rcpp::Rcout << "[sample_hmc_gamma]\n";
  }
  // choose between NUTS
  start = std::chrono::steady_clock::now();
  
  arma::mat gmat_rnd = mrstdnorm(pg, q);
  arma::vec bunifv = vrunif(q);
  arma::vec bunifv2 = vrunif(q);
  
  for(unsigned int j=0; j<q; j++){
    
    arma::vec offsets_for_gamma = arma::zeros(ix_by_q_a(j).n_elem);
    gamma_node.at(j).update_mv(offsets_for_gamma, Gim, Gi);
    
    arma::vec n_pop_obs = Npop( ix_by_q_a(j), oneuv * j );
    gamma_node.at(j).update_nbin(n_pop_obs);
  
    // nongaussian
    gamma_hmc_adapt.at(j).step();
    if(gamma_hmc_started(j) == 0){
      // wait a few iterations before starting adaptation
      //Rcpp::Rcout << "reasonable stepsize " << endl;
      double gamma_eps = find_reasonable_stepsize(gamma.col(j), gamma_node.at(j), gmat_rnd.cols(oneuv * j));
      //Rcpp::Rcout << "adapting scheme starting " << endl;
      AdaptE new_adapting_scheme;
      new_adapting_scheme.init(gamma_eps, pg, w_hmc_srm, w_hmc_nuts, 1e4);
      gamma_hmc_adapt.at(j) = new_adapting_scheme;
      gamma_hmc_started(j) = 1;
      //Rcpp::Rcout << "done initiating adapting scheme" << endl;
    }
    
    arma::vec sampled;
    if(which_hmc == 0){
      // some form of manifold mala
      sampled = simpa_cpp(gamma.col(j), gamma_node.at(j), gamma_hmc_adapt.at(j), 
                          gmat_rnd.cols(oneuv * j), bunifv(j), bunifv2(j), 
                          true, debug);
    }
    if(which_hmc == 1){
      // mala
      sampled = mala_cpp(gamma.col(j), gamma_node.at(j), gamma_hmc_adapt.at(j), 
                         gmat_rnd.cols(oneuv * j), bunifv(j), true, debug);
    }
    if(which_hmc == 2){
      // nuts
      sampled = nuts_cpp(gamma.col(j), gamma_node.at(j), gamma_hmc_adapt.at(j)); 
    }
    
    if((which_hmc == 3) || (which_hmc == 4)){
      // some form of manifold mala
      sampled = manifmala_cpp(gamma.col(j), gamma_node.at(j), gamma_hmc_adapt.at(j), 
                              gmat_rnd.cols(oneuv * j), bunifv(j), bunifv2(j), 
                              true, debug);
    }
    if(which_hmc == 6){
      sampled = hmc_cpp(gamma.col(j), gamma_node.at(j), gamma_hmc_adapt.at(j), 
                        gmat_rnd.cols(oneuv * j), bunifv(j), 0.1, true, debug);
    }
    
    gamma.col(j) = sampled;
    Zg.col(j) = Z * gamma.col(j);
  }
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[sample_hmc_gamma] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}


