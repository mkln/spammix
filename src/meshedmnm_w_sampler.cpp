#include "meshedmnm.h"

using namespace std;

void MeshedMNM::deal_with_w(MeshDataLMC& data){
  // ensure this is independent on the number of threads being used
  Rcpp::RNGScope scope;
  rand_norm_mat = mrstdnorm(coords.n_rows, k);
  rand_unif = vrunif(n_blocks);
  rand_unif2 = vrunif(n_blocks);

  nongaussian_w(data);
}

void MeshedMNM::nongaussian_w(MeshDataLMC& data){
  if(verbose & debug){
    Rcpp::Rcout << "[hmc_sample_w] " << endl;
  }
  
  start_overall = std::chrono::steady_clock::now();
  
  
  int mala_timer = 0;
  
  arma::mat offset_for_w = offsets + XB;
  arma::mat prob_offset = 1.0/(1.0 + exp(-Zg));
  
  for(int g=0; g<n_gibbs_groups; g++){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(unsigned int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      if((block_ct_obs(u) > 0)){
        
        start = std::chrono::steady_clock::now();
        //Rcpp::Rcout << "u :  " << u << endl;
        w_node.at(u).update_mv(offset_for_w, prob_offset, Lambda);
        
        if(parents(u).n_elem > 0){
          //w_node.at(u).Kxxi = (*data.w_cond_prec_parents_ptr.at(u));
          w_node.at(u).Kcx = data.w_cond_mean_K_ptr.at(u);
          //w_node.at(u).w_parents = w.rows(parents_indexing(u));
        }
        
        //message("step 3");
        w_node.at(u).Ri = data.w_cond_prec_ptr.at(u);
        
        if(parents_indexing(u).n_rows > 0){
          w_node.at(u).Kcxpar = arma::zeros((*w_node.at(u).Kcx).n_rows, k);
          for(unsigned int j=0; j<k; j++){
            w_node.at(u).Kcxpar.col(j) = (*w_node.at(u).Kcx).slice(j) * w.submat(parents_indexing(u), oneuv*j);//w_node.at(u).w_parents.col(j);
          }
        } else {
          w_node.at(u).Kcxpar = arma::zeros(0,0);
        }
          
        for(unsigned int c=0; c<w_node.at(u).num_children; c++ ){
          int child = children(u)(c);
          //Rcpp::Rcout << "child [" << child << "]\n";
          arma::uvec c_ix = indexing(child);
          arma::uvec pofc_ix = parents_indexing(child);
          
          arma::uvec pofc_ix_x = u_is_which_col_f(u)(c)(0);
          arma::uvec pofc_ix_other = u_is_which_col_f(u)(c)(1);
          
          arma::mat w_childs_parents = w.rows(pofc_ix);
          arma::mat w_otherparents = w_childs_parents.rows(pofc_ix_other);
          
          w_node.at(u).w_child(c) = w.rows(c_ix);
          w_node.at(u).Ri_of_child(c) = data.w_cond_prec_ptr.at(child);
          
          for(unsigned int r=0; r<k; r++){
            w_node.at(u).Kcx_x(c).slice(r) = (*data.w_cond_mean_K_ptr.at(child)).slice(r).cols(pofc_ix_x);
          }
          
          //Rcpp::Rcout << "hmc_sample_w " << arma::size(w_node.at(u).Kcx_x(c)) << endl;
          
          if(w_otherparents.n_rows > 0){
            //arma::cube Kcx_other = //(*param_data.w_cond_mean_K.at(child)).cols(pofc_ix_other);
            //  cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), pofc_ix_other);
            
            w_node.at(u).Kco_wo(c) = arma::zeros(c_ix.n_elem, k); //cube_times_mat(Kcx_other, w_otherparents);
            for(unsigned int r=0; r<k; r++){
              w_node.at(u).Kco_wo(c).col(r) = (*data.w_cond_mean_K_ptr.at(child)).slice(r).cols(pofc_ix_other) * w_otherparents.col(r);
            }
            
          } else {
            w_node.at(u).Kco_wo(c) = arma::zeros(0,0);
          }
          //Rcpp::Rcout << "child done " << endl;
        } 
        
        // -----------------------------------------------
          
        arma::mat w_current = w.rows(indexing(u));
        
        
        // adapting eps
        hmc_eps_adapt.at(u).step();
        
        if((hmc_eps_started_adapting(u) == 0) && (hmc_eps_adapt.at(u).i==10)){
          // wait a few iterations before starting adaptation
          //message("find reasonable");
          hmc_eps(u) = find_reasonable_stepsize(w_current, w_node.at(u), rand_norm_mat.rows(indexing(u)));
          //message("found reasonable");
          int blocksize = indexing(u).n_elem * k;
          AdaptE new_adapting_scheme;
          new_adapting_scheme.init(hmc_eps(u), blocksize, w_hmc_srm, w_hmc_nuts, 1e4);
          
          hmc_eps_adapt.at(u) = new_adapting_scheme;
          hmc_eps_started_adapting(u) = 1;
        }
        
        arma::mat w_temp = w_current;
         
        if(which_hmc == 0){
          w_temp = simpa_cpp(w_current, w_node.at(u), hmc_eps_adapt.at(u),
                                 rand_norm_mat.rows(indexing(u)),
                                 rand_unif(u), rand_unif2(u),
                                 true, debug);
        }
        if(which_hmc == 1){
          // mala
          w_temp = mala_cpp(w_current, w_node.at(u), hmc_eps_adapt.at(u),
                                       rand_norm_mat.rows(indexing(u)),
                                       rand_unif(u), true, debug);
        }
        if(which_hmc == 2){
          // nuts
          w_temp = nuts_cpp(w_current, w_node.at(u), hmc_eps_adapt.at(u)); 
        }
        if((which_hmc == 3) || (which_hmc == 4)){
          // some form of manifold mala
          w_temp = manifmala_cpp(w_current, w_node.at(u), hmc_eps_adapt.at(u),
                                       rand_norm_mat.rows(indexing(u)),
                                       rand_unif(u), rand_unif2(u),
                                       true, debug);
        }
        if(which_hmc == 5){
          w_temp = ellipt_slice_sampler(w_current, w_node.at(u), hmc_eps_adapt.at(u),
                                        rand_norm_mat.rows(indexing(u)),
                                        rand_unif(u), rand_unif2(u),
                                        true, debug);
        }
        if(which_hmc == 6){
          w_temp = hmc_cpp(w_current, w_node.at(u), hmc_eps_adapt.at(u),
                           rand_norm_mat.rows(indexing(u)),
                           rand_unif(u), 0.1, true, debug);
        }

        
        end = std::chrono::steady_clock::now();
        
        hmc_eps(u) = hmc_eps_adapt.at(u).eps;
        
        w.rows(indexing(u)) = w_temp;//arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));

        mala_timer += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //Rcpp::Rcout << "done sampling "<< endl;

      }
    }
  }
  
  LambdaHw = w * Lambda.t();
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[hmc_sample_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
    //Rcpp::Rcout << "mala: " << mala_timer << endl;
  }
  
}


void MeshedMNM::predict(){
  start_overall = std::chrono::steady_clock::now();
  if(predict_group_exists == 1){
    if(verbose & debug){
      Rcpp::Rcout << "[predict] start \n";
    }
    
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(unsigned int i=0; i<u_predicts.n_elem; i++){ //*** subset to blocks with NA
      int u = u_predicts(i);// u_predicts(i);
      // only predictions at this block. 
      arma::uvec predict_parent_indexing, cx;
      arma::cube Kxxi_parents;

      // no observed locations, use line of sight
      predict_parent_indexing = parents_indexing(u);
      CviaKron_HRj_chol_bdiag(Hpred(i), Rcholpred(i), Kxxi_parents,
                              na_1_blocks(u),
                              coords, indexing_obs(u), predict_parent_indexing, k, param_data.theta, matern);
    
      
      //Rcpp::Rcout << "step 1 "<< endl;
      arma::mat wpars = w.rows(predict_parent_indexing);
      
      for(unsigned int ix=0; ix<indexing_obs(u).n_elem; ix++){
        if(na_1_blocks(u)(ix) == 0){
          arma::rowvec wtemp = arma::sum(arma::trans(Hpred(i).slice(ix)) % wpars, 0);
          
          wtemp += arma::trans(Rcholpred(i).col(ix)) % rand_norm_mat.row(indexing_obs(u)(ix));
          
          
          w.row(indexing_obs(u)(ix)) = wtemp;
          
          LambdaHw.row(indexing_obs(u)(ix)) = w.row(indexing_obs(u)(ix)) * Lambda.t();
        }
      }
      
      //Rcpp::Rcout << "done" << endl;
      
      
    }
    
    if(verbose & debug){
      end_overall = std::chrono::steady_clock::now();
      Rcpp::Rcout << "[predict] "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                  << "us. ";
    }
  }
}

void MeshedMNM::predicty(){ 
  if(verbose & debug){
    Rcpp::Rcout << "[predicty] start" << endl;
  }
  Rcpp::RNGScope scope;
  for(unsigned int j=0; j<q; j++){
    yhat.col(j) = vrpois(exp(XB.col(j) + LambdaHw.col(j))/(1.0+exp(-Zg.col(j))));
  }
  if(verbose & debug){
    Rcpp::Rcout << "[predicty] end" << endl;
  }
}



// use ncurrent and metropolis a/r
int sample_one_N(int y, double lambda, double p){
  return y + R::rpois(lambda * (1-p));
}

void MeshedMNM::deal_with_N(){
  if(verbose & debug){
    Rcpp::Rcout << "[deal_with_N] start" << endl;
  }
  for(unsigned int j=0; j<q; j++){
    for(unsigned int i=0; i<n; i++){
      if(!std::isnan(y(i, j))){
        Npop(i, j) = y(i,j) + R::rpois( exp(XB(i, j) + LambdaHw(i, j)) * (1-1.0/(1.0+exp(-Zg(i, j)))) );
      } else {
        Npop(i, j) = R::rpois( exp(XB(i, j) + LambdaHw(i, j)) );
      }
      
    }
  }
  if(verbose & debug){
    Rcpp::Rcout << "[deal_with_N] end \n";
  }
}