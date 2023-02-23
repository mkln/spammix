#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>

#include "distrib_densities_gradients.h"

class NodeData {
public:
  // common stuff
  // latent process type
  //std::string latent;
  arma::mat y; // output data with nan
  
  //arma::umat ystar; // for binomial 
  
  arma::mat offset; // offset for this update
  int n;
  
  double logfullcondit(const arma::vec& x);
  arma::vec gradient_logfullcondit(const arma::vec& x);
  
  NodeData();
  
};


class NodeDataW : public NodeData {
public:
  
  int k;
  //arma::vec z;
  
  arma::mat Lambda_lmc;

  arma::umat na_mat;
  
  int block_ct_obs; // number of not-na
  
  arma::uvec indexing_target;
  
  arma::mat prob_offset;
  
  //arma::cube Kxxi;
  arma::cube * Kcx;
  arma::cube * Ri;
  arma::cube * Hproject;
  
  //arma::vec parKxxpar;
  arma::mat Kcxpar;
  //arma::mat w_parents;
  
  unsigned int num_children;
  double parents_dim;
  //arma::vec dim_of_pars_of_children;
  
  arma::field<arma::cube> Kcx_x;//(c) = (*param_data).w_cond_mean_K(child).cols(pofc_ix_x);
  //arma::field<arma::cube> Kxxi_x;//(c) = (*param_data).w_cond_prec_parents(child)(pofc_ix_x, pofc_ix_x);
  arma::field<arma::mat> w_child;//(c) = arma::vectorise( (*w_full).rows(c_ix) );
  arma::field<arma::cube *> Ri_of_child;//(c) = (*param_data).w_cond_prec(child);
  //arma::field<arma::mat> Kxo_wo;//(c) = Kxxi_xo * w_otherparents;
  arma::field<arma::mat> Kco_wo;//(c) = Kcx_other * w_otherparents;
  
  void update_mv(const arma::mat& new_offset, 
                 const arma::mat& new_prob_offset,
                 const arma::mat& Lambda_lmc_in);

  
  
  double fwdcond_dmvn(const arma::mat& x, 
                      const arma::cube* Ri,
                      const arma::mat& Kcxpar);
  arma::vec grad_fwdcond_dmvn(const arma::mat& x);
  
  void fwdconditional_mvn(double& logtarget, arma::vec& gradient, 
                          const arma::mat& x);
  
  double bwdcond_dmvn(const arma::mat& x, int c);
  
  arma::vec grad_bwdcond_dmvn(const arma::mat& x, int c);
  void bwdconditional_mvn(double& xtarget, arma::vec& gradient, const arma::mat& x, int c);
  
  void neghess_fwdcond_dmvn(arma::mat& result, const arma::mat& x);
  void neghess_bwdcond_dmvn(arma::mat& result, const arma::mat& x, int c);
  void mvn_dens_grad_neghess(double& xtarget, arma::vec& gradient, arma::mat& neghess,
                             const arma::mat& x, int c);
  
  // **
  double logfullcondit(const arma::mat& x);
  double loglike(const arma::mat& x);
  
  arma::vec gradient_logfullcondit(const arma::mat& x);
  arma::mat neghess_logfullcondit(const arma::mat& x);
  arma::mat neghess_prior(const arma::mat& x);
  
  arma::vec compute_dens_and_grad(double& xtarget, const arma::mat& x);
  arma::mat compute_dens_grad_neghess(double& xtarget, arma::vec& xgrad, const arma::mat& x);
  //arma::mat compute_dens_grad_neghess2(double& xtarget, arma::vec& xgrad, const arma::mat& x);
  
  NodeDataW(const arma::mat& y_all, //const arma::mat& Z_in,
            const arma::umat& na_mat_all, const arma::mat& offset_all, 
            const arma::uvec& indexing_target,
            int k);
  
  NodeDataW();
  
};


class NodeDataBLG : public NodeData{
public:
  int n, p, pg;
  
  arma::mat X, Z;
  arma::mat XZm; //XtX, ZtZ, XtZ;
  
  arma::vec beta, gamma;
  arma::vec lambda, prob;
  
  arma::vec mstar;
  arma::mat Vw_i;
  
  void update_mv(const arma::vec& Smu_tot, const arma::mat& Sigi_tot);
  void update_X(const arma::mat& X_in);
  void set_XtDX(const arma::vec& x);
  
  NodeDataBLG(const arma::vec& y_in, const arma::mat& X_in, const arma::mat Z_in);
  NodeDataBLG();
  
  double logfullcondit(const arma::vec& x);
  arma::vec gradient_logfullcondit(const arma::vec& x);
  arma::mat neghess_logfullcondit(const arma::vec& x);
  
  // used in sampling
  arma::vec compute_dens_and_grad(double& xtarget, const arma::mat& x);
  arma::mat compute_dens_grad_neghess(double& xtarget, arma::vec& xgrad, const arma::mat& x);
  
};

