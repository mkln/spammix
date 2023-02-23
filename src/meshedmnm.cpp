#include "meshedmnm.h"
using namespace std;


MeshedMNM::MeshedMNM(
  const arma::mat& y_in, 
  const arma::mat& X_in, 
  const arma::mat& Z_in,
  
  const arma::mat& coords_in, 
  
  int k_in,
  
  const arma::field<arma::uvec>& parents_in,
  const arma::field<arma::uvec>& children_in,
  
  const arma::vec& block_names_in,
  const arma::vec& block_groups_in,
  
  const arma::field<arma::uvec>& indexing_in,
  const arma::field<arma::uvec>& indexing_obs_in,
  
  int matern_twonu_in,
  
  const arma::mat& w_in,
  
  const arma::mat& beta_in,
  const arma::mat& gamma_in,
  
  const arma::mat& lambda_in,
  const arma::umat& lambda_mask_in,
  const arma::mat& theta_in,
  
  const arma::mat& beta_Vi_in,
  const arma::mat& gamma_Gi_in,
  
  int which_hmc_in,
  bool adapting_theta,
  const arma::mat& metrop_theta_sd,
  const arma::mat& metrop_theta_bounds,
  
  bool use_cache=true,
  
  bool use_ps=true,
  
  bool verbose_in=false,
  bool debugging=false,
  int num_threads = 1){
  
  oneuv = arma::ones<arma::uvec>(1);//utils
  hl2pi = -.5 * log(2.0 * M_PI);
  
  verbose = verbose_in;
  debug = debugging;
  
  cached = use_cache;
  
  if(verbose & debug){
    Rcpp::Rcout << "MeshedMNM::MeshedMNM initialization.\n";
  }
  
  // data
  y = y_in;
  
  offsets = arma::zeros(arma::size(y));
  
  X = X_in;
  Z = Z_in;
  
  na_mat = arma::zeros<arma::umat>(arma::size(y));
  na_mat.elem(arma::find_finite(y)).fill(1);
  
  p  = X.n_cols;
  pg = Z.n_cols;
  
  // spatial coordinates and dimension
  coords = coords_in;
  dd = coords.n_cols;
  q = y.n_cols;
  k = k_in;
  
  Lambda = lambda_in; 
  Lambda_mask = lambda_mask_in;
  
  // NAs at blocks of outcome variables 
  ix_by_q_a = arma::field<arma::uvec>(q);
  for(unsigned int j=0; j<q; j++){
    ix_by_q_a(j) = arma::find_finite(y.col(j));
    if(verbose){
      Rcpp::Rcout << "Y(" << j+1 << ") : " << ix_by_q_a(j).n_elem << " observed locations.\n";
    }
  }
  
  // DAG
  parents             = parents_in;
  children            = children_in;
  block_names         = block_names_in;
  block_groups        = block_groups_in;
  block_groups_labels = arma::unique(block_groups);
  n_gibbs_groups      = block_groups_labels.n_elem;
  n_blocks            = block_names.n_elem;
  
  // domain partitioning
  indexing    = indexing_in;
  indexing_obs = indexing_obs_in;
  
  // initial values
  w = w_in; 
  
  if(verbose & debug){
    Rcpp::Rcout << "Lambda size: " << arma::size(Lambda) << "\n";
  }
  
  XB = arma::zeros(coords.n_rows, q);
  Zg = arma::zeros(coords.n_rows, q);
  linear_predictor = arma::zeros(coords.n_rows, q);
  
  Bcoeff = beta_in; 
  gamma = gamma_in;
  for(unsigned int j=0; j<q; j++){
    XB.col(j) = X * Bcoeff.col(j);
    Zg.col(j) = Z * gamma.col(j);
  }
  
  Npop = y;
  for(int i=0; i<n; i++){
    for(int j=0; j<q; j++){
      if(std::isnan(y(i,j))){
        Npop(i, j) = R::rpois( exp(XB(i,j)) );
      } 
    }
  }
  
  Nmax = arma::trans(arma::max(Npop)) * 2;
  if(Nmax.n_elem != q){
    Rcpp::stop("wrong dim\n");
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "Beta size: " << arma::size(Bcoeff) << "\n"; 
  }
  
  // prior params
  Vi    = beta_Vi_in;
  bprim = arma::zeros(p);
  Vim   = Vi * bprim;
  
  // prior params
  Gi    = gamma_Gi_in;
  gprim = arma::zeros(pg);
  Gim   = Gi * gprim;
  
  // init
  u_is_which_col_f    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  
  predicting = true;
  
  // now elaborate
  init_indexing();
  na_study();
  // now we know where NAs are, we can erase them
  //y.elem(arma::find_nonfinite(y)).fill(0);
  n = y.n_rows;
  yhat = arma::zeros(n, q);
  
  init_gibbs_index();
  make_gibbs_groups();
  init_cache();
  
  init_meshdata(theta_in);
  
  // RAMA for theta
  theta_mcmc_counter = 0;
  theta_unif_bounds = metrop_theta_bounds;
  theta_metrop_sd = metrop_theta_sd;
  theta_adapt = RAMAdapt(theta_in.n_elem, theta_metrop_sd, 0.24);
  theta_adapt_active = adapting_theta;
  
  init_matern(num_threads, matern_twonu_in, use_ps);
  
  LambdaHw = w * Lambda.t(); // arma::zeros(coords.n_rows, q); 

  rand_norm_mat = arma::zeros(coords.n_rows, k);
  rand_unif = arma::zeros(n_blocks);

  
  which_hmc = which_hmc_in;
  init_for_mcmc();
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "MeshedMNM::MeshedMNM initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
}


void MeshedMNM::make_gibbs_groups(){
  if(verbose & debug){
    Rcpp::Rcout << "[make_gibbs_groups] start\n";
  }
  
  // checks -- errors not allowed. use check_groups.cpp to fix errors.
  for(int g=0; g<n_gibbs_groups; g++){
    for(unsigned int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(block_groups(u) == block_groups_labels(g)){
        if(indexing(u).n_elem > 0){ //**
          
          for(unsigned int pp=0; pp<parents(u).n_elem; pp++){
            if(block_groups(parents(u)(pp)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " <--- " << parents(u)(pp) 
                          << ": same group (" << block_groups(u) 
                          << ")." << "\n";
              Rcpp::stop("Invalid coloring of the DAG.\n");
            }
          }
          for(unsigned int cc=0; cc<children(u).n_elem; cc++){
            if(block_groups(children(u)(cc)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " ---> " << children(u)(cc) 
                          << ": same group (" << block_groups(u) 
                          << ")." << "\n";
              Rcpp::stop("Invalid coloring of the DAG.\n");
            }
          }
        }
      }
    }
  }
  
  int gx=0;
  arma::field<arma::vec> u_by_block_groups_temp(n_gibbs_groups);
  u_by_block_groups = arma::field<arma::vec>(n_gibbs_groups);
  /// create list of groups for gibbs
  
  for(int g=0; g<n_gibbs_groups; g++){
    u_by_block_groups_temp(g) = arma::zeros(0);
    
    for(unsigned int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      
      if(block_groups(u) == block_groups_labels(g)){
        if(block_ct_obs(u) > 0){ //**
          arma::vec uhere = arma::zeros(1) + u;
          u_by_block_groups_temp(g) = arma::join_vert(u_by_block_groups_temp(g), uhere);
        } 
      }
    }
    if(u_by_block_groups_temp(g).n_elem > 0){
      u_by_block_groups(gx) = u_by_block_groups_temp(g);
      gx ++;
    }
  }
  
  int pblocks = 0;
  for(unsigned int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    // original grid, then predict blocks are the empty ones
    if(block_ct_obs(u) == 0){
      pblocks ++;
    }
  }
  
  if(pblocks > 0){
    u_predicts = arma::zeros<arma::uvec>(pblocks);
    predict_group_exists = 1;
  } else {
    predict_group_exists = 0;
  }
  
  if(predict_group_exists == 1){
    int p=0; 
    for(unsigned int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      // original grid, then predict blocks are the empty ones
      if(block_ct_obs(u) == 0){
        u_predicts(p) = u;
        p ++;
      }
      
    }
  } else {
    if(verbose & debug){
      Rcpp::Rcout << "No prediction group " << endl;
    }
  }
  
  // predict_initialize
  if(verbose & debug){
    Rcpp::Rcout << "predict initialize \n";
  }
  if(predict_group_exists == 1){
    Hpred = arma::field<arma::cube>(u_predicts.n_elem);
    Rcholpred = arma::field<arma::mat>(u_predicts.n_elem);
    
    for(unsigned int i=0; i<u_predicts.n_elem; i++){
      int u = u_predicts(i);
      if(block_ct_obs(u) > 0){
        Hpred(i) = arma::zeros(k,indexing(u).n_elem,indexing_obs(u).n_elem);
      } else {
        Hpred(i) = arma::zeros(k,parents_indexing(u).n_elem,indexing_obs(u).n_elem);
      }
      Rcholpred(i) = arma::zeros(k,indexing_obs(u).n_elem);
    }
  }
  if(verbose & debug){
    Rcpp::Rcout << "[make_gibbs_groups] done.\n";
  }
}

void MeshedMNM::na_study(){
  // prepare stuff for NA management
  if(verbose & debug){
    Rcpp::Rcout << "[na_study] start \n"; 
  }
  na_1_blocks = arma::field<arma::uvec> (n_blocks);
  na_0_blocks = arma::field<arma::uvec> (n_blocks);
  na_ix_blocks = arma::field<arma::uvec> (n_blocks);
  n_loc_ne_blocks = 0;
  block_ct_obs = arma::zeros(n_blocks);
  
  if(verbose & debug){
    Rcpp::Rcout << "[na_study] step 1.\n";
  }
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<n_blocks;i++){
    arma::mat yvec = y.rows(indexing_obs(i));
    na_1_blocks(i) = arma::zeros<arma::uvec>(yvec.n_rows);
    na_0_blocks(i) = arma::zeros<arma::uvec>(yvec.n_rows);
    // consider NA if all margins are missing
    // otherwise it's available
    for(unsigned int ix=0; ix<yvec.n_rows; ix++){
      arma::uvec yfinite_row = arma::find_finite(yvec.row(ix));
      if(yfinite_row.n_elem > 0){
        // at least one is available
        na_1_blocks(i)(ix) = 1;
      }
      if(yfinite_row.n_elem < q){
        // at least one is missing
        na_0_blocks(i)(ix) = 1;
      }
    }
    na_ix_blocks(i) = arma::find(na_1_blocks(i) == 1); 
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[na_study] step 2.\n";
  }
  
  n_ref_blocks = 0;
  for(unsigned int i=0; i<n_blocks; i++){
    block_ct_obs(i) = arma::accu(na_1_blocks(i));
    if(block_ct_obs(i) > 0){
      n_loc_ne_blocks += indexing(i).n_elem;
      n_ref_blocks += 1;
    } 
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[na_study] step 3.\n";
  }
  
  int j=0;
  reference_blocks = arma::zeros<arma::uvec>(n_ref_blocks);
  //ref_block_names = arma::zeros<arma::uvec>(n_ref_blocks);
  for(unsigned int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    if(block_ct_obs(u) > 0){
      reference_blocks(j) = i;
      //ref_block_names(j) = u;
      j ++;
    } 
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[na_study] done.\n";
  }
  
}

void MeshedMNM::init_cache(){
  // coords_caching stores the layer names of those layers that are representative
  // coords_caching_ix stores info on which layers are the same in terms of rel. distance
  
  if(verbose & debug){
    Rcpp::Rcout << "init_cache start \n";
  }
  //coords_caching_ix = caching_pairwise_compare_uc(coords_blocks, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching_ix = caching_pairwise_compare_uci(coords, indexing, block_names, block_ct_obs, cached); // uses block_names(i)-1 !
  coords_caching = arma::unique(coords_caching_ix);
  
  //parents_caching_ix = caching_pairwise_compare_uc(parents_coords, block_names, block_ct_obs);
  //parents_caching_ix = caching_pairwise_compare_uci(coords, parents_indexing, block_names, block_ct_obs);
  //parents_caching = arma::unique(parents_caching_ix);
  
  arma::field<arma::mat> kr_pairing(n_blocks);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i = 0; i<n_blocks; i++){
    int u = block_names(i)-1;
    arma::mat cmat = coords.rows(indexing(u));
    if(parents_indexing(u).n_elem > 0){
      arma::mat pmat = coords.rows(parents_indexing(u));
      arma::mat kr_mat_c = arma::join_vert(cmat, pmat);
      kr_pairing(u) = kr_mat_c;
    } else {
      kr_pairing(u) = cmat;
    }
  }
  
  kr_caching_ix = caching_pairwise_compare_uc(kr_pairing, block_names, block_ct_obs, cached);
  kr_caching = arma::unique(kr_caching_ix);
  
  starting_kr = 0;

  cx_and_kr_caching = kr_caching;
  
  
  // 
  findkr = arma::zeros<arma::uvec>(n_blocks);
  findcc = arma::zeros<arma::uvec>(n_blocks);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    int kr_cached_ix = kr_caching_ix(u);
    arma::uvec cpx = arma::find(kr_caching == kr_cached_ix, 1, "first");
    findkr(u) = cpx(0);
    
    //if(forced_grid){
    int u_cached_ix = coords_caching_ix(u);
    arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
    findcc(u) = cx(0);
    //}
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "Caching c: " << coords_caching.n_elem 
                << " k: " << kr_caching.n_elem << "\n";
  }
  
}

void MeshedMNM::init_indexing(){
  
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_indexing] parent_indexing\n";
  }
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents(u).n_elem > 0){
      arma::field<arma::uvec> pixs(parents(u).n_elem);
      for(unsigned int pi=0; pi<parents(u).n_elem; pi++){
        pixs(pi) = indexing(parents(u)(pi));
      }
      parents_indexing(u) = field_v_concat_uv(pixs);
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_indexing] done.\n";
  }
  
}

void MeshedMNM::init_matern(int num_threads, int matern_twonu_in=1, bool use_ps=true){
  nThreads = num_threads;
  
  int bessel_ws_inc = 5;
  matern.bessel_ws = (double *) R_alloc(nThreads*bessel_ws_inc, sizeof(double));
  matern.twonu = matern_twonu_in;
  matern.using_ps = use_ps;
  matern.estimating_nu = (dd == 2) & (param_data.theta.n_rows == 3);
  
}

void MeshedMNM::init_gibbs_index(){
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_gibbs_index] dim_by_parent, parents_coords, children_coords\n";
  }
  arma::field<arma::uvec> dim_by_parent(n_blocks);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // block name
    
    if(indexing_obs(u).n_elem > 0){ 
      // number of coords of the jth parent of the child
      dim_by_parent(u) = arma::zeros<arma::uvec>(parents(u).n_elem + 1);
      for(unsigned int j=0; j<parents(u).n_elem; j++){
        dim_by_parent(u)(j+1) = indexing(parents(u)(j)).n_elem;
      }
      dim_by_parent(u) = arma::cumsum(dim_by_parent(u));
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_gibbs_index] u_is_which_col_f\n";
  }
  

  for(unsigned int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(indexing(u).n_elem > 0){
      // children-parent relationship variables
      u_is_which_col_f(u) = arma::field<arma::field<arma::uvec> > (children(u).n_elem);
      
      for(unsigned int c=0; c<children(u).n_elem; c++){
        int child = children(u)(c);
        // which parent of child is u which we are sampling
        arma::uvec u_is_which = arma::find(parents(child) == u, 1, "first"); 
        
        // which columns correspond to it
        int firstcol = dim_by_parent(child)(u_is_which(0));
        int lastcol = dim_by_parent(child)(u_is_which(0)+1);
        
        int dimen = parents_indexing(child).n_elem;
        
        // this is for w=mat and fields
        arma::vec colix = arma::zeros(dimen);
        for(int s=0; s<1; s++){
          int shift = s * dimen;
          colix.subvec(shift + firstcol, shift + lastcol-1).fill(1);
        }
        
        u_is_which_col_f(u)(c) = arma::field<arma::uvec> (2);
        u_is_which_col_f(u)(c)(0) = arma::find(colix == 1); // u parent of c is in these columns for c
        u_is_which_col_f(u)(c)(1) = arma::find(colix != 1); // u parent of c is NOT in these columns for c
      }
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_gibbs_index] done.\n";
  }
  
}

void MeshedMNM::init_meshdata(const arma::mat& theta_in){
  if(verbose & debug){
    Rcpp::Rcout << "[init_meshdata]\n";
  }
  // block params
  //param_data.w_cond_mean_K = arma::field<arma::cube> (n_blocks);
  //param_data.w_cond_prec   = arma::field<arma::cube> (n_blocks);
  
  //param_data.Rproject = arma::field<arma::cube>(n_blocks);
  //param_data.Riproject = arma::field<arma::cube>(n_blocks);
  //param_data.Hproject = arma::field<arma::cube>(n_blocks);
  
  param_data.Smu_start = arma::field<arma::mat>(n_blocks);
  param_data.Sigi_chol = arma::field<arma::mat>(n_blocks);
  param_data.AK_uP = arma::field<arma::field<arma::cube> >(n_blocks);
  param_data.CC_cache = arma::field<arma::cube>(coords_caching.n_elem);
  
  param_data.Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<n_blocks; i++){
    //int u=block_names(i) - 1;
    //param_data.w_cond_mean_K(i) = arma::zeros(indexing(i).n_elem, parents_indexing(i).n_elem, k);
    //param_data.w_cond_prec(i) = arma::zeros(indexing(i).n_elem, indexing(i).n_elem, k);

    
    param_data.Smu_start(i) = arma::zeros(k*indexing(i).n_elem, 1);
    param_data.Sigi_chol(i) = arma::zeros(k*indexing(i).n_elem, k*indexing(i).n_elem);
    param_data.AK_uP(i) = arma::field<arma::cube>(children(i).n_elem);
    for(unsigned int c=0; c<children(i).n_elem; c++){
      int child = children(i)(c);
      param_data.AK_uP(i)(c) = arma::zeros(indexing(i).n_elem, indexing(child).n_elem, k);
    }
  }
  param_data.w_cond_prec_ptr.reserve(n_blocks);
  param_data.w_cond_mean_K_ptr.reserve(n_blocks);
  param_data.w_cond_prec_parents_ptr.reserve(n_blocks);
  
  for(unsigned int i=0; i<n_blocks; i++){
    arma::cube jibberish = arma::zeros(1,1,1);
    param_data.w_cond_prec_ptr.push_back(&jibberish);
    param_data.w_cond_mean_K_ptr.push_back(&jibberish);
    param_data.w_cond_prec_parents_ptr.push_back(&jibberish);
  }
  
  param_data.Kxxi_cache = arma::field<arma::cube>(coords_caching.n_elem);
  for(unsigned int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i);
    param_data.Kxxi_cache(i) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
    if(block_ct_obs(u) > 0){
      param_data.CC_cache(i) = arma::cube(indexing(u).n_elem, indexing(u).n_elem, k);
    }
  }
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  
  // ***
  param_data.wcore = arma::zeros(n_blocks, 1);
  param_data.loglik_w_comps = arma::zeros(n_blocks, 1);
  param_data.loglik_w       = 0; 
  param_data.theta          = theta_in;//##
  
  // noncentral parameters
  //param_data.ll_y = arma::zeros(coords.n_rows, 1);
  //param_data.ll_y_all       = 0; 
  
  //param_data.DplusSi = arma::zeros(q, q, y.n_rows);
  //param_data.DplusSi_c = arma::zeros(q, q, y.n_rows);
  //param_data.DplusSi_ldet = arma::zeros(y.n_rows);
  
  param_data.H_cache = arma::field<arma::cube> (kr_caching.n_elem);
  param_data.Ri_cache = arma::field<arma::cube> (kr_caching.n_elem);
  param_data.Kppi_cache = arma::field<arma::cube> (kr_caching.n_elem);
  for(unsigned int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    param_data.Ri_cache(i) = 
      arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
    if(parents(u).n_elem > 0){
      param_data.H_cache(i) = 
        arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem, k);
      param_data.Kppi_cache(i) = 
        arma::zeros(parents_indexing(u).n_elem, parents_indexing(u).n_elem, k);
    }
  }
  
  alter_data = param_data; 
  if(verbose & debug){
    Rcpp::Rcout << "[init_meshdata] done.\n";
  }
  
}

bool MeshedMNM::refresh_cache(MeshDataLMC& data){
  start_overall = std::chrono::steady_clock::now();
  
  if(verbose & debug){
    Rcpp::Rcout << "[refresh_cache] start.\n";
  }
  
  
  data.Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
  int errtype = -1;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); 
    if(block_ct_obs(u) > 0){
      for(unsigned int j=0; j<k; j++){
        data.CC_cache(i).slice(j) = Correlationf(coords, indexing(u), indexing(u), //coords.rows(indexing(u)), coords.rows(indexing(u)), 
                      data.theta.col(j), matern, true);
      }
    }
  }
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int it=0; it<cx_and_kr_caching.n_elem; it++){
    int i = 0;
    if(it < starting_kr){
      // this means we are caching coords
      i = it;
      int u = coords_caching(i); // block name of ith representative
      try {
        CviaKron_invsympd_(data.Kxxi_cache(i),
                           coords, indexing(u), k, data.theta, matern);
      } catch (...) {
        errtype = 1;
      }
    } else {
      // this means we are caching kr
      i = it - starting_kr;
      int u = kr_caching(i);
      try {
        if(block_ct_obs(u) > 0){
          //int u_cached_ix = coords_caching_ix(u);
          //arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first");
          
          int ccfound = findcc(u);
          //arma::cube Cxx = CC_cache(ccfound);
          
          data.Ri_chol_logdet(i) = CviaKron_HRi_(data.H_cache(i), data.Ri_cache(i), 
                              data.Kppi_cache(i), data.CC_cache(ccfound),
                              coords, indexing(u), parents_indexing(u), k, data.theta, matern);
        }
      } catch (...) {
        errtype = 2;
      }
    }
  }
  
  if(false & (verbose & debug)){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[refresh_cache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
  //Rcpp::Rcout << "refresh_cache " << errtype << endl;
  
  if(errtype > 0){
    if(verbose & debug){
      Rcpp::Rcout << "Cholesky failed at some point. Here's the value of theta that caused this" << "\n";
      Rcpp::Rcout << "theta: " << data.theta.t() << "\n";
      Rcpp::Rcout << " -- auto rejected and proceeding." << "\n";
    }
    return false;
  }
  return true;
}

void MeshedMNM::update_block_covpars(int u, MeshDataLMC& data){
  //message("[update_block_covpars] start.");
  // given block u as input, this function updates H and R
  // which will be used later to compute logp(w | theta)
  int krfound = findkr(u);
  
  //data.w_cond_prec(u) = data.Ri_cache(krfound);
  data.w_cond_prec_ptr.at(u) = &data.Ri_cache(krfound);
  
  data.logdetCi_comps(u) = data.Ri_chol_logdet(krfound);
  
  if( parents(u).n_elem > 0 ){
    //data.w_cond_mean_K(u) = H_cache(krfound);
    data.w_cond_mean_K_ptr.at(u) = &data.H_cache(krfound);
    data.w_cond_prec_parents_ptr.at(u) = &data.Kppi_cache(krfound);
  } 
  
  
}

void MeshedMNM::update_block_wlogdens(int u, MeshDataLMC& data){
  //message("[update_block_wlogdens].");
  arma::mat wx = w.rows(indexing(u));
  arma::mat wcoresum = arma::zeros(1, k);
  if( parents(u).n_elem > 0 ){
    arma::mat wpar = w.rows(parents_indexing(u));
    for(unsigned int j=0; j<k; j++){
      wx.col(j) = wx.col(j) - 
        (*data.w_cond_mean_K_ptr.at(u)).slice(j) *
        //data.w_cond_mean_K(u).slice(j) * 
        wpar.col(j);
    }
  }
  
  for(unsigned int j=0; j<k; j++){
    wcoresum(j) = 
      arma::conv_to<double>::from(arma::trans(wx.col(j)) * 
      //data.w_cond_prec(u).slice(j) * 
      (*data.w_cond_prec_ptr.at(u)).slice(j) *
      wx.col(j));
  }
  
  data.wcore.row(u) = arma::accu(wcoresum);
  data.loglik_w_comps.row(u) = (indexing(u).n_elem+.0) * hl2pi -.5 * arma::accu(wcoresum); //
  //arma::accu(data.wcore.slice(u).diag());
  
  //message("[update_block_wlogdens] done.");
}

bool MeshedMNM::calc_ywlogdens(MeshDataLMC& data){
  start_overall = std::chrono::steady_clock::now();
  // called for a proposal of theta
  // updates involve the covariances
  // and Sigma for adjusting the error terms
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    update_block_covpars(u, data);
    update_block_wlogdens(u, data);
  }
  
  data.loglik_w = 
    arma::accu(data.logdetCi_comps) + 
    arma::accu(data.loglik_w_comps);// + 
    //arma::accu(data.ll_y); //****
  
  if(std::isnan(data.loglik_w)){
    Rcpp::Rcout << "Logdens components: \n" <<
      arma::accu(data.logdetCi_comps) << " " << 
        arma::accu(data.loglik_w_comps) << endl;
  }
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[calc_ywlogdens] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n"
                << "of which " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us. for [update_lly]\n";
  }
  
  return true;
}

bool MeshedMNM::get_loglik_comps_w(MeshDataLMC& data){
  bool acceptable = refresh_cache(data);
  if(acceptable){
    acceptable = calc_ywlogdens(data);
    return acceptable;
  } else {
    return acceptable;
  }
}


void MeshedMNM::logpost_refresh_after_gibbs(MeshDataLMC& data){
  
  if(verbose & debug){
    Rcpp::Rcout << "logpost_refresh_after_gibbs\n";
    start_overall = std::chrono::steady_clock::now();
  }
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    //update_block_covpars(u, data);
    update_block_wlogdens(u, data);

  }
  
  data.loglik_w = arma::accu(data.logdetCi_comps) + 
    arma::accu(data.loglik_w_comps);// + arma::accu(data.ll_y); //***
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[logpost_refresh_after_gibbs] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count() 
                << "us.\n"
                << "of which " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                << "us to do [update_lly].\n";
  }
}


void MeshedMNM::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

void MeshedMNM::accept_make_change(){
  std::swap(param_data, alter_data);
}

// --- 
void MeshedMNM::init_for_mcmc(){
  if(verbose & debug){
    Rcpp::Rcout << "[init_for_mcmc]\n";
  }
  
  
  // defaults 
  w_hmc_nuts = false;
  w_hmc_rm = true;
  w_hmc_srm = true;
  

  // let user choose what to use
  if(which_hmc == 0){
    // rm-mala then s-mmala
    if(verbose){
      Rcpp::Rcout << "Using SiMPA" << endl;
    }
    w_hmc_nuts = false;
    w_hmc_rm = true;
    w_hmc_srm = true;
  }
  if(which_hmc == 1){
    // mala
    if(verbose){
      Rcpp::Rcout << "Using MALA" << endl;
    }
    w_hmc_nuts = false;
    w_hmc_rm = false;
    w_hmc_srm = false;
  }
  if(which_hmc == 2){
    // nuts
    if(verbose){
      Rcpp::Rcout << "Using NUTS" << endl;
    }
    w_hmc_nuts = true;
    w_hmc_rm = false;
    w_hmc_srm = false;
  }
  if(which_hmc == 3){
    // rm-mala
    if(verbose){
      Rcpp::Rcout << "Using simplified manifold MALA" << endl;
    }
    w_hmc_nuts = false;
    w_hmc_rm = true;
    w_hmc_srm = false;
  }
  if(which_hmc == 4){
    // rm-mala then s-mmala
    if(verbose){
      Rcpp::Rcout << "Using SiMPA (previous version)" << endl;
    }
    w_hmc_nuts = false;
    w_hmc_rm = true;
    w_hmc_srm = true;
  }
  if(which_hmc == 5){
    // nuts
    if(verbose){
      Rcpp::Rcout << "Using Elliptical slice sampler" << endl;
    }
    w_hmc_nuts = true;
    w_hmc_rm = false;
    w_hmc_srm = false;
  }
  if(which_hmc == 6){
    // nuts
    if(verbose){
      Rcpp::Rcout << "Using HMC" << endl;
    }
    w_hmc_nuts = true; // target an acceptance rate similar to nuts
    w_hmc_rm = false;
    w_hmc_srm = false;
  }

  
  blg_node.reserve(q); // for lambda & beta
  //gamma_node.reserve(q); // for gamma
  // start with small epsilon for a few iterations,
  // then find reasonable and then start adapting
  //gamma_hmc_started = arma::zeros<arma::uvec>(q);
  blg_hmc_started = arma::zeros<arma::uvec>(q);
  
  for(unsigned int j=0; j<q; j++){
    arma::vec yj_obs = y( ix_by_q_a(j), oneuv * j );
    //arma::vec N_obs = Npop( ix_by_q_a(j), oneuv * j );
    
    // Gamma
    //arma::mat offsets_gamma = arma::zeros(ix_by_q_a(j).n_elem);
    
    arma::mat Z_obs = Z.rows(ix_by_q_a(j));
    //int family_gamma = 2; // binomial family for gamma updates
    //NodeDataB new_gamma_block(yj_obs, N_obs, Z_obs, family_gamma);
    
    //gamma_node.push_back(new_gamma_block);
    
    //AdaptE new_gamma_hmc_adapt;
    //new_gamma_hmc_adapt.init(.05, pg, w_hmc_srm, w_hmc_nuts);
    //gamma_hmc_adapt.push_back(new_gamma_hmc_adapt);
    
    //gamma_hmc_started(j) = 0;
    
    // Lambda & Beta
    arma::mat X_obs = X.rows(ix_by_q_a(j));
    arma::mat W_obs = w.rows(ix_by_q_a(j));
    arma::mat XW = arma::join_horiz(X_obs, W_obs);
    
    //arma::vec not_needed = arma::zeros(yj_obs.n_elem);
    //int family_lambda = 1; // poisson family for lambda-beta updates
    NodeDataBLG new_blg_block(yj_obs, XW, Z_obs);
    blg_node.push_back(new_blg_block);
    
    // *** sampling beta and lambda together so we use p+k here
    arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
    int n_lambdas = subcols.n_elem;
    AdaptE new_blg_adapt;
    new_blg_adapt.init(.05, p+n_lambdas+pg, w_hmc_srm, w_hmc_nuts);
    blg_hmc_adapt.push_back(new_blg_adapt);
  }

  if(verbose & debug){
    Rcpp::Rcout << "[init nongaussian outcome]\n";
  }
  
  w_node.reserve(n_blocks); // for w
  hmc_eps = .025 * arma::ones(n_blocks);
  hmc_eps_started_adapting = arma::zeros<arma::uvec>(n_blocks);
  
  for(unsigned int i=0; i<n_blocks; i++){
    NodeDataW new_block;
    w_node.push_back(new_block);
    
    int blocksize = indexing(i).n_elem * k;
    AdaptE new_eps_adapt;
    new_eps_adapt.init(hmc_eps(i), blocksize, w_hmc_srm, w_hmc_nuts);
    hmc_eps_adapt.push_back(new_eps_adapt);
  }
  
  //Rcpp::Rcout << " Initializing HMC for W -- 2" << endl;
  arma::mat offset_for_w = offsets + XB;
  arma::mat prob_offset = 1.0/(1.0 + exp(-Zg));
  
  //#pragma omp parallel for
  for(unsigned int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    
    NodeDataW new_block(y, na_mat, offset_for_w,
                        indexing(u), k);
    
    new_block.update_mv(offset_for_w, prob_offset, Lambda);
    
    // other fixed pars
    new_block.parents_dim = parents_indexing(u).n_rows;
    new_block.num_children = children(u).n_elem;
    
    new_block.w_child = arma::field<arma::mat> (children(u).n_elem); 
    new_block.Ri_of_child = arma::field<arma::cube* > (children(u).n_elem); 
    new_block.Kco_wo = arma::field<arma::mat>(children(u).n_elem);
    new_block.Kcx_x = arma::field<arma::cube>(children(u).n_elem);
    
    for(unsigned int c=0; c<children(u).n_elem; c++){
      int child = children(u)(c);
      new_block.Kcx_x(c) = arma::zeros(indexing(child).n_elem, indexing(u).n_elem, k);
    }
    
    w_node.at(u) = new_block;
  }
  
}
