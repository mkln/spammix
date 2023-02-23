rm(list=ls())
library(spammix)
library(magrittr)
library(dplyr)
library(ggplot2)


set.seed(2022)

SS <- 50 # coord values for jth dimension 
dd <- 2 # spatial dimension
n <- SS^2 # number of locations
q <- 5 # number of outcomes
k <- 1 # number of spatial factors used to make the outcomes
p <- 2 # number of covariates

xlocs <- seq(0, 1, length.out=SS)
coords <- expand.grid(list(xlocs, xlocs)) %>% 
  as.data.frame() 

clist <- 1:k %>% lapply(function(i) coords %>% 
                          mutate(mv_id=i) %>% 
                          as.matrix()) 

philist <- rep(3, k) # spatial decay for each factor

# cholesky decomp of covariance matrix
LClist <- 1:k %>% lapply(function(i) t(chol(
  #exp(- philist[i] * as.matrix(dist(clist[[i]])))))) #^2 + diag(nrow(clist[[i]]))*1e-5))))
  meshed:::Cov_matern(clist[[i]], clist[[i]], 1, philist[i], 1.5, 0, T, 10))))

# generating the factors
wlist <- 1:k %>% lapply(function(i) LClist[[i]] %*% rnorm(n))

# factor matrix
WW <- do.call(cbind, wlist)

# factor loadings
Lambda <- matrix(0, q, ncol(WW))
diag(Lambda) <- runif(k, 1, 3)
Lambda[lower.tri(Lambda)] <- runif(sum(lower.tri(Lambda)), -1, 1)

Lambda <- Lambda

XX <- 1:p %>% lapply(function(i) rnorm(n, 0, .1)) %>% do.call(cbind, .)
Beta <- matrix(rnorm(p*q), ncol=q) 

ZZ <- 1:p %>% lapply(function(i) rnorm(n, 0, .1)) %>% do.call(cbind, .)
GammaCoeff <- matrix(rnorm(p*q), ncol=q)
#ZZ <- matrix(1, nrow=n, ncol=1)


# outcome matrix, fully observed
LambdaW <- WW %*% t(Lambda)
linear_predictor <- XX %*% Beta + LambdaW

detect_prob <- 1/(1 + exp(-ZZ %*% GammaCoeff))
#detect_prob <- c(0.7, 0.3, .5, .8, .99)

N_full <- YY_full <- YY <- matrix(0, ncol=q, nrow=nrow(linear_predictor))
for(i in 1:q){
  N_full[,i] <- rpois(n, exp(linear_predictor[,i]))
  YY_full[,i] <- rbinom(n, N_full[,i], detect_prob[,i])
  YY[,i] <- YY_full[,i]
  YY[sample(1:n, n/5, replace=FALSE), i] <- NA
}


simdata <- coords %>%
  cbind(data.frame(Outcome_full=YY_full, 
                   Pop_full = N_full,
                   Lat = LambdaW,
                   Outcome_obs = YY)) 

simdata %>% dplyr::select(Var1, Var2, contains("_full")) %>%
  tidyr::gather(Outcome, Value, -all_of(colnames(coords))) %>%
  ggplot(aes(Var1, Var2, fill=Value)) + 
  geom_raster() + 
  facet_wrap(Outcome ~., ncol=2, scales="free") +
  scale_fill_viridis_c()

mcmc_keep <- 500
mcmc_burn <- 500
mcmc_thin <- 1

axis_partition <- c(15 , 15) # resulting in blocks of approx size 32

set.seed(1)
mesh_total_time <- system.time({
  mixout <- spammix(YY[,c(1,2,3,4,5)], XX, ZZ, coords, k = 1,
                      axis_partition=axis_partition,
                      n_samples = mcmc_keep, n_burn = mcmc_burn, n_thin = mcmc_thin, 
                      n_threads = 16,
                      #starting=list(lambda = Lambda, beta=Beta, phi=1, v=WW),
                      prior = list(phi=c(.1, 5), nu=c(1.5,1.5)),
                      settings = list(adapting=T, cache=T, saving=F, ps=T, hmc=0),
                      verbose = 20,
                      debug=list(sample_beta=T, sample_theta=T, sample_N=F,
                                 sample_w=T, sample_lambda=T, sample_gamma=T,
                                 verbose=T, debug=F)
  )})

if(0){
  mesh_total_time <- system.time({
    meshout <- meshed::spmeshed(YY, XX, family=rep("poisson", q), coords, k = 1,
                                axis_partition=axis_partition,
                                n_samples = mcmc_keep, n_burn = mcmc_burn, n_thin = mcmc_thin, 
                                n_threads = 16,
                                #starting=list(lambda = Lambda, beta=Beta, phi=1, v=WW),
                                prior = list(phi=c(.1, 5), nu = c(1.5, 1.5)),
                                #settings = list(adapting=T, forced_grid=F, cache=T, saving=F, ps=T, hmc=4),
                                verbose= 20,
                                debug=list(sample_beta=T, sample_theta=T, #sample_N=T,
                                           sample_w=T, sample_lambda=T, #sample_gamma=T,
                                           verbose=F, debug=F)
    )})
}



plot_cube <- function(cube_mcmc, q, k, name="Parameter"){
  par(mar=c(2.5,2,1,1), mfrow=c(q,k))
  for(i in 1:q){
    for(j in 1:k){
      cube_mcmc[i, j,] %>% plot(type='l', main="{name} {i}, {j}" %>% glue::glue())
    }
  }
}
plot_cube(mixout$gamma_mcmc[,,seq(1,mcmc_keep*mcmc_thin, mcmc_thin),drop=F], ncol(ZZ), q, "Gamma")


# chain plots
plot_cube(mixout$theta_mcmc, 1, k, "theta")
plot_cube(mixout$lambda_mcmc, q, k, "Lambda")
plot_cube(mixout$beta_mcmc, p, q, "Beta")


# posterior means
mixout$lambda_mcmc %>% apply(1:2, mean) #dlm::ergMean) %>% `[`(,1,1) %>% plot(type='l')
meshout$beta_mcmc %>% apply(1:2, mean)
mixout$gamma_mcmc %>% apply(1:2, mean)
mixout$theta_mcmc %>% apply(1:2, mean)

target_model <- mixout

# process means
wmesh <- data.frame(target_model$w_mcmc %>% summary_list_mean())
colnames(wmesh) <- paste0("wmesh_", 1:q)
# predictions
ymesh <- data.frame(target_model$yhat_mcmc %>% summary_list_mean())
colnames(ymesh) <- paste0("ymesh_", 1:q)
# population size
npopmesh <- data.frame(mixout$npop_mcmc %>% summary_list_mean())
colnames(npopmesh) <- paste0("nmesh_", 1:q)


mesh_df <- 
  target_model$coordsdata %>% 
  cbind(ymesh, wmesh#, npopmesh
        )
results <- simdata %>% left_join(mesh_df)

rmse_calc <- function(i){
  varname <- paste0("Outcome_obs.", i)
  targname <- paste0("Outcome_full.", i)
  meshname <- paste0("ymesh_", i)
  return(
    results %>% filter(!complete.cases(.data[[varname]])) %>% 
      mutate(se = (.data[[targname]] - .data[[meshname]])^2) %>% pull(se) %>% median()# %>% sqrt()
  )
}

1:q %>% sapply(rmse_calc)


# prediction rmse, out of sample

results %>% filter(!complete.cases(Outcome_obs.2)) %>% 
  with((Outcome_full.2 - ymesh_2)^2) %>% mean() %>% sqrt()
results %>% filter(!complete.cases(Outcome_obs.3)) %>% 
  with((Outcome_full.3 - ymesh_3)^2) %>% mean() %>% sqrt()

(postmeans1 <- results %>% dplyr::select(Var1, Var2, 
                                        Pop_full.1, nmesh_1,
                                        Outcome_full.1, ymesh_1) %>%
    tidyr::gather(Variable, Value, -Var1, -Var2) %>%
    ggplot(aes(Var1, Var2, fill=log1p(Value))) +
    geom_raster() +
    facet_wrap(Variable ~ ., ncol= 2) +
    scale_fill_viridis_c())

(postmeans2 <- results %>% dplyr::select(Var1, Var2, 
                                        Pop_full.2, nmesh_2,
                                        Outcome_full.2, ymesh_2) %>%
    tidyr::gather(Variable, Value, -Var1, -Var2) %>%
    ggplot(aes(Var1, Var2, fill=(Value))) +
    geom_raster() +
    facet_wrap(Variable ~ ., ncol= 2) +
    scale_fill_viridis_c())




