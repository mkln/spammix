rm(list=ls())
library(spammix)
library(magrittr)
library(dplyr)
library(ggplot2)


set.seed(2022)

SS <- 50 # coord values for jth dimension 
dd <- 2 # spatial dimension
n <- SS^2 # number of locations
q <- 2 # number of outcomes
k <- 2 # number of spatial factors used to make the outcomes
p <- 2 # number of covariates

xlocs <- seq(0, 1, length.out=SS)
coords <- expand.grid(list(xlocs, xlocs)) %>% 
  as.data.frame() 

clist <- 1:q %>% lapply(function(i) coords %>% 
                          mutate(mv_id=i) %>% 
                          as.matrix()) 

philist <- c(5, 5) # spatial decay for each factor

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
ZZ <- matrix(1, nrow=n, ncol=1)
Beta <- matrix(rnorm(p*q), ncol=q) 

# outcome matrix, fully observed
LambdaW <- WW %*% t(Lambda)
linear_predictor <- XX %*% Beta + LambdaW
N_full <- matrix(0, ncol=q, nrow=nrow(linear_predictor))
N_full[,1] <- rpois(n, exp(linear_predictor[,1]))
N_full[,2] <- rpois(n, exp(linear_predictor[,2]))


detect_prob <- c(0.2, 0.7)

YY_full <- matrix(0, nrow=n, ncol=q)
YY_full[,1] <- rbinom(n, N_full[,1], detect_prob[1])
YY_full[,2] <- rbinom(n, N_full[,2], detect_prob[2])

YY <- YY_full

YY[sample(1:n, n/5, replace=FALSE), 1] <- NA
YY[sample(1:n, n/5, replace=FALSE), 2] <- NA


simdata <- coords %>%
  cbind(data.frame(Outcome_full=YY_full, 
                   Pop_full = N_full,
                   Lat = LambdaW,
                   Outcome_obs = YY)) 

simdata %>%
  tidyr::gather(Outcome, Value, -all_of(colnames(coords))) %>%
  ggplot(aes(Var1, Var2, fill=Value)) + 
  geom_raster() + 
  facet_wrap(Outcome ~., ncol=2, scales="free") +
  scale_fill_viridis_c()

mcmc_keep <- 2000
mcmc_burn <- 15000
mcmc_thin <- 50

axis_partition <- c(10 , 10) # resulting in blocks of approx size 32

set.seed(1)
mesh_total_time <- system.time({
  mixout <- spammix(YY, XX, ZZ, coords, k = 2,
                      axis_partition=axis_partition,
                      n_samples = mcmc_keep, n_burn = mcmc_burn, n_thin = mcmc_thin, 
                      n_threads = 16,
                      #starting=list(lambda = Lambda, beta=Beta, phi=1),
                      prior = list(phi=c(.1, 15), nu=c(1.5,1.5)),
                      settings = list(adapting=T, cache=T, saving=F, ps=T, hmc=0),
                      verbose=10,
                      debug=list(sample_beta=T, sample_theta=T, sample_N=T,
                                 sample_w=T, sample_lambda=T, sample_gamma=T,
                                 verbose=F, debug=F)
  )})

if(0){
  mesh_total_time <- system.time({
    meshout <- meshed::spmeshed(YY, XX, family=c("poisson", "poisson"), coords, k = 2,
                                axis_partition=axis_partition,
                                n_samples = mcmc_keep, n_burn = mcmc_burn, n_thin = mcmc_thin, 
                                n_threads = 16,
                                #starting=list(lambda = Lambda, beta=Beta, phi=1),
                                prior = list(phi=c(.1, 5)),
                                #settings = list(adapting=T, forced_grid=F, cache=T, saving=F, ps=T, hmc=4),
                                verbose=10,
                                debug=list(sample_beta=T, sample_theta=T, #sample_N=T,
                                           sample_w=T, sample_lambda=T, #sample_gamma=T,
                                           verbose=F, debug=F)
    )})
}



sum(mixout$w_mcmc[[1]])

plot_cube <- function(cube_mcmc, q, k, name="Parameter"){
  par(mar=c(2.5,2,1,1), mfrow=c(q,k))
  for(i in 1:q){
    for(j in 1:k){
      cube_mcmc[i, j,] %>% plot(type='l', main="{name} {i}, {j}" %>% glue::glue())
    }
  }
}

# chain plots
plot_cube(mixout$theta_mcmc, 1, k, "theta")
plot_cube(mixout$lambda_mcmc, q, k, "Lambda")
plot_cube(mixout$beta_mcmc, p, q, "Beta")
plot_cube(1/(1+exp(-mixout$gamma_mcmc)), ncol(ZZ), q, "Gamma")


# posterior means
mixout$lambda_mcmc %>% apply(1:2, mean) #dlm::ergMean) %>% `[`(,1,1) %>% plot(type='l')
mixout$beta_mcmc %>% apply(1:2, mean)
mixout$gamma_mcmc %>% apply(1:2, mean)
mixout$theta_mcmc %>% apply(1:2, mean)

target_model <- mixout

# process means
wmesh <- data.frame(target_model$w_mcmc %>% summary_list_mean())
colnames(wmesh) <- paste0("wmesh_", 1:k)
# predictions
ymesh <- data.frame(target_model$yhat_mcmc %>% summary_list_mean())
colnames(ymesh) <- paste0("ymesh_", 1:q)
# population size
npopmesh <- data.frame(mixout$npop_mcmc %>% summary_list_mean())
colnames(npopmesh) <- paste0("nmesh_", 1:q)


mesh_df <- 
  mixout$coordsdata %>% 
  cbind(ymesh, wmesh, npopmesh
        )
results <- simdata %>% left_join(mesh_df)

# prediction rmse, out of sample
results %>% filter(!complete.cases(Outcome_obs.1)) %>% 
  with((Outcome_full.1 - ymesh_1)^2) %>% mean() %>% sqrt()
results %>% filter(!complete.cases(Outcome_obs.2)) %>% 
  with((Outcome_full.2 - ymesh_2)^2) %>% mean() %>% sqrt()

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
    ggplot(aes(Var1, Var2, fill=log1p(Value))) +
    geom_raster() +
    facet_wrap(Variable ~ ., ncol= 2) +
    scale_fill_viridis_c())




