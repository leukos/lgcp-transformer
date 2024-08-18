# install.packages('patchwork')
# install.packages('tidyverse')
# install.packages('tidyr')
# install.packages('dplyr')
# install.packages('pbmcapply')
# install.packages('spatstat')
# install.packages('RJSONIO')

library(dplyr)
library(tidyr)
library(pbmcapply)
library(patchwork)
library(tidyverse)
library(spatstat)
library(RJSONIO)
library(utils)

# set the number of cores
ncores <- ifelse(detectCores() == 1, 1, detectCores() - 1)


###########################################
# Simple helper function for the data gen
###########################################
generate_samples <- function(mu, var, scale, ncores) {  
  LGCP_sims <- pbmcmapply(function(mu, var, scale){
    rLGCP(mu = mu, var = var, scale = scale, saveLambda = FALSE)
  }, mu, var, scale, SIMPLIFY = FALSE, mc.cores = ncores)
  Data_LGCP <- tibble(mu = mu, var = var, scale = scale, pp = LGCP_sims) %>%
    mutate(N = map_dbl(pp, npoints)) %>%
    mutate(L = pbmclapply(pp, function(ppp){
      L <- Lest(ppp, correction = "best")
      L$iso - L$r
    }, mc.cores = ncores)) %>%
    mutate(X = pbmclapply(pp, function(ppp){
      ppp$x
    }, mc.cores = ncores)) %>%
    mutate(Y = pbmclapply(pp, function(ppp){
      ppp$y
    }, mc.cores = ncores)) %>%
    select(-pp)
  idx_na_L <- sapply(Data_LGCP$L, function(L){any(is.na(L))})
  Data_LGCP <- filter(Data_LGCP, !idx_na_L)
  return(Data_LGCP)
}


###########################################
# Create the first trainging data set
# consisting of 100k samples where each
# parameter combination was sampled
# uniformly over the defined space.
###########################################

# set the number of training fsamples
ntrain <- 100000

# set the range for the parameters that we simulate from
mu <- runif(ntrain, 4, 6)
var <- runif(ntrain, 0, 4)
scale <- runif(ntrain, 0.001, 0.1)

train_set <- generate_samples(mu, var, scale, ncores)

write(toJSON(train_set), "sample_100000.json")
zip(zipfile = 'sample_100000.zip', files = 'sample_100000.json')

###########################################
# Create the test data set
# consisting of 10k samples where each
# parameter combination was sampled
# uniformly over the defined space.
###########################################

ntest <- 10000

# set the range for the parameters that we simulate from
mu <- runif(ntest, 4, 6)
var <- runif(ntest, 0, 4)
scale <- runif(ntest, 0.001, 0.1)

test_set <- generate_samples(mu, var, scale, ncores)

write(toJSON(test_set), "sample_10000.json")
zip(zipfile = 'sample_10000.zip', files = 'sample_10000.json')

###########################################
# Create a second train data set
# consisting of 100k samples where 10k
# parameter combination were sampled
# uniformly over the defined space and then
# repeated 10 times with one random sample
# for each of these 10 repeated parameter
# vectors
###########################################

# repeat the samples 10 times
mu <- rep(mu, 10)
var <- rep(var, 10)
scale <- rep(scale, 10)

test_set <- generate_samples(mu, var, scale, ncores)

write(toJSON(test_set), "sample_10x10000.json")
zip(zipfile = 'sample_10x10000.zip', files = 'sample_10x10000.json')
