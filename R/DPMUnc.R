#!/usr/bin/Rscript

read_line_n <- function(filepath, nDim, n) {
  values = as.numeric(read.table(filepath, skip=n-1, nrow=1, sep=','))
  mat = matrix(values, ncol=nDim)
  return (mat)
}

count_lines <- function(filepath) {
  f <- file(filepath, open="rb")
  nlines <- 0L
  while (length(chunk <- readBin(f, "raw", 65536)) > 0) {
      nlines <- nlines + sum(chunk == as.raw(10L))
  }
  close(f)
  return(nlines)
}

#' Scale observed data so variance of every variable is 1
#'
#' @param obsData The observed data in matrix form (n observations x p variables)
#' @param obsVars The observed variances of the data (n observations x p variables)
#'
#' @return Named list containing scaled data and scaled variances
#' @export
#'
#' @examples
#' obsData <- matrix(rnorm(400, mean=0, sd=c(1,2,3,4)), ncol=4, byrow = TRUE)
#' obsVars <- matrix(rep(c(1, 4, 9, 16), 10), ncol=4, byrow = TRUE)
#' result = scale_data(obsData, obsVars)
#' # After scaling, var of every variable in data should be 1
#' apply(result$data, var, MARGIN=2)
#' # and the scaled variances will be approximately 1
#' apply(result$vars, mean, MARGIN=2)
scale_data <- function(obsData, obsVars) {
    scaledData <- scale(obsData, scale=TRUE, center=TRUE)
    scaledVars <- scale(obsVars,
                        # Don't center the variances! This wouldn't make any sense!
                        center=FALSE,
                        # Scale the variances by the square of the scale factors we used for the data
                        scale=attr(scaledData, "scaled:scale") ** 2)

    return (list("data"=scaledData, "vars"=scaledVars))
}

#' DPMUnc - Run Dirichlet Process Mixture Modeller taking uncertainty of data points into account
#'
#' @param obsData The observed data in matrix form (n observations x p variables)
#' @param obsVars The observed variances of the data (n observations x p variables)
#' @param saveFileDir Directory where all output will be saved
#' @param seed Seed for random number generator, to make the function deterministic.
#' @param kappa0 Hyperparameter for Normal-Gamma prior on cluster parameters
#' @param alpha0 Hyperparameter for Normal-Gamma prior on cluster parameters
#' @param beta0 Hyperparameter for Normal-Gamma prior on cluster parameters
#' @param K Initial number of clusters.
#' @param nIts Total number of iterations to run. The user should check they are happy
#' that the model has converged before using any of the results.
#' @param thinningFreq Controls how many samples are saved. E.g. a value of 10 means
#' every 10th sample will be saved.
#' @param saveClusterParams Boolean, determining whether the cluster parameters (mean
#' and variance of every cluster) for every saved iteration should be saved in a file or not.
#' Both cluster parameters and latent observations take up more space than other saved variables.
#' The files clusterVars.csv and clusterMeans.csv will be created in either case, but
#' will be left empty if saveClusterParams is FALSE.
#' @param saveLatentObs Boolean, determining whether the latent observations (underlying true observations)
#' for every saved iteration should be saved in a file or not. Both cluster parameters and
#' latent observations take up more space than other saved variables.
#' The file latentObservations.csv will be created in either case, but
#' will be left empty if saveLatentObs is FALSE.
#' @param quiet Boolean. If FALSE, information will be printed to the terminal including
#' current iteration, current value of K and number of items per cluster.
#' @param scaleData Boolean. If TRUE, data will be scaled so that the variance of every
#' variable (column) in obsData is 1 (and obsVars will be scaled to fit this rescaling).
#' Else, the raw data will be used.
#'
#' @export
#'
#' @examples
#' n = 50; d = 5; k = 4
#' classes = sample(1:k, n, replace=TRUE)
#' group_means = matrix(rep(classes, d), n, d)
#' true_means = group_means + matrix(rnorm(n*d, sd=0.1),n,d)
#' obsVars = matrix(rchisq(n*d,1), n, d)
#' obsData = matrix(rnorm(n*d, mean=as.vector(true_means), sd=sqrt(as.vector(obsVars))), n, d)
#' # The hyperparameters should be carefully checked against the data.
#' alpha0 = 2; beta0 = 0.2 * mean(apply(obsData, 2, var)); kappa0 = 0.5
#' DPMUnc(obsData, obsVars, "test_output", 1234,
#'        kappa0=kappa0, alpha0=alpha0, beta0=beta0)
DPMUnc <- function(obsData,obsVars,saveFileDir,seed,
                   kappa0, alpha0, beta0,
                   K=if(is.vector(obsData)) { floor(length(obsData)/2) } else { floor(nrow(obsData)/2) },
                   nIts = 100000, thinningFreq = 10,
                   saveClusterParams=TRUE, saveLatentObs=FALSE,
                   quiet=TRUE, scaleData=FALSE) {
  if(!dir.exists(saveFileDir)) {
      dir.create(saveFileDir, recursive=TRUE)
  }

  ## deal with vector arguments
  if(is.vector(obsData))
    obsData=matrix(obsData,ncol=1)
  if(is.vector(obsVars))
    obsVars=matrix(obsVars,ncol=1)

  set.seed(seed, sample.kind="Rejection", normal.kind="Inversion", kind="Mersenne-Twister")

  if (scaleData) {
      scaled <- scale_data(obsData, obsVars)
      obsData <- scaled$data
      obsVars <- scaled$vars
  }

  # start in an odd place. convergence longer, but avoids getting stuck in kmeans solution
  kmeansInit             <- kmeans(obsData, centers = K)
  currentAllocations     <- sample(kmeansInit$cluster)

  runDPMUnc(obsData, obsVars,
            nIts, thinningFreq, quiet,
            saveClusterParams, saveLatentObs,
            saveFileDir,
            currentAllocations,
            kappa0,
            alpha0,
            beta0)
}

#' experimental_resumeDPMUnc - Resume run of Dirichlet Process Mixture Modeller taking uncertainty of data points into account
#'
#' @param obsData The observed data in matrix form (n observations x p variables)
#' @param obsVars The observed variances of the data (n observations x p variables)
#' @param saveFileDir Directory where all output will be saved, and where existing output should be found
#' @param seed Seed for random number generator, to make the function deterministic.
#' @param kappa0 Hyperparameter for Normal-Gamma prior on cluster parameters
#' @param alpha0 Hyperparameter for Normal-Gamma prior on cluster parameters
#' @param beta0 Hyperparameter for Normal-Gamma prior on cluster parameters
#' @param K Initial number of clusters.
#' @param nIts Total number of iterations to run. The user should check they are happy
#' that the model has converged before using any of the results.
#' @param thinningFreq Controls how many samples are saved. E.g. a value of 10 means
#' every 10th sample will be saved.
#' @param saveClusterParams Boolean, determining whether the cluster parameters (mean
#' and variance of every cluster) for every saved iteration should be saved in a file or not.
#' Both cluster parameters and latent observations take up more space than other saved variables.
#' The files clusterVars.csv and clusterMeans.csv will be created in either case, but
#' will be left empty if saveClusterParams is FALSE.
#' @param saveLatentObs Boolean, determining whether the latent observations (underlying true observations)
#' for every saved iteration should be saved in a file or not. Both cluster parameters and
#' latent observations take up more space than other saved variables.
#' The file latentObservations.csv will be created in either case, but
#' will be left empty if saveLatentObs is FALSE.
#' @param quiet Boolean. If FALSE, information will be printed to the terminal including
#' current iteration, current value of K and number of items per cluster.
#' @param scaleData Boolean. If TRUE, data will be scaled so that the variance of every
#' variable (column) in obsData is 1 (and obsVars will be scaled to fit this rescaling).
#' Else, the raw data will be used.
#'
#' @export
#'
#' @examples
#' n = 50; d = 5; k = 4
#' classes = sample(1:k, n, replace=TRUE)
#' group_means = matrix(rep(classes, d), n, d)
#' true_means = group_means + matrix(rnorm(n*d, sd=0.1),n,d)
#' obsVars = matrix(rchisq(n*d,1), n, d)
#' obsData = matrix(rnorm(n*d, mean=as.vector(true_means), sd=sqrt(as.vector(obsVars))), n, d)
#' # The hyperparameters should be carefully checked against the data.
#' alpha0 = 2; beta0 = 0.2 * mean(apply(obsData, 2, var)); kappa0 = 0.5
#' DPMUnc(obsData, obsVars, "test_output", 1234,
#'        kappa0=kappa0, alpha0=alpha0, beta0=beta0, nIts=10000)
#' experimental_resumeDPMUnc(obsData, obsVars, "test_output", 1234,
#'                           kappa0=kappa0, alpha0=alpha0, beta0=beta0, nIts=100000)
experimental_resumeDPMUnc <- function(obsData,obsVars,saveFileDir,seed,
                                      kappa0, alpha0, beta0,
                                      K=floor(nrow(obsData)/2), nIts = 100000, thinningFreq = 10,
                                      saveClusterParams=TRUE, saveLatentObs=FALSE,
                                      quiet=TRUE, scaleData=FALSE) {
  numLinesSoFar = count_lines(paste0(saveFileDir, "/alpha.csv"))
  latentObservations = read_line_n(paste0(saveFileDir, "/latentObservations.csv"), numLinesSoFar, nDim=ncol(obsData))
  clusterAllocations = read_line_n(paste0(saveFileDir, "/clusterAllocations.csv"), numLinesSoFar, nDim=1)
  alpha_concentration = read_line_n(paste0(saveFileDir, "/alpha.csv"), numLinesSoFar, nDim=1)
  alpha_concentration = as.numeric(alpha_concentration)

  iterationsSoFar = numLinesSoFar * thinningFreq
  remainingIts = nIts - iterationsSoFar

  set.seed(seed, sample.kind="Rejection", normal.kind="Inversion", kind="Mersenne-Twister")

  if (scaleData) {
      scaled <- scale_data(obsData, obsVars)
      obsData <- scaled$data
      obsVars <- scaled$vars
  }

  resumeDPMUnc(obsData, obsVars,
               remainingIts, thinningFreq, quiet,
               saveClusterParams, saveLatentObs,
               saveFileDir,
               clusterAllocations,
               latentObservations,
               alpha_concentration,
               kappa0,
               alpha0,
               beta0)
}
