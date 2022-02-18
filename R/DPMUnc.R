#!/usr/bin/Rscript
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
#' @param K Initial number of clusters.
#' @param nIts Total number of iterations to run. The user should check they are happy
#' that the model has converged before using any of the results.
#' @param thinningFreq Controls how many samples are saved. E.g. a value of 10 means
#' every 10th sample will be saved.
#' @param saveClusterParams Boolean, determining whether the cluster parameters (mean
#' and variance of every cluster) for every saved iteration should be saved in a file or not.
#' Both cluster parameters and latent observations take up more space than other saved variables.
#' The files clusterVars.tsv and clusterMeans.tsv will be created in either case, but
#' will be left empty if saveClusterParams is FALSE.
#' @param saveLatentObs Boolean, determining whether the latent observations (underlying true observations)
#' for every saved iteration should be saved in a file or not. Both cluster parameters and
#' latent observations take up more space than other saved variables.
#' The file latentObservations.tsv will be created in either case, but
#' will be left empty if saveLatentObs is FALSE.
#' @param quiet Boolean. If FALSE, information will be printed to the terminal including
#' current iteration, current value of K and number of items per cluster.
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
#' DPMUnc(obsData, obsVars, "test_output", 1234)
DPMUnc <- function(obsData,obsVars,saveFileDir,seed,
                   K=floor(nrow(obsData)/2), nIts = 100000, thinningFreq = 10,
                   saveClusterParams=TRUE, saveLatentObs=FALSE,
                   quiet=TRUE) {
  if(!dir.exists(saveFileDir)) {
      dir.create(saveFileDir, recursive=TRUE)
  }

  set.seed(seed, sample.kind="Rejection", normal.kind="Inversion", kind="Mersenne-Twister")

  scaled <- scale_data(obsData, obsVars)
  obsData <- scaled$data
  obsVars <- scaled$vars

  # start in an odd place. convergence longer, but avoids getting stuck in kmeans solution
  kmeansInit             <- kmeans(obsData, centers = K)
  currentAllocations     <- sample(kmeansInit$cluster)

  runDPMUnc(obsData, obsVars,
            nIts, thinningFreq, quiet,
            saveClusterParams, saveLatentObs,
            saveFileDir,
            currentAllocations)
}
