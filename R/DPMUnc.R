scale_data <- function(obsData, obsVars) {
    scaledData <- scale(obsData, scale=TRUE, center=TRUE)
    scaledVars <- scale(obsVars,
                        # Don't center the variances! This wouldn't make any sense!
                        center=FALSE,
                        # Scale the variances by the square of the scale factors we used for the data
                        scale=attr(scaledData, "scaled:scale") ** 2)

    return (list("data"=scaledData, "vars"=scaledVars))
}

DPMUnc <- function(obsData,obsVars,saveFileDir,unique_id,
                   K=floor(nrow(obsData)/2), nIts = 100000, thinningFreq = 10,
                   saveClusterParams=TRUE, saveLatentObs=FALSE,
                   verbose=FALSE,quiet=TRUE){

  set.seed(unique_id, sample.kind="Rejection", normal.kind="Inversion", kind="Mersenne-Twister")

  scaled <- scale_data(obsData, obsVars)
  obsData <- scaled$data
  obsVars <- scaled$vars

  kmeansInit             <- kmeans(obsData, centers = K)
  currentAllocations     <- sample(kmeansInit$cluster) # start in an odd place. convergence longer, but avoids getting stuck in kmeans solution

  runDPMUnc(obsData, obsVars,
              nIts, thinningFreq, quiet,
              saveClusterParams, saveLatentObs,
              saveFileDir,
              currentAllocations)

  return(saveFileDir)

}
