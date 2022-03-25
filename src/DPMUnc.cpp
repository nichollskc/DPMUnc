#include <RcppArmadillo.h>
#include <cmath>
#include <math.h>
#include "ezETAProgressBar.h"

// A smaller number corresponds to a more serious and less verbose debug statement
// So setting VERBOSITY here to a larger number means increasing the number of messages printed
#ifndef VERBOSITY
#  define VERBOSITY 3
#endif

// DEBUG macro which prints line info etc. and will only be printed if the
// assigned verbosity X is less than VERBOSITY
#define DEBUG(X,Y) \
  if(X <= VERBOSITY) { \
    std::cerr << "DEBUG " << X << " " \
              << __FILE__ << "::" << __FUNCTION__ << "::L" << __LINE__ << " "; \
    do {std::cerr << Y << std::endl;} while(0); \
  };

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// Append a single line containing the given vector, with elements separated
// by commas. Append to the file stream given.
void append_vec_cs(arma::vec vector, std::ofstream& file_stream) {
  int final_index = vector.size() - 1;
  for(int j=0; j < final_index; j++) {
    file_stream << vector[j] << ",";
  }
  if (final_index > 0) {
    file_stream << vector[final_index];
  }
  file_stream << std::endl;
}

// Append a single line containing the given vector, with elements separated
// by commas. Append to the file stream given.
void append_uvec_cs(arma::uvec vector, std::ofstream& file_stream) {
  int final_index = vector.size() - 1;
  for(int j=0; j < final_index; j++) {
    file_stream << vector[j] << ",";
  }
  if (final_index > 0) {
    file_stream << vector[final_index];
  }
  file_stream << std::endl;
}

arma::vec colMeans(arma::mat X) {
  arma::mat means_mat = arma::mean(X, 0);
  arma::vec means = arma::conv_to<arma::vec>::from(means_mat);
  return means;
}

int random_integer_weighted(arma::vec weights) {
  arma::vec cumSums = arma::cumsum(weights);
  DEBUG(10, weights.t());
  double u = arma::randu() * cumSums[cumSums.n_rows - 1];
  DEBUG(10, "Random number " << u << " cumSums " << cumSums.t());
  arma::uvec indices_above_u = arma::find(cumSums > u);
  DEBUG(10, indices_above_u.t());
  int first_above_u = indices_above_u[0];
  return first_above_u;
}

// We can generate values from a Beta distribution using two Gammas
// See https://en.wikipedia.org/wiki/Beta_distribution#Related_distributions
double rBeta(double a, double b) {
  double theta = 1.0;
  double X = arma::randg( arma::distr_param(a, theta) );
  double Y = arma::randg( arma::distr_param(b, theta) );
  double beta = X / (X + Y);
  return(beta);
}

arma::mat sample_normal_mat(arma::mat mean, arma::mat var) {
  arma::mat z = arma::randn( mean.n_rows, mean.n_cols );
  return z % arma::sqrt(var) + mean;
}

arma::vec sample_normal(arma::vec mean, arma::vec var) {
  arma::vec z = arma::randn( mean.size() );
  return z % sqrt(var) + mean;
}

// Alpha is shape parameter, beta is rate parameter
arma::vec sample_gamma(double alpha, arma::vec beta) {
  // Armadillo uses shape and scale, so need 1/beta
  return arma::randg( beta.size(), arma::distr_param(alpha, 1.0) ) / beta;
}

arma::uvec shift_cluster_allocations(arma::uvec current, int indexToRemove) {
  //  We have deleted cluster indToRemove, so we need to shift all labels higher than this value
  arma::uvec labelsToShift = arma::find(current > indexToRemove);
  // Calculate shifted labels - shift down by 1
  arma::uvec shiftedAllocations = current.rows(labelsToShift) - 1;

  // Create a new allocation vector with the necessary labels shifted
  arma::uvec newAllocations = current;
  newAllocations.rows(labelsToShift) = shiftedAllocations;
  return newAllocations;
}

double sample_alpha(double a0, double b0, int K, int nObs, double currentAlpha) {
  double newAlpha;

  DEBUG(7, "First sampling eta using currentAlpha " << currentAlpha << " nObs " << nObs);
  // sample a new alpha (see Escobar and West, 1995)
  double eta = rBeta(currentAlpha + 1, nObs);

  // Escobar & West gives (pi_eta/(1 - pi_eta)) = A/B, where A & B are:
  double A = a0 + K - 1;
  double B = nObs * (b0 - log(eta));

  // Rearranging, we have pi_eta = A/(A+B)
  double pi_eta = A / ((double) (A + B));

  double scale = 1 / ((double) b0 - log(eta));

  DEBUG(5, "Sampling alpha A " << A << " B " << B << " pi_eta " << pi_eta << " scale "  << scale << " nObs " << nObs << " eta " << eta);

  if (arma::randu() < pi_eta) {
    newAlpha = arma::randg( arma::distr_param(a0 + K, scale) );
  } else {
    newAlpha = arma::randg( arma::distr_param(a0 + K - 1, scale) );
  }

  return newAlpha;
}

class NormalGammaDistributionPosterior {
  private:
    double kappa, alpha;
    arma::vec beta;
    arma::vec mu;

  public:
    NormalGammaDistributionPosterior(double kappa,
                                     double alpha,
                                     arma::vec beta,
                                     arma::vec mu)
      : kappa(kappa), alpha(alpha), beta(beta), mu(mu) {}

    std::string as_str() {
      std::ostringstream oss;
      oss << "Alpha " << alpha;
      oss << "\nKappa " << kappa;
      oss << "\nBeta" << beta.t();
      oss << "Mu" << mu.t();
      return oss.str();
    }

    void sample_from_dist(arma::vec *means,
                          arma::vec *vars) {
      *vars = 1 / sample_gamma(this->alpha, this->beta);
      *means = sample_normal(this->mu, *vars * (1 / this->kappa));
    }
};

class NormalGammaDistribution {
  private:
    int numDim;
    double kappa, alpha, beta;
    arma::vec mu;
    // Log of normalisation constant
    double logZ;

  public:
    NormalGammaDistribution(double kappa,
                            double alpha,
                            double beta,
                            arma::vec mu)
      : kappa(kappa), alpha(alpha), beta(beta), mu(mu) {
        numDim = mu.n_rows;
        logZ = lgamma(alpha) + log(2 * M_PI) / 2 - log(kappa) / 2 - alpha * log(beta);
      }

    NormalGammaDistributionPosterior calcPosterior(int numObservations,
                                                   arma::vec observedMeans,
                                                   arma::vec observedRSS) {
      double kappaN = this->kappa + numObservations;
      double alphaN = this->alpha + ((double) numObservations/2);

      arma::vec muN = (this->kappa * this->mu + numObservations * observedMeans) / kappaN;

      arma::vec betaN(numDim);
      const double interaction_constant = this->kappa * numObservations / (2 * kappaN);
      for(int j=0; j < numDim; j++) {
        betaN[j] = this->beta + observedRSS[j] / 2 + interaction_constant * pow((observedMeans[j] - this->mu[j]), 2);
      }

      return NormalGammaDistributionPosterior(kappaN, alphaN, betaN, muN);
    }

    // Calculate the log marginal likelihood of some observed Gaussian data
    // using this Normal-Gamma distribution as the prior
    // This will be logZn - logZ0 - logZl
    // where:
    //  logZ0 is the normalisation constant of the prior (i.e. this
    //    Normal-Gamma distribution)
    //  logZl is the normalisation constant of the likelihood function (i.e.
    //    numObservations independent Gaussians)
    //  logZn is normalisation constant of posterior (after seeing these
    //    numObservations new observations)
    double calcLogMarginalLikelihood(int numObservations,
                                     arma::vec observedMeans,
                                     arma::vec observedRSS) {
      double kappaN = this->kappa + numObservations;
      double alphaN = this->alpha + ((double) numObservations/2);

      double logMarginalLikelihood = 0;
      double betaN, logZn, logEv;
      double logZl = ((numObservations/(double) 2) * log(2 * M_PI));
      const double interaction_constant = this->kappa * numObservations / (2 * kappaN);
      const double logZn_constant = lgamma(alphaN) + log(2 * M_PI) / 2 - log(kappaN) / 2;
      for(int j=0; j < numDim; j++) {
        betaN = this->beta + observedRSS[j] / 2 + interaction_constant * pow((observedMeans[j] - this->mu[j]), 2);
        logZn = logZn_constant - alphaN * log(betaN);
        logEv = logZn - this->logZ - logZl;
        logMarginalLikelihood += logEv;
      }

      return logMarginalLikelihood;
    }

    double calcLogMarginalLikelihoodSingleton(arma::vec observation) {
      arma::vec empiricalRSS = arma::zeros(numDim);
      return calcLogMarginalLikelihood(1, observation, empiricalRSS);
    }
};

class Clustering {
  private:
    NormalGammaDistribution* clusterParamPrior;

    // Length K vectors
    arma::vec nObsInCluster;
    arma::vec logMarginalLikelihood;
    // K x d matrices
    arma::mat empiricalMeans;
    arma::mat empiricalRSS;

    // K x d matrices that ease calculation of Means and RSS
    arma::mat empiricalSum;
    arma::mat empiricalSumSquares;

    void update_MeanRSS() {
      int K = nObsInCluster.n_rows;
      for (int k = 0; k < K; k++) {
        update_MeanRSS_cluster(k);
      }
    }

    void update_MeanRSS_cluster(int k) {
      empiricalMeans.row(k) = empiricalSum.row(k) / nObsInCluster[k];
      empiricalRSS.row(k) = empiricalSumSquares.row(k) - \
                            arma::square(empiricalSum.row(k)) / nObsInCluster[k];
    }

    void update_logMarginalLikelihood(int k) {
      DEBUG(7, "Updating log likelihood for cluster " << k);
      logMarginalLikelihood[k] = clusterParamPrior->calcLogMarginalLikelihood(nObsInCluster[k],
                                                                              empiricalMeans.row(k).t(),
                                                                              empiricalRSS.row(k).t());
      DEBUG(7, "Updated log likelihood for cluster " << k);
    }

    void update_cluster(int k,
                        int change_nObs,
                        arma::vec change_Sum,
                        arma::vec change_SumSquares) {
      int numDim = change_Sum.n_rows;
      nObsInCluster[k] += change_nObs;

      if (nObsInCluster[k] == 0) {
        empiricalSum.row(k) = arma::zeros(1, numDim);
        empiricalSumSquares.row(k) = arma::zeros(1, numDim);
        empiricalMeans.row(k) = arma::zeros(1, numDim);
        empiricalRSS.row(k) = arma::zeros(1, numDim);
        logMarginalLikelihood[k] = 0;
      } else {
        empiricalSum.row(k) += change_Sum.t();
        empiricalSumSquares.row(k) += change_SumSquares.t();
        update_MeanRSS_cluster(k);
        update_logMarginalLikelihood(k);
      }
    }

  public:
    Clustering(NormalGammaDistribution* clusterParamPrior,
               arma::vec nObsInCluster,
               arma::mat empiricalMeans,
               arma::mat empiricalRSS)
      : clusterParamPrior(clusterParamPrior), empiricalMeans(empiricalMeans),
        empiricalRSS(empiricalRSS), nObsInCluster(nObsInCluster) {
      int K = nObsInCluster.n_rows;
      empiricalSum = arma::diagmat(nObsInCluster) * empiricalMeans;
      empiricalSumSquares = empiricalRSS + arma::diagmat(1 / nObsInCluster) * arma::square(empiricalSum);
      logMarginalLikelihood = arma::zeros(K);
      for (int k = 0; k < K; k++) {
        update_logMarginalLikelihood(k);
      }
    }

    Clustering(NormalGammaDistribution* clusterParamPrior,
               arma::uvec clusterAllocations,
               arma::mat dataPoints) : clusterParamPrior(clusterParamPrior) {
      int K = arma::max(clusterAllocations);
      int numDim = dataPoints.n_cols;

      nObsInCluster = arma::zeros(K);
      logMarginalLikelihood = arma::zeros(K);
      empiricalSum = arma::zeros(K, numDim);
      empiricalSumSquares = arma::zeros(K, numDim);
      empiricalMeans = arma::zeros(K, numDim);
      empiricalRSS = arma::zeros(K, numDim);

      for (int k = 0; k < K; k++) {
        int r_k = k + 1;
        arma::mat dataInCluster = dataPoints.rows(arma::find(clusterAllocations == r_k));
        nObsInCluster[k] = dataInCluster.n_rows;
        empiricalSum.row(k) = arma::sum(dataInCluster, 0);
        empiricalSumSquares.row(k) = arma::sum(arma::square(dataInCluster), 0);

        if (nObsInCluster[k] == 1) {
          empiricalMeans.row(k) = dataInCluster;
          empiricalRSS.row(k) = arma::zeros(1, numDim);
        } else {
          update_MeanRSS_cluster(k);
        }
        update_logMarginalLikelihood(k);
      }
    }

    void add_observation_to_every_cluster(arma::vec observation) {
      int K = getK();
      nObsInCluster += 1;

      // Add observation to each cluster's sum
      empiricalSum.each_row() += observation.t();

      // Add square of observation to each cluster's sum of squares
      empiricalSumSquares.each_row() += arma::square(observation).t();

      update_MeanRSS();
      for (int k = 0; k < K; k++) {
        update_logMarginalLikelihood(k);
      }
    }

    void add_observation_to_cluster(int k, arma::vec observation) {
      update_cluster(k, 1, observation, arma::square(observation));
    }

    void remove_observation_from_cluster(int k, arma::vec observation) {
      update_cluster(k, -1, -observation, -arma::square(observation));
    }

    void add_singleton_cluster(arma::vec observation) {
      int K = getK();
      int numDim = observation.n_rows;

      empiricalMeans.resize(K + 1, numDim);
      empiricalMeans.row(K) = observation.t();

      empiricalRSS.resize(K + 1, numDim);
      empiricalRSS.row(K) = arma::zeros(1, numDim);

      empiricalSum.resize(K + 1, numDim);
      empiricalSum.row(K) = observation.t();

      empiricalSumSquares.resize(K + 1, numDim);
      empiricalSumSquares.row(K) = arma::square(observation).t();

      logMarginalLikelihood.resize(K + 1);
      logMarginalLikelihood[K] = clusterParamPrior->calcLogMarginalLikelihoodSingleton(observation);

      nObsInCluster.resize(K + 1);
      nObsInCluster[K] = 1;
    }

    void remove_empty_cluster(int k) {
      empiricalMeans.shed_row(k);
      empiricalRSS.shed_row(k);
      empiricalSum.shed_row(k);
      empiricalSumSquares.shed_row(k);

      nObsInCluster.shed_row(k);
      logMarginalLikelihood.shed_row(k);
    }

    int getK() {
      return empiricalMeans.n_rows;
    }

    arma::vec getNObsInCluster() {
      return nObsInCluster;
    }

    int getNObsInCluster(int k) {
      int nObs = 0;
      if (k < nObsInCluster.size()) {
        nObs = nObsInCluster[k];
      }
      return nObs;
    }

    arma::vec getLogMarginalLikelihood() {
      return logMarginalLikelihood;
    }

    arma::vec getEmpiricalMeans(int k) { return empiricalMeans.row(k).t(); }
    arma::vec getEmpiricalRSS(int k) { return empiricalRSS.row(k).t(); }

    std::string as_str() {
      std::ostringstream oss;
      oss << "Clustering\n";
      oss << "**NumberObs\n**" << nObsInCluster.t();
      oss << "**LogMarginalLikelihood of Cluster\n**" << logMarginalLikelihood.t();
      oss << "**Empirical Means\n" << empiricalMeans;
      oss << "**Empirical RSS\n" << empiricalRSS;
      oss << "**Empirical Sum\n" << empiricalSum;
      oss << "**Empirical SumSquares\n" << empiricalSumSquares;
      return oss.str();
    }

};

class MixtureModellerOutputter {
  private:
    std::ofstream file_ClusterMeans;
    std::ofstream file_ClusterVars;
    std::ofstream file_Allocations;
    std::ofstream file_Latents;
    std::ofstream file_Alpha;
    std::ofstream file_pLatentsGivenClusters;
    std::ofstream file_K;

    bool quiet;
    bool saveClusterParams;
    bool saveLatentObs;
    int totalIterations, thinningFreq;

    ez::ezETAProgressBar progressBar;

  public:
    MixtureModellerOutputter(int totalIterations,
                             int thinningFreq,
                             bool quiet,
                             bool saveClusterParams,
                             bool saveLatentObs,
                             std::string outputDir) :
          file_ClusterMeans(outputDir + "/clusterMeans.csv"),
          file_ClusterVars(outputDir + "/clusterVars.csv"),
          file_Allocations(outputDir + "/clusterAllocations.csv"),
          file_Latents(outputDir + "/latentObservations.csv"),
          file_pLatentsGivenClusters(outputDir + "/pLatentsGivenClusters.csv"),
          file_K(outputDir + "/K.csv"),
          totalIterations(totalIterations), thinningFreq(thinningFreq),
          saveClusterParams(saveClusterParams), saveLatentObs(saveLatentObs),
          quiet(quiet), progressBar(totalIterations/thinningFreq) {
      file_Alpha.open(outputDir + "/alpha.csv", std::ios::app);
      if (quiet) {
        progressBar.start();
      }
    }

    void print_update(int iterations,
                      int K,
                      double alpha_concentration,
                      arma::vec clusterCounts) {
      if (iterations % thinningFreq == 0) {
        if (quiet) {
          ++progressBar;
        } else {
          Rcout << iterations << " of " << totalIterations << " iterations\n";
          Rcout << K << " clusters\n";
          Rcout << alpha_concentration << " alpha\n";
          Rcout << "counts per cluster\n" << clusterCounts.t() << "\n";
        }
      }
    }

    void save_sample_to_file(int iterations,
                             arma::mat clusterMeans,
                             arma::mat clusterVars,
                             arma::uvec clusterAllocations,
                             int K,
                             arma::mat latentObservations,
                             double alpha_concentration,
                             double pLatentsGivenClusters) {
      if (iterations % thinningFreq == 0) {
        append_uvec_cs(clusterAllocations, file_Allocations);
        file_Alpha << alpha_concentration << std::endl;
        file_pLatentsGivenClusters << pLatentsGivenClusters << std::endl;

        file_K << K << std::endl;

        if (saveClusterParams) {
          append_vec_cs(clusterMeans.as_col(), file_ClusterMeans);
          append_vec_cs(clusterMeans.as_col(), file_ClusterVars);
        }
        if (saveLatentObs) {
          append_vec_cs(latentObservations.as_col(), file_Latents);
        }

        Rcpp::checkUserInterrupt();
      }
    }
};

class MixtureModeller {
  private:
    arma::mat observedData, observedVars;
    // Priors for alpha (Escobar and West)
    double a0, b0;

    MixtureModellerOutputter outputter;

    // Store p(z|c) - the marginal "likelihood" of the latents given the clustering
    double pLatentsGivenClusters;

    double alpha_concentration;
    NormalGammaDistribution clusterParamPrior;
    Clustering currentClustering;
    arma::uvec clusterAllocations;

    // Clusterings we use for calculation - allocation and freeing seemed to be a bottleneck
    Clustering clusteringExcludingObs;
    Clustering clusteringWithObsInEveryCluster;

    // K x d matrix
    arma::mat latentObservations;

    // K x d matrices
    arma::mat clusterMeans;
    arma::mat clusterVars;

    void move_observation(int i,
                          int r_oldCluster,
                          int r_newCluster,
                          arma::vec observation) {
      if (r_oldCluster != r_newCluster) {
        _move_observation(i, r_oldCluster, r_newCluster, observation);
      }
    }

    void _move_observation(int i,
                           int r_oldCluster,
                           int r_newCluster,
                           arma::vec observation) {
      int newCluster = r_newCluster - 1;
      int oldCluster = r_oldCluster - 1;

      int K = currentClustering.getK();

      clusterAllocations[i] = r_newCluster;
      if (newCluster == K) {
        DEBUG(4, "Adding singleton cluster with observation " << i );
        currentClustering.add_singleton_cluster(observation);
      } else {
        DEBUG(4, "Adding observation " << i << " to cluster " << newCluster);
        currentClustering.add_observation_to_cluster(newCluster, observation);
      }

      DEBUG(4, "Removing observation " << i << " from cluster " << oldCluster);
      currentClustering.remove_observation_from_cluster(oldCluster, observation);

      if (currentClustering.getNObsInCluster(oldCluster) == 0) {
        clusterAllocations = shift_cluster_allocations(clusterAllocations, r_oldCluster);
        currentClustering.remove_empty_cluster(oldCluster);
      }

    }

    double resample_alpha() {
      int K = currentClustering.getK();
      int nObs = observedData.n_rows;
      DEBUG(7, "Original alpha " << alpha_concentration);
      alpha_concentration = sample_alpha(a0, b0, K, nObs, alpha_concentration);
      DEBUG(5, "Resampled alpha " << alpha_concentration);
      return alpha_concentration;
    }

    void resample_cluster_params() {
      int K = currentClustering.getK();
      int numDim = observedData.n_cols;
      clusterMeans.resize(K, numDim);
      clusterVars.resize(K, numDim);
      arma::vec newMeans(numDim);
      arma::vec newVars(numDim);

      for (int k = 0; k < K; k++) {
        NormalGammaDistributionPosterior posterior = clusterParamPrior.calcPosterior(currentClustering.getNObsInCluster(k),
                                                                                     currentClustering.getEmpiricalMeans(k),
                                                                                     currentClustering.getEmpiricalRSS(k));
        DEBUG(7, " k " << k << "\n" << posterior.as_str());
        posterior.sample_from_dist(&newMeans,
                                   &newVars);
        clusterMeans.row(k) = newMeans.t();
        clusterVars.row(k) = newVars.t();
      }
      DEBUG(7, clusterMeans << "\n" << clusterVars );
    }

    void resample_latent_observations() {
      int nObs = observedData.n_rows;
      int d = observedData.n_cols;

      arma::mat mu_n_matrix(nObs, d);
      arma::mat sigmasq_n_matrix(nObs, d);

      int clusterInd;
      double x, sumx, mu_n2, sigmasq_n2;
      for(int i = 0; i < nObs; i++) {
        clusterInd = clusterAllocations[i] - 1;

        for(int j = 0; j < d; j++) {
          x          = observedData(i,j);
          sumx       = x;
          sigmasq_n2 = 1/((1/clusterVars(clusterInd,j)) + (1/observedVars(i,j)));
          mu_n2      = sigmasq_n2 *( (clusterMeans(clusterInd,j)/clusterVars(clusterInd,j)) + (sumx/observedVars(i,j)) );
          mu_n_matrix(i, j)      = mu_n2;
          sigmasq_n_matrix(i, j) = sigmasq_n2;
        }
      }
      DEBUG(10, mu_n_matrix << "\n" << sigmasq_n_matrix);

      latentObservations = sample_normal_mat(mu_n_matrix, sigmasq_n_matrix);
    }

    void calculateClusterStats() {
      arma::mat dataPoints = latentObservations;
      DEBUG(7, "before\n" << currentClustering.as_str());
      DEBUG(7, "data points\n" << dataPoints);
      Clustering current(&(clusterParamPrior),
                         clusterAllocations,
                         dataPoints);
      currentClustering = current;
      DEBUG(7, "after\n" << currentClustering.as_str());
    }

    void print_update(int iterations) {
      int K = currentClustering.getK();
      arma::vec counts = currentClustering.getNObsInCluster();
      outputter.print_update(iterations, K, alpha_concentration, counts);
    }

    void save_sample_to_file(int iterations) {
      // To calculate p(z|c), sum up marginal log likelihoods for each cluster
      double pLatentsGivenClusters = arma::sum(currentClustering.getLogMarginalLikelihood());
      int K = currentClustering.getK();
      outputter.save_sample_to_file(iterations,
                                    clusterMeans,
                                    clusterVars,
                                    clusterAllocations,
                                    K,
                                    latentObservations,
                                    alpha_concentration,
                                    pLatentsGivenClusters);
    }

    arma::vec calc_allocation_probs(arma::vec observation,
                                    int r_currentAlloc) {
      DEBUG(7, "alpha " << alpha_concentration);
      int currentAlloc = r_currentAlloc - 1;

      DEBUG(7, currentClustering.as_str());
      clusteringExcludingObs = currentClustering;
      clusteringExcludingObs.remove_observation_from_cluster(currentAlloc, observation);
      DEBUG(7, "excluding obs\n" << clusteringExcludingObs.as_str());

      clusteringWithObsInEveryCluster = clusteringExcludingObs;
      clusteringWithObsInEveryCluster.add_observation_to_every_cluster(observation);
      DEBUG(7, "obs in every\n" << clusteringWithObsInEveryCluster.as_str());

      int K = clusteringWithObsInEveryCluster.getK();
      arma::vec logMarginalLikelihoodObs(K + 1);
      arma::vec clusterSizeWeights(K + 1);

      arma::vec logMarginalLikelihoodObsOverPosterior = clusteringWithObsInEveryCluster.getLogMarginalLikelihood() - \
                                                        clusteringExcludingObs.getLogMarginalLikelihood();
      DEBUG(7, "logMarginalLikelihoodObsOverPosterior\n" << logMarginalLikelihoodObsOverPosterior.t());
      logMarginalLikelihoodObs.head_rows(K) = logMarginalLikelihoodObsOverPosterior;
      logMarginalLikelihoodObs[K] = clusterParamPrior.calcLogMarginalLikelihoodSingleton(observation);

      clusterSizeWeights.head_rows(K) = clusteringExcludingObs.getNObsInCluster();
      clusterSizeWeights[K] = alpha_concentration;

      // Adjust by this constant factor to keep values in a more normal range
      double constant_factor = arma::max(logMarginalLikelihoodObs);
      arma::vec clusterProbs = clusterSizeWeights % arma::exp(logMarginalLikelihoodObs - constant_factor);
      DEBUG(5, "Cluster probs\n" << clusterProbs.t());
      clusterProbs = arma::normalise(clusterProbs);

      return clusterProbs;
    }

    void update_cluster_allocations(int i,
                                    int r_oldCluster,
                                    int r_newCluster,
                                    arma::vec observation) {
      DEBUG(4, "Attempting move from (R indexing) " << r_oldCluster << " to " << r_newCluster << " of obs " << i);
      DEBUG(7, "before\n" << currentClustering.as_str());
      move_observation(i, r_oldCluster, r_newCluster, observation);
      DEBUG(7, "after\n" << currentClustering.as_str());
    }

    void resample_allocations() {
      arma::vec observation;
      int r_currentAlloc;
      for (int i = 0; i < latentObservations.n_rows; i++) {
        observation = latentObservations.row(i).t();
        r_currentAlloc = clusterAllocations[i];
        arma::vec allocation_probs = calc_allocation_probs(observation, r_currentAlloc);
        int r_selectedCluster = random_integer_weighted(allocation_probs) + 1;
        update_cluster_allocations(i, r_currentAlloc, r_selectedCluster, observation);
      }
    }

    void show_progress(int iterations) {
      print_update(iterations);
      save_sample_to_file(iterations);
    }

    void next_iteration(int iterations) {
      resample_allocations();

      // Having completed the Gibbs sampling for the component indicator
      // variables, we now sample a new alpha (see Escobar and West, 1995)
      resample_alpha();

      resample_cluster_params();

      resample_latent_observations();

      // Now need to update the component-specific statistics:
      calculateClusterStats();

      show_progress(iterations);
    }

    void run_iterations(int totalIterations) {
      for (int i = 1; i <= totalIterations; i++) {
        next_iteration(i);
      }
    }

  public:
    MixtureModeller(arma::mat observedData,
                    arma::mat observedVars,
                    int totalIterations,
                    int thinningFreq,
                    bool quiet,
                    bool saveClusterParams,
                    bool saveLatentObs,
                    std::string outputDir,
                    arma::uvec clusterAllocations,
                    double kappa0 = 0.01,
                    double alpha0 = 2,
                    double beta0 = 0.1)
        : observedData(observedData), observedVars(observedVars),
          latentObservations(observedData),
          clusterAllocations(clusterAllocations),
          a0(3), b0(4),
          outputter(totalIterations, thinningFreq, quiet, saveClusterParams, saveLatentObs, outputDir),
          clusterParamPrior(kappa0, alpha0, beta0, colMeans(observedData)),
          currentClustering(&clusterParamPrior, clusterAllocations, observedData),
          clusteringExcludingObs(&clusterParamPrior, clusterAllocations, observedData),
          clusteringWithObsInEveryCluster(&clusterParamPrior, clusterAllocations, observedData),
          alpha_concentration(1) {
      calculateClusterStats();
      run_iterations(totalIterations);
    }

};

// DPMUnc - Run Dirichlet Process Mixture Modeller taking uncertainty of data points into account
//
// @param observedData The observed data in matrix form (n observations x p variables)
// @param observedVars The observed variances of the data (n observations x p variables)
// @param totalIterations Total number of iterations to run. The user should check they are happy
// that the model has converged before using any of the results.
// @param thinningFreq Controls how many samples are saved. E.g. a value of 10 means
// every 10th sample will be saved.
// @param quiet Boolean. If FALSE, information will be printed to the terminal including
// current iteration, current value of K and number of items per cluster.
// @param saveClusterParams Boolean, determining whether the cluster parameters (mean 
// and variance of every cluster) for every saved iteration should be saved in a file or not.
// Both cluster parameters and latent observations take up more space than other saved variables.
// @param saveLatentObs Boolean, determining whether the latent observations (underlying true observations)
// for every saved iteration should be saved in a file or not. Both cluster parameters and
// latent observations take up more space than other saved variables.
// @param outputDir Directory where all output will be saved
// @param clusterAllocations Initial cluster allocations.
//
// @export
//
// [[Rcpp::export]]
void runDPMUnc(arma::mat observedData,
               arma::mat observedVars,
               int totalIterations,
               int thinningFreq,
               bool quiet,
               bool saveClusterParams,
               bool saveLatentObs,
               std::string outputDir,
               arma::uvec clusterAllocations) {
  DEBUG(4, "Initialised modeller with data\n" << observedData)
  MixtureModeller(observedData,
                  observedVars,
                  totalIterations,
                  thinningFreq,
                  quiet,
                  saveClusterParams,
                  saveLatentObs,
                  outputDir,
                  clusterAllocations);
}
