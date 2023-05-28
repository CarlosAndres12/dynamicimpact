#include include/utils.stan

data {

  int<lower=1> T; //End time index
  int<lower=1, upper=T> T0; // event time index


  int<lower=0> K; // number of stations
  int<lower=0> P; // number of regresors

  vector<lower=0> [K] Y[T];
  vector<lower=0> [P] X[T];

  int<lower=0, upper=1> use_log;
  int<lower=0, upper=1> scale;
  int<lower=0, upper=1> force_positive;

  vector[K] lb;
  vector[K] ub;


  // s[k] ==  0 implies no constraint; otherwise
  // s[k] == -1 -> b[k] is an upper bound
  // s[k] == +1 -> b[k] is a lower bound
  vector<lower=-1,upper=1>[K] s;

  real<lower=0> obs_scale;
  real<lower=0> evol_scale;

  real<lower=0> eta_obs;
  real<lower=0> eta_evol;



}

transformed data {

  int T_after = T - T0; // number of times after the

  vector[P] X_before[T0];   // predictor matrix
  vector[P] X_after[T_after];   // predictor matrix

  vector[P] X_fit[T];
  vector[P] X_fit_before[T0];   // predictor matrix
  vector[P] X_fit_after[T_after];   // predictor matrix

  vector[K] Y_before[T0];   // predictor matrix
  vector[K] Y_after[T_after];   // predictor matrix

  vector[K] Y_fit_before[T0];   // predictor matrix
  vector[K] Y_fit_after[T_after];   // predictor matrix
  vector[K] Y_fit[T];   // predictor matrix

  X_before = X[1:T0];
  X_after  = X[(T0+1):T];

  Y_before = Y[1:T0];
  Y_after  = Y[(T0+1):T];

  if(scale) {

      Y_fit_before = scale_vector_array(Y, use_log)[1:T0];
      Y_fit_after  = scale_vector_array(Y, use_log)[(T0+1):T];
      Y_fit        = scale_vector_array(Y, use_log);

      X_fit_before = scale_vector_array(X, use_log)[1:T0];
      X_fit_after  = scale_vector_array(X, use_log)[(T0+1):T];
      X_fit    = scale_vector_array(X, use_log);

  } else {

    if(use_log) {

      Y_fit_before = log(Y_before);
      Y_fit_after = log(Y_fit_after);
      Y_fit = log(Y_fit);

      X_fit_before = log(X_before);
      X_fit_after = log(X_after);
      X_fit = log(X);

    } else {

      Y_fit_before = Y_before;
      Y_fit_after  = Y_after;
      Y_fit = Y;

      X_fit_before = X_before;
      X_fit_after  = X_after;
      X_fit = X;
    }

  }


}



parameters {

  cholesky_factor_cov[K] sigma_entry_cholesky;
  vector<lower=0>[K] tau;
  vector<lower=0>[P*K] sigma;


  vector[P*K] theta_vec[T0];



  cholesky_factor_cov[P*K] theta_vec_cholesky_matrix;


  vector<lower=0,upper=1>[K] u;

}

transformed parameters {

  matrix[K,K] L = diag_pre_multiply(tau, sigma_entry_cholesky);
  matrix[P*K,P*K] L_evol = diag_pre_multiply(sigma, theta_vec_cholesky_matrix);

}


model {



  vector[K] mu[T0];
  tau ~ cauchy(0, obs_scale);
  sigma ~ cauchy(0, evol_scale);
  sigma_entry_cholesky ~ lkj_corr_cholesky(eta_obs);


  for (t in 1:T0) {

    if(force_positive) {
       mu[t] = relu((to_matrix(theta_vec[t], P, K)') * X_before[t]) ;
    } else {
       mu[t] = (to_matrix(theta_vec[t], P, K)') * X_before[t];
    }
   }


  theta_vec_cholesky_matrix ~ lkj_corr_cholesky(eta_evol);


  theta_vec[1]    ~  multi_normal_cholesky(rep_vector(0, K*P) , L_evol);
  theta_vec[2:T0] ~ multi_normal_cholesky(theta_vec[1:(T0-1)] , L_evol);


 if(force_positive) {
    for(i in 1:T0) {
      target += log(cholesky_truncate_normal_base(Y_fit_before[i], L, lb, s, u)[2]);
      target += multi_normal_cholesky_lpdf(Y_fit_before[i] | mu, L);
    }


 } else {
   Y_fit_before ~ multi_normal_cholesky(mu, L);
 }


}

generated quantities {

  vector[K] scaled_Y_pred[T]; // matricial version of y
  vector[K] Y_pred[T]; // matricial version of y
  vector[K*P] theta_vec_pred[T]; // matricial version of theta
  vector[K] mu_pred[T];

  vector[K] difference[T];
  vector[K] cumsum_difference[T]; // array of cumsum matrices, it starts of the initial time.
  vector[K] cumsum_only_after[T]; // array of cumsum matrices, it starts after the intervention.
  vector[K] arco_only_after[T]; // Artificial counter factual
  vector[T] arco_only_after_aggregated;





  theta_vec_pred[1:T0] = theta_vec;

  for (t in (T0+1):T) {
    theta_vec_pred[t] = theta_vec[T0];
  }

  for (t in 1:T) {

    if(force_positive) {
      mu_pred[t] = relu((to_matrix(theta_vec_pred[t], P, K)') * X_fit[t]);
    } else {

      mu_pred[t] = to_matrix(theta_vec_pred[t], P, K)' * X_fit[t];
    }

  }


  if(force_positive) {

    for(t in 1:T) {
    scaled_Y_pred[t] = mu_pred[t] + L * cholesky_truncate_normal_base(mu_pred[t], L, lb, s, u)[1];


    }

  } else {
    scaled_Y_pred = multi_normal_cholesky_rng(mu_pred, sigma_entry_cholesky);
  }



  if(scale) {

    Y_pred = unscale_vector_array(scaled_Y_pred,  Y, use_log);
  } else {
    if(use_log) {
      Y_pred = exp(scaled_Y_pred);
    } else {
      Y_pred = scaled_Y_pred;
    }
  }


  for (t in 1:T) {
    difference[t] = Y[t] - Y_pred[t];
  }

  cumsum_only_after[1:T0] = rep_array(rep_vector(0, K), T0);
  cumsum_only_after[(T0+1):T] = cumsum_vector(difference[(T0+1):T]);

  cumsum_difference = cumsum_vector(difference);

  arco_only_after[1:T0] =  rep_array(rep_vector(0, K), T0);
  arco_only_after_aggregated[1:T0] = rep_vector(0, T0);


  for(t in (T0+1):T) {
    arco_only_after[t] = (1.0/(t-T0)) * cumsum_only_after[t];
    arco_only_after_aggregated[t] = mean(arco_only_after[t]);
  }



}



