#include include/utils.stan

data {



  int<lower=0> T; // total of observations
  int<lower=0> T0;

  int<lower=0> K; // number of stations
  int<lower=0> P; // number of regresors

  vector[K] Y[T];
  vector[P] X[T];

  int<lower=0, upper=1> use_log;
  int<lower=0, upper=1> scale;

  // real<lower=0> obs_scale;
  // real<lower=0> evol_scale;

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
  cov_matrix[K] sigma_entry_obs;
  vector[P*K] theta_vec[T0];
  cov_matrix[P*K] theta_vec_cov_matrix;
}


model {

  vector[K] mu[T0];

  //  https://mc-stan.org/docs/2_22/stan-users-guide/multivariate-outcomes.html
  for (t in 1:T0) {
    mu[t] = (to_matrix(theta_vec[t], P, K)') * X[t]  ;
  }

  sigma_entry_obs ~ inv_wishart(1.0*K, diag_matrix(rep_vector(100, K)) );

  theta_vec_cov_matrix ~ inv_wishart(1.0*K*P, diag_matrix(rep_vector(100, K*P)) );

  theta_vec[1]          ~  multi_normal(rep_vector(0, K*P), cov_matrix_correction(theta_vec_cov_matrix) );
  theta_vec[2:T0] ~  multi_normal(theta_vec[1:(T0-1)], cov_matrix_correction(theta_vec_cov_matrix));

  Y[1:T0] ~ multi_normal(mu[1:T0] , cov_matrix_correction(sigma_entry_obs));
}


generated quantities {

  vector[K] scaled_Y_pred[T]; // matricial version of y
  vector[K] Y_pred[T]; // matricial version of y
  vector[K*P] theta_vec_pred[T]; // matricial version of theta
  vector[K] mu[T];

  vector[K] difference[T];
  vector[K] cumsum_difference[T]; // array of cumsum matrices, it starts of the initial time.
  vector[K] cumsum_only_after[T]; // array of cumsum matrices, it starts after the intervention.
  vector[K] arco_only_after[T]; //
  vector[T] arco_only_after_aggregated;


  scaled_Y_pred[1:T0] = Y[1:T0];
  theta_vec_pred[1:T0] = theta_vec[1:T0];

  for (t in (T0+1):T) {

    theta_vec_pred[t] = theta_vec[T0];
  }

  for (t in 1:T) {
    mu[t] = (to_matrix(theta_vec_pred[t], P, K)') * X[t]  ;
  }


  scaled_Y_pred = multi_normal_rng(mu, cov_matrix_correction(sigma_entry_obs));


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



