#include include/utils.stan

data {

  int T; // total of observations
  int T0;


  int K; // number of stations
  int P; // number of regresors
  int R; // number of sensors



  matrix[R, K] y[T0]; // array of response matrices
  matrix[R, P] X[T0]; // array of input matrices


  int<lower=0, upper=1> keep_theta_static_for_prediction; // stop the evoluation of theta after the intervention

  int<lower=0, upper=1> use_log;
  int<lower=0, upper=1> scale;

}

transformed data {

  int T_after = T - T0; // number of times after the

  vector[R*K] y_vector[T]; // requiered to use the kronecker product, rstan does not support the normal matrix variate distribution

  // vector version of the observation to use the multivariate normal distribution
  for(t in 1:T) {

    if(scale) {

      y_vector[t] = to_vector(scale_matrix(y[t], use_log));

    } else {

      if(use_log) {

        y_vector[t] = to_vector(log(y[t]));

      } else {
        y_vector[t] = to_vector(y[t]);
      }


    }


  }

}



parameters {



  cov_matrix[R]  sigma_entry_obs_rows; // v_t
  cov_matrix[K]  sigma_entry_obs_cols; // Sigma


  cov_matrix[P] level_sigma_rows; // w_t
  cov_matrix[K] level_sigma_cols;


  vector[P*K] theta[T0]; // theta is defined as vector, to use the multivariate normal distribution
}





model {

  vector[R*K]  mu[T0];


  // initilize mu.

  // mu is also a vector to. Because X is matrix  theta is converted to matrix form
  // before the multiplication, the result is then converted to a vector
  mu[1] = to_vector( X[1] * to_matrix(theta[1], P, K) );

   // repeat for all times before the intervention
   for (t in 2:T0) {

      mu[t] =   to_vector( ( X[t] * to_matrix(theta[t], P, K)) );

   }

  // only use a distribution for sigma_entry_obs_stations if the user dones not pass a matrix
  sigma_entry_obs_cols ~ inv_wishart(1.0*K, diag_matrix(rep_vector(100, K)) );


  // only use a distribution for sigma_entry_obs_sensores if the user dones not pass a matrix

  sigma_entry_obs_rows  ~ inv_wishart(1.0*R, diag_matrix(rep_vector(100, R)) );



  level_sigma_rows     ~ inv_wishart(1.0*P, diag_matrix(rep_vector(100, P)) );
  level_sigma_cols     ~ inv_wishart(1.0*K, diag_matrix(rep_vector(100, K)) );



  // repeat for all times before the intervention
  for(t in 1:T0) {

    if(t <= 1) {

    // initilization
    theta[t] ~ multi_normal(rep_vector(0.0, P*K), cov_matrix_correction(kronecker_prod(diag_matrix(rep_vector(1, P)), diag_matrix(rep_vector(1, K)))) );

    } else {


      // kronecker_prod is use to be able to fit the  multivariate normal distribution
      theta[t] ~ multi_normal(theta[t-1], cov_matrix_correction(kronecker_prod(level_sigma_rows,     level_sigma_cols)));

    }
  }

  // repeat for all times before the intervention
  for (t in 2:T0) {

     // kronecker_prod is use to be able to fit the  multivariate normal distribution
     y_vector[t] ~  multi_normal(mu[t], cov_matrix_correction(kronecker_prod(sigma_entry_obs_rows, sigma_entry_obs_cols)));

  }

}


generated quantities {

  //moved from transformed parameters
  vector[R*K]  mu[T0];
  //

  // these variables are computed after MCMC
  matrix[R, K] scaled_Y_pred[T]; // scaled matricial version of y
  matrix[R, K] Y_pred[T]; // matricial version of y
  matrix[P, K] theta_pred[T]; // matricial version of theta
  matrix[R, K] mu_pred[T]; // matricial version of mu
  matrix[R, K] difference[T]; // array of diference matrices
  matrix[R, K] cumsum_difference[T]; // array of cumsum matrices, it starts of the initial time.
  matrix[R, K] cumsum_only_after[T]; // array of cumsum matrices, it starts after the intervention.


  matrix[R, K] arco_only_after[T]; // Artificial counter factual
  vector[T] arco_only_after_aggregated;

  // mu is also a vector to. Because X is matrix  theta is converted to matrix form
  // before the multiplication, the result is then converted to a vector
  mu[1] = to_vector( X[1] * to_matrix(theta[1], P, K) );

   // repeat for all times before the intervention
   for (t in 2:T0) {

      mu[t] =   to_vector( ( X[t] * to_matrix(theta[t], P, K)) );

   }


  // get all paramters in their correct dimension
  for(t in 1:T0) {

    theta_pred[t] = to_matrix(theta[t], P, K);

    mu_pred[t] = to_matrix(mu[t], R, K);

    // repeat the processs in the models block.
    // here we need also need to use multi_normal_rng to reproduce the variation

    scaled_Y_pred[t] = to_matrix(multi_normal_rng(to_vector( mu_pred[t] ), cov_matrix_correction(kronecker_prod(sigma_entry_obs_rows, sigma_entry_obs_cols))), R, K);

    // diference variable
    difference[t] = y[t] - scaled_Y_pred[t];

    // the cumsum_only_after is the cero matrix for all times before the intervention
    cumsum_only_after[t] = to_matrix(rep_vector(0, R*K), R, K);
  }

  // predictions
  for(t in (T0+1):T) {

    // the variable keep_theta_static_for_prediction controls if theta is evolves after the intervention or not.
    if(keep_theta_static_for_prediction) {

      // if theta does not evolves after the prediction, make it equal to theta in the time before the intervention.
      theta_pred[t] = theta_pred[T0];

    } else {

      theta_pred[t] = to_matrix(multi_normal_rng(to_vector(theta_pred[t-1]), cov_matrix_correction(kronecker_prod(level_sigma_rows, level_sigma_cols))), P, K);
    }

    mu_pred[t] = (X[t] * theta_pred[t]);

    scaled_Y_pred[t] = to_matrix(multi_normal_rng(to_vector( mu_pred[t] ), cov_matrix_correction(kronecker_prod(level_sigma_rows, level_sigma_cols))), R, K);



    if(scale) {

      Y_pred[t] = unscale_matrix(scaled_Y_pred[t], y[t], use_log);

    } else {

      if(use_log) {

        Y_pred[t] = exp(scaled_Y_pred[t]);

      } else {
        Y_pred[t] = scaled_Y_pred[t];
      }

    }

    // diference variable
    difference[t] = y[t] - Y_pred[t];

  }

  // compute cumsums
  cumsum_difference                 = cumsum_matrix(difference);
  cumsum_only_after[(T0+1):T] = cumsum_matrix(difference[(T0+1):T]);


  for(t in (T0+1):T) {

    arco_only_after[t] = (1.0/(t-T0)) * cumsum_only_after[t];
    arco_only_after_aggregated[t] = mean(arco_only_after[t]);

  }
}



