functions {

  vector relu(vector input) {

    int N_SIZE = dims(input)[1];

    vector[N_SIZE] result;



    for(i in 1:N_SIZE) {

      if(input[i] > 0) {
        result[i] = input[i];
      } else {
        result[i] = 0;
      }

    }

    return result;

  }


  // taken from  https://discourse.mc-stan.org/t/multi-normal-lcdf/6652/3
  vector[] cholesky_truncate_normal_base(vector mu, matrix L, vector b, vector s, vector u) {
    int K = rows(mu); vector[K] d; vector[K] z; vector[K] out[2];
    for (k in 1:K) {
      int km1 = k - 1;
      if (s[k] != 0) {
        real z_star = (b[k] -
                      (mu[k] + ((k > 1) ? L[k,1:km1] * head(z, km1) : 0))) /
                      L[k,k];
        real v; real u_star = Phi(z_star);
        if (s[k] == -1) {
          v = u_star * u[k];
          d[k] = u_star;
        }
        else {
          d[k] = 1 - u_star;
          v = u_star + d[k] * u[k];
        }
        z[k] = inv_Phi(v);
      }
      else {
        z[k] = inv_Phi(u[k]);
        d[k] = 1;
      }
    }
    out[1] = z;
    out[2] = d;
    return out;
  }

  // taken from https://spinkney.github.io/helpful_stan_functions/group__tmvn.html
  real multi_normal_cholesky_truncated_lpdf(vector u, vector mu, matrix L, vector lb,
                                          vector ub, vector lb_ind, vector ub_ind) {
  int K = rows(u);
  vector[K] z;
  real lp = 0;

  for (k in 1 : K) {
    // if kth u is unbounded
    // else kth u has at least one bound
    if (lb_ind[k] == 0 && ub_ind[k] == 0)
      z[k] = inv_Phi(u[k]);
    else {
      int km1 = k - 1;
      real v;
      real z_star;
      real logd;
      row_vector[2] log_ustar = [negative_infinity(), 0]; // [-Inf, 0] = log([0,1])
      real constrain = mu[k] + ((k > 1) ? L[k, 1 : km1] * head(z, km1) : 0);

      // obtain log of upper and lower bound (if applicable)
      if (lb_ind[k] == 1)
        log_ustar[1] = normal_lcdf((lb[k] - constrain) / L[k, k] | 0.0, 1.0);
      if (ub_ind[k] == 1)
        log_ustar[2] = normal_lcdf((ub[k] - constrain) / L[k, k] | 0.0, 1.0);

      // update log gradient and z
      logd = log_diff_exp(log_ustar[2], log_ustar[1]);
      v = exp(log_sum_exp(log_ustar[1], log(u[k]) + logd)); // v = ustar[1] + (ustar[2] - ustar[1]) * u[k] ~ U(ustar[1], ustar[2])
      z[k] = inv_Phi(v); // z ~ TN
      lp += logd; // increment by log gradient
    }
  }
  return lp;
}

// taken from https://spinkney.github.io/helpful_stan_functions/group__tmvn.html
vector multi_normal_cholesky_truncated_rng(vector u, vector mu, matrix L, vector lb,
                                           vector ub, vector lb_ind, vector ub_ind) {
  int K = rows(u);
  vector[K] z;

  for (k in 1 : K) {
    // if kth u is unbounded
    // else kth u has at least one bound
    if (lb_ind[k] == 0 && ub_ind[k] == 0)
      z[k] = inv_Phi(u[k]);
    else {
      int km1 = k - 1;
      real v;
      real z_star;
      real logd;
      row_vector[2] log_ustar = [negative_infinity(), 0]; // [-Inf, 0] = log([0,1])
      real constrain = mu[k] + ((k > 1) ? L[k, 1 : km1] * head(z, km1) : 0);

      // obtain log of upper and lower bound (if applicable)
      if (lb_ind[k] == 1)
        log_ustar[1] = normal_lcdf((lb[k] - constrain) / L[k, k] | 0.0, 1.0);
      if (ub_ind[k] == 1)
        log_ustar[2] = normal_lcdf((ub[k] - constrain) / L[k, k] | 0.0, 1.0);

      // update log gradient and z
      logd = log_diff_exp(log_ustar[2], log_ustar[1]);
      v = exp(log_sum_exp(log_ustar[1], log(u[k]) + logd)); // v = ustar[1] + (ustar[2] - ustar[1]) * u[k] ~ U(ustar[1], ustar[2])
      z[k] = inv_Phi(v); // z ~ TN
    }
  }
  return mu + L * z;
}

// taken from https://spinkney.github.io/helpful_stan_functions/group__tmvn.html
void multi_normal_cholesky_truncated_lp(vector u, vector mu, matrix L, vector lb,
                                        vector ub, vector lb_ind, vector ub_ind) {
  int K = rows(u);
  vector[K] z;

  for (k in 1 : K) {
    // if kth u is unbounded
    // else kth u has at least one bound
    if (lb_ind[k] == 0 && ub_ind[k] == 0)
      z[k] = inv_Phi(u[k]);
    else {
      int km1 = k - 1;
      real v;
      real z_star;
      real logd;
      row_vector[2] log_ustar = [negative_infinity(), 0]; // [-Inf, 0] = log([0,1])
      real constrain = mu[k] + ((k > 1) ? L[k, 1 : km1] * head(z, km1) : 0);

      // obtain log of upper and lower bound (if applicable)
      if (lb_ind[k] == 1)
        log_ustar[1] = normal_lcdf((lb[k] - constrain) / L[k, k] | 0.0, 1.0);
      if (ub_ind[k] == 1)
        log_ustar[2] = normal_lcdf((ub[k] - constrain) / L[k, k] | 0.0, 1.0);

      // update log gradient and z
      logd = log_diff_exp(log_ustar[2], log_ustar[1]);
      v = exp(log_sum_exp(log_ustar[1], log(u[k]) + logd)); // v = ustar[1] + (ustar[2] - ustar[1]) * u[k] ~ U(ustar[1], ustar[2])
      z[k] = inv_Phi(v); // z ~ TN
      target += logd; // increment by log gradient
    }
  }
}



  // rtruncnorm <- function(n, mu, sigma, low, high) {
  // # find quantiles that correspond the the given low and high levels.
  // p_low <- pnorm(low, mu, sigma)
  // p_high <- pnorm(high, mu, sigma)
  //
  // # draw quantiles uniformly between the limits and pass these
  // # to the relevant quantile function.
  // qnorm(runif(n, p_low, p_high), mu, sigma)
  // }

  // truncate normal
  vector truncate_multi_normal_rng(real lowerbound,  vector mu, matrix Sigma) {

    int N_SIZE = dims(mu)[1];

    vector[N_SIZE] result;

    result = multi_normal_rng(mu, Sigma);

    return result;

  }

  vector pow_vector(vector input, vector powers) {

    int N_SIZE = dims(input)[1];

    vector[N_SIZE] result;

    for(i in 1:N_SIZE) {
      result[i] = pow(input[i], powers[i]);
    }

    return result;

  }

  vector abs_vector(vector input) {

    int N_SIZE = dims(input)[1];

    vector[N_SIZE] result;

    for(i in 1:N_SIZE) {
      result[i] = fabs(input[i]);
    }

    return result;

  }

  vector[] vector_abs(vector[] input) {

    int ARRAY_LENGTH = dims(input)[1];
    int VECTOR_LENGTH = dims(input)[2];

    vector[VECTOR_LENGTH] result[ARRAY_LENGTH];

    for(i in 1:ARRAY_LENGTH) {
      result[i] = abs_vector(input[i]);
    }

    return result;

  }

  real elastic_net_regularizaion_vector(vector params, real alpha1, real alpha2) {

    int N_SIZE = dims(params)[1];

    real result;

    real squere_part = alpha1*sum(pow_vector(params, rep_vector(2.0, N_SIZE)));

    real abs_part = alpha2*sum(abs_vector(params));

    result = squere_part + abs_part;

    return result;

  }

  real elastic_net_regularizaion(vector[] params, real alpha1, real alpha2) {

    int ARRAY_LENGTH = dims(params)[1];
    int VECTOR_LENGTH = dims(params)[2];

    real result = 0;

    for(j in 1:VECTOR_LENGTH) {

      result += elastic_net_regularizaion_vector( to_vector(params[,j]), alpha1, alpha2);

    }

    return result;

  }

  // taken from https://jrnold.github.io/ssmodels-in-stan/stan-functions.html
  // caclulates the kronecker product of two matrices as defined in https://en.wikipedia.org/wiki/Kronecker_product
  matrix kronecker_prod(matrix A, matrix B) {

    matrix[rows(A) * rows(B), cols(A) * cols(B)] C = rep_matrix(0.0, rows(A) * rows(B), rows(A) * rows(B) );
    int m;
    int n;
    int p;
    int q;

    m = rows(A);
    n = cols(A);
    p = rows(B);
    q = cols(B);

    if( (rows(A) == 1) && (cols(A) == 1) ) {
      return (A[1][1] * B);
    }

    if( (rows(B) == 1) && (cols(B) == 1) ) {
      return (B[1][1] * A);
    }

    for (i in 1:m) {

      for (j in 1:n) {
        int row_start;
        int row_end;
        int col_start;
        int col_end;
        row_start = (i - 1) * p + 1;
        row_end = (i - 1) * p + p;
        col_start = (j - 1) * q + 1;
        col_end = (j - 1) * q + q;
        C[row_start:row_end, col_start:col_end] = A[i, j] * B;

      }
    }

    return C;
  }



  real log_matrix_normal_density(matrix y_real, matrix y_est, matrix V, matrix U) {

    real det_V = determinant(V);
    real det_U = determinant(U);


    // use inverse_spd for speed as the V and U matrices must be semi-positive definite by definition
    matrix[dims(V)[1], dims(V)[2]] inv_V  = inverse_spd(V);
    matrix[dims(U)[1], dims(U)[2]] inv_U  = inverse_spd(U);

    int n = dims(y_real)[1]; // R
    int p = dims(y_real)[2]; // K


    matrix[p, p] temp_matrix = (-1/2.0) * (inv_V * ((y_real - y_est)') * inv_U * (y_real - y_est));


    real result_value = trace(temp_matrix) - log( pow(2.0*pi(), 1.0*n*p/2.0) ) - log(pow(det_V, 1.0*n/2.0)) - log(pow(det_U, 1.0*p/2.0));

    return(result_value);
  }

  // takes as input an array of matrices and returns the an array of "cumsum matrices"
  matrix[] cumsum_matrix(matrix[] input_matrix) {

    matrix[dims(input_matrix)[2], dims(input_matrix)[3]] result_matrix[dims(input_matrix)[1]];

    for(t in 1:(dims(input_matrix)[1]) ) {

      if(t == 1) {
        result_matrix[t] = input_matrix[t];
      } else {
        result_matrix[t] = result_matrix[t-1] + input_matrix[t];
      }

    }
    return(result_matrix);
  }


  vector[] cumsum_vector(vector[] input_vector) {

    int N_SIZE = dims(input_vector)[1];

    vector[dims(input_vector)[2]] result_vector[N_SIZE];

    result_vector[1] = input_vector[1];

    for(t in 2:(N_SIZE) ) {
       result_vector[t] = result_vector[t - 1] + input_vector[t];

    }
    return(result_vector);
  }

  vector scale_vector(vector input_vector, int use_log) {

    int N_SIZE = dims(input_vector)[1];

    vector[N_SIZE] result_vector;
    vector[N_SIZE] log_input_vector;

    if(use_log) {
      log_input_vector = log(input_vector);
      result_vector = (log_input_vector - mean(log_input_vector))/sd(log_input_vector);
    } else {
      result_vector = (input_vector - mean(input_vector))/sd(input_vector);
    }


    return(result_vector);

  }



  vector unscale_vector(vector input_vector, vector original_vector, int use_log) {

    int N_SIZE = dims(input_vector)[1];
    int N_SIZE_ORIGINAL = dims(original_vector)[1];

    vector[N_SIZE_ORIGINAL] log_original_vector;
    vector[N_SIZE] result_vector;


    if(use_log) {

      log_original_vector = log(original_vector);

      result_vector = (sd(log_original_vector)*input_vector) + mean(log_original_vector);
      result_vector = exp(result_vector);


    } else {
      result_vector = (sd(original_vector)*input_vector) + mean(original_vector);
    }



    return(result_vector);

  }

  real[] unscale_array(real[] input_vector, real[] original_vector, int use_log) {

    int N_SIZE = dims(input_vector)[1];

    real result_vector[N_SIZE];

    result_vector = to_array_1d(unscale_vector(to_vector(input_vector) , to_vector(original_vector), use_log));


    return(result_vector);

  }

  matrix scale_matrix(matrix input_matrix, int use_log) {

    int N_ROW = dims(input_matrix)[1];
    int N_COL = dims(input_matrix)[2];


    matrix[N_ROW, N_COL] result_matrix;

    for(j in 1:N_COL) {

      result_matrix[,j] =  scale_vector(input_matrix[,j], use_log);

    }

    return(result_matrix);

  }

  matrix unscale_matrix(matrix input_matrix, matrix original_matrix, int use_log) {

    int N_ROW = dims(input_matrix)[1];
    int N_COL = dims(input_matrix)[2];


    matrix[N_ROW, N_COL] result_matrix;

    for(j in 1:N_COL) {

      result_matrix[,j] =  unscale_vector(input_matrix[,j], original_matrix[,j], use_log);

    }

    return(result_matrix);

  }

 matrix[] scale_matrix_array(matrix[] input_matrix_array, int use_log) {

    int ARRAY_LENGTH = dims(input_matrix_array)[1];
    int N_ROW = dims(input_matrix_array)[2];
    int N_COL = dims(input_matrix_array)[3];


    matrix[N_ROW, N_COL] result_matrix_array[ARRAY_LENGTH];

    for(j in 1:ARRAY_LENGTH) {

      result_matrix_array[j] =  scale_matrix(input_matrix_array[j], use_log);

    }

    return(result_matrix_array);

  }

   matrix[] unscale_matrix_array(matrix[] input_matrix_array,
                                 matrix[] original_matrix_array, int use_log) {

    int ARRAY_LENGTH = dims(input_matrix_array)[1];
    int N_ROW = dims(input_matrix_array)[2];
    int N_COL = dims(input_matrix_array)[3];


    matrix[N_ROW, N_COL] result_matrix_array[ARRAY_LENGTH];

    for(j in 1:ARRAY_LENGTH) {

      result_matrix_array[j] =  unscale_matrix(input_matrix_array[j], original_matrix_array[j], use_log);

    }

    return(result_matrix_array);

  }


  vector[] scale_vector_array(vector[] input_vector_array, int use_log) {

    int ARRAY_LENGTH = dims(input_vector_array)[1];
    int VECTOR_LENGTH = dims(input_vector_array)[2];


    vector[VECTOR_LENGTH] result_array[ARRAY_LENGTH];

    for(j in 1:ARRAY_LENGTH) {

      result_array[j] =  scale_vector(input_vector_array[j], use_log);

    }

    return(result_array);

  }


  vector[] unscale_vector_array(vector[] input_vector_array,
                                vector[] original_vector_array, int use_log) {

    int ARRAY_LENGTH = dims(input_vector_array)[1];
    int VECTOR_LENGTH = dims(input_vector_array)[2];


    vector[VECTOR_LENGTH] result_array[ARRAY_LENGTH];

    for(j in 1:VECTOR_LENGTH) {

      result_array[,j] =  unscale_array(input_vector_array[,j], original_vector_array[,j], use_log);

    }

    return(result_array);

  }

  matrix cov_matrix_correction(matrix input_matrix)  {

    int N_ROW = dims(input_matrix)[1];
    int N_COL = dims(input_matrix)[2];

    matrix[N_ROW, N_COL] result_matrix_array;

    result_matrix_array = (0.5*(input_matrix + input_matrix ')) + diag_matrix(rep_vector(1e-9, N_ROW));


    return result_matrix_array;


  }




}


