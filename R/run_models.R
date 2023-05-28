.get_dates_df <- function(X_data, dates) {

  if(is.null(dates)) {
    return(dates)
  }

  time_index  <- 1:(dim(X_data)[1])

  if(!lubridate::is.Date(dates) ) {
    stop("the dates are not invalid")
  }

  if(length(time_index) != length(time_index)) {

    stop("the legend of dates is different from the first dimension of the X_data and Y_data matrices")

  }

  dates_df  <- data.frame(
    "time_index" = time_index,
    "Date" = dates

  )

  return(dates_df)

}


.validate_input <- function(X_data,
                            Y_data,
                            event_initial,
                            log_transform,
                            scale,
                            force_positive,
                            model_type=c(VECTOR_MODEL, MATRIX_MODEL),
                            credibility_level) {

  model_type <- match.arg(model_type, several.ok=FALSE)


  if( scale & force_positive) {
    stop("a standarized vector can not be force to be positive")
  }


  if( scale & force_positive) {
    stop("a log vector can not be force to be positive")
  }

  if(log_transform) {

    if(any(Y_data <= 0) ) {
      stop("when using log elements of Y must be positive")
    }

  }


  if(is.null(dim(Y_data))) {
    stop('Incorrect Y_data dimensions')
  }

  if(is.null(dim(X_data))) {
    stop('Incorrect X_data dimensions')
  }

  if(model_type %in% c(VECTOR_MODEL, VECTOR_MODEL_CHOLESKY)) {

    if( !(length(dim(Y_data)) %in% c(2)) ) {
      stop('Incorrect Y_data dimensions')

    }

    if( !(length(dim(X_data)) %in% c(2)) ) {
      stop('Incorrect X_data dimensions')

    }

  } else {

    if( !(length(dim(Y_data)) %in% c(3)) ) {
      stop('Incorrect Y_data dimensions')

    }

    if( !(length(dim(X_data)) %in% c(3)) ) {
      stop('Incorrect X_data dimensions')

    }
  }


  if(any(is.na(Y_data))) {
    stop("There should be no missing values in Y.")
  }

  if(any(is.na(X_data))) {
    stop("There should be no missing values in X.")
  }

  if(dim(Y_data)[1] != dim(X_data)[1]) {
    stop("The length of the time series(first dim of X and Y) is diferent.")
  }




  if(model_type == MATRIX_MODEL) {

    if(dim(Y_data)[2] != dim(X_data)[2]) {
      stop("The number of rows(second dim of X and Y) is diferent.")
    }

    if(dim(Y_data)[3] != dim(X_data)[3]) {
      stop("The third dim of X and Y is diferent.")
    }

    if(force_positive) {
      stop("force positive is only avalible for the vector model")
    }

  }

  if( (credibility_level >= 1) |  (credibility_level <= 0)) {
    stop("Confidence level must be between 0 and 1")
  }




}


.get_stan_data <- function(model, X_data, Y_data, event_initial,
                           log_transform, scale,
                           evolution_scale = 1, observation_scale = 1,
                           force_positive=FALSE,
                           keep_theta_static_for_prediction = TRUE) {


  if(model %in% c(VECTOR_MODEL, VECTOR_MODEL_CHOLESKY)) {
    K <- Y_data |> ncol()
    P <- X_data |> ncol()
    R <- 0
  } else {
    R <- dim(Y_data)[2]
    K <- dim(Y_data)[3]
    P <- dim(X_data)[3]
  }



  stan_data = list(

    "T" = dim(X_data)[1],
    "T0" = event_initial,

    "R" = R,
    "P" = P,
    "K" = K,

    "Y" = Y_data,
    "X" = X_data,

    "use_log" = log_transform |> as.integer(),
    "scale" = scale |> as.integer(),
    "force_positive" = force_positive |> as.integer(),

    "obs_scale" = observation_scale,
    "evol_scale" = evolution_scale,

    "eta_obs" = 1,
    "eta_evol" = 1,

    "keep_theta_static_for_prediction" = keep_theta_static_for_prediction |> as.integer()

  )

  stan_data$lb  <-  rep(0, stan_data$K)
  stan_data$ub  <-  rep(0, stan_data$K)

  stan_data$s <- rep(1, stan_data$K)

  return(stan_data)


}

#' @name ImpactModelVector
#' @title ImpactModelVector
#' @description
#' This function creates models in which \code{Y_data[t,]} is a
#' vector with \code{q} elements The posterior distribution for this
#' model is approximated using \code(stan).
#' @references
#' \insertRef{quintana1987multivariate}{dynamicimpact}
#'
#' \insertRef{west1997bayesian}{dynamicimpact}
#'
#' \insertRef{brodersen_inferring_2015}{dynamicimpact}
#' @details
#' Create a new \code{ImpactModelVector} object.
#'
#' @param  Y_data: response matrix.
#' @param  X_data: covariates matrix.
#' @param  event_initial: The time when the event starts.
#' @param  credibility_level: level of the credible interval.
#' @param  variables_names: name of each of the components of vector (only used in plots).
#' @param  n_simul: number of the iteration for MCMC, half of the iteration will be burned, default 2000.
#' @param  n_chains: number of chains for MCMC, default 4.
#' @param  n_cores: number of cores used in MCMC, default 4.
#' @param  thin: thinning used in MCMC, default 1.
#' @param log_transform: make \code{Y_data = log(Y_data)} during the fit, the transformation will be reversed for the final result. It will force force the posterior of the predictive distribution to be positive.
#' @param scale: standardize \code(Y_data) during the fit, the transformation will be reversed for the final result.
#' @param force_positive: It will a use a truncated normal distribution with lower limit at zero, the mean of \code{Y_data} will transformed using \code{ReLU}.
#' @param dates: An optional vector of dates only use for plotting.
#' @return An object containing the the fitted \code{stan} model the extracted data from the chains
#' @examples
#' \donttest{
#' data("X_vector", package="dynamicimpact")
#' data("Y_vector", package="dynamicimpact")
#' result <- ImpactModelVector(X_data=X_vector, Y_data=Y_vector, event_initial=97, credibility_level=0.89)
#'
#' }
#' @export
ImpactModelVector <- function(X_data,
                              Y_data,
                              event_initial,
                              credibility_level,
                              evolution_scale = 1,
                              observation_scale = 1,
                              use_cholesky = FALSE,
                              variables_names=NULL,
                              n_simul=2000,
                              n_chains=4,
                              n_cores=4,
                              thin=1,
                              log_transform = FALSE,
                              scale= FALSE,
                              force_positive=FALSE,
                              dates=NULL
                              ) {



  .validate_input(

    X_data=X_data,
    Y_data=Y_data,
    event_initial=event_initial,
    log_transform=log_transform,
    scale=scale,
    force_positive=force_positive,
    model_type=VECTOR_MODEL,
    credibility_level=credibility_level

  )

  if(is.null(variables_names)) {
    variables_names  <- paste0("Variable ", 1:ncol(Y_data))
  }

  dates <-  .get_dates_df(X_data=X_data, dates=dates)

  vec_model <- ifelse(use_cholesky, VECTOR_MODEL_CHOLESKY, VECTOR_MODEL)

  stan_data <- .get_stan_data(model=vec_model,
                              X_data=X_data,
                              Y_data=Y_data,
                              event_initial=event_initial,
                              log_transform=log_transform,
                              scale=scale,
                              evolution_scale = evolution_scale,
                              observation_scale = observation_scale,
                              force_positive=force_positive,
                              keep_theta_static_for_prediction = TRUE)

  model  <-  .create_model(
    stan_data, model_type=vec_model,
    chains=n_chains, iter=n_simul, thin=thin, cores=n_cores
  )

  model$dates <- dates
  model$stan_data <- stan_data
  model$credibility_level  <- credibility_level
  model$variables_names <- variables_names

  return(model)

}



#' @name ImpactModelMatrix
#' @title ImpactModelMatrix
#' @description
#' This function creates models in which \code{Y_data[t,,]} is a
#' matrix with \code{r} rows and \q{columns}, The posterior distribution for this
#' model is approximated using \code(stan).
#' @references
#' \insertRef{quintana1987multivariate}{dynamicimpact}
#'
#' \insertRef{west1997bayesian}{dynamicimpact}
#'
#' \insertRef{brodersen_inferring_2015}{dynamicimpact}
#' @details
#' Create a new \code{ImpactModelMatrix} object.
#'
#' @param  Y_data: response matrix.
#' @param  X_data: covariates matrix.
#' @param  event_initial: The time when the event starts.
#' @param  credibility_level: level of the credible interval.
#' @param  variables_names: name of each of the components of vector (only used in plots).
#' @param  n_simul: number of the iteration for MCMC, half of the iteration will be burned, default 2000.
#' @param  n_chains: number of chains for MCMC, default 4.
#' @param  n_cores: number of cores used in MCMC, default 4.
#' @param  thin: thinning used in MCMC, default 1.
#' @param log_transform: make \code{Y_data = log(Y_data)} during the fit, the transformation will be reversed for the final result. It will force force the posterior of the predictive distribution to be positive.
#' @param scale: standardize \code(Y_data) during the fit, the transformation will be reversed for the final result.
#' @param dates: An optional vector of dates only use for plotting.
#' @return An object containing the the fitted stan model the extracted data from the chains
#' @examples
#' \donttest{
#' data("X_matrix", package="dynamicimpact")
#' data("Y_matrix", package="dynamicimpact")
#' result <- ImpactModelMatrix(X_data=X_matrix, Y_data=Y_matrix, event_initial=97, credibility_level=0.89)
#'
#' }
#' @export
ImpactModelMatrix <- function(X_data,
                              Y_data,
                              event_initial,
                              credibility_level,
                              variables_names=NULL,
                              n_simul=2000,
                              n_chains=4,
                              n_cores=4,
                              thin=1,
                              log_transform = FALSE,
                              scale= FALSE,
                              dates=NULL) {


   if(is.null(variables_names)) {
     variables_names <-
       expand.grid(paste0("R", 1:dim(Y_data)[2]), paste0("C", 1:dim(Y_data)[3])) |>
       data.frame() |>
       dplyr::mutate(elem = paste0(Var1, "-", Var2 ) ) |>
       dplyr::pull(elem) |>
       matrix(nrow=dim(Y_data)[2],byrow = F)
   }


  .validate_input(

    X_data=X_data,
    Y_data=Y_data,
    event_initial=event_initial,
    log_transform=log_transform,
    scale=scale,
    force_positive=FALSE,
    model_type=MATRIX_MODEL,
    credibility_level=credibility_level

  )

  dates <-  .get_dates_df(X_data=X_data, dates=dates)

  stan_data <- .get_stan_data(model=MATRIX_MODEL,
                              X_data=X_data,
                              Y_data=Y_data,
                              event_initial=event_initial,
                              log_transform=log_transform,
                              scale=scale,
                              force_positive=FALSE,
                              keep_theta_static_for_prediction = TRUE)

  model  <-  .create_model(
    stan_data, model_type=MATRIX_MODEL,
    chains=n_chains, iter=n_simul, thin=thin, cores=n_cores
  )

  model$dates <- dates
  model$stan_data <- stan_data
  model$credibility_level  <- credibility_level
  model$variables_names <- variables_names

  return(model)

}
