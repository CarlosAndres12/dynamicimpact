VECTOR_MODEL <-  "vector_model"
VECTOR_MODEL_CHOLESKY <-  "vector_model_cholesky"
MATRIX_MODEL <-  "matrix_model"

.get_model <- function(model_type=c(names(stanmodels))) {

  model_type <- match.arg(model_type, several.ok=FALSE)

  result <- stanmodels[[model_type]]

  return(result)

}


.create_model <- function(stan_data, model_type=c(names(stanmodels)),
                          chains, iter, thin, cores) {

  model_type <- match.arg(model_type, several.ok=FALSE)

  stan_model <- .get_model(model_type)

  fitted_model <- rstan::sampling(stan_model,
                                  data = stan_data,
                                  chains = chains,
                                  iter=iter,
                                  thin=thin,
                                  cores = cores)

  extracted_data <- rstan::extract(fitted_model)

  result <- list(
    "stan_fit"  = fitted_model,
    "extracted_data" = extracted_data
  )

  attr(result, "class") <- paste0("impact_", model_type)


  return(result)


}


