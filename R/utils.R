.get_credible_interval <- function(m_array, credibility_level, global=FALSE) {

  temp_array <- m_array

  if(global) {

    temp_array <- m_array |>  apply(c(1,2), mean)

  }

  temp_ic <-
    apply(temp_array, c(2),
          function(x, ci) {
            result <- bayestestR::hdi(x, ci=ci)
            return ( c(result$CI_low, result$CI_high))
          }, ci=credibility_level)



  result_df <- data.frame(value = temp_array |> apply(2, median))
  result_df$time_index = 1:nrow(result_df)

  result_df$lower_limit <-  temp_ic[1,]
  result_df$upper_limit <-  temp_ic[2,]




  return(result_df)


}



