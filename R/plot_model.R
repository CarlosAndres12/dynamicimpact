.get_break_dates  <- function(max_date, event_initial, min_date=1, length.out=5, min_date_dist=5) {

  m_break_dates  <- seq(min_date,max_date, length.out=length.out) |> floor()



  remove_index <- abs(m_break_dates - event_initial) > min_date_dist
  remove_index <-  (remove_index)

  m_break_dates <- m_break_dates[remove_index]

  m_break_dates  <- c(event_initial, m_break_dates)

  return(m_break_dates)


}


.build_plot_df = function(fitted_model, model_data, event_initial ,
                          group_names, credibility_level,  global_name=NULL) {



  df_list <- list()
  df_list_aggregate <- list()


  if(is.null(global_name)) {
    global_name <- "global"
  }

  global_name <-  global_name[1]

  for(m_index in 1:(length(group_names)+1)  ) {

    is_global <- m_index > length(group_names)


    if(is_global) {
      m_variable_name <-  global_name
      idx = 1:length(group_names)
    } else {
      idx = m_index
      m_variable_name <-  group_names[idx]
    }


    m_df_stan_pred <- fitted_model$extracted_data$Y_pred[,,idx]  |>
      .get_credible_interval(credibility_level=credibility_level,
                             global = is_global)

    m_df_stan_pred$variable <- m_variable_name
    m_df_stan_pred$type <- "prediction"
    m_df_stan_pred$class <- "prediction"

    if(is_global) {

      m_df_real <- data.frame(

        variable=m_variable_name,
        time_index = 1:(dim(model_data$X)[1]),
        value = model_data$Y |> apply(c(1), mean),
        type = "prediction",
        class = 'real'

      )

      m_df_real_input <- data.frame(

        variable=m_variable_name,
        time_index = 1:(dim(model_data$X)[1]),
        value = model_data$X |> apply(c(1), mean),
        type = "prediction",
        class = 'input'

      )

    } else {

      m_df_real <- data.frame(

        variable=m_variable_name,
        time_index = 1:(dim(model_data$X)[1]),
        value = model_data$Y[, idx],
        type = "prediction",
        class = 'real'

      )

      m_df_real_input <- data.frame(

        variable=m_variable_name,
        time_index = 1:(dim(model_data$X)[1]),
        value = model_data$X[, idx],
        type = "prediction",
        class = 'input'

      )
    }

    m_df_real_input$upper_limit <- NA
    m_df_real_input$lower_limit <- NA


    m_df_real$upper_limit <- NA
    m_df_real$lower_limit <- NA


    m_df_stan_error <- fitted_model$extracted_data$difference[,,idx]  |>
      .get_credible_interval(credibility_level=credibility_level,
                             global = is_global)


    m_df_stan_error$variable <- m_variable_name
    m_df_stan_error$type <- "error"
    m_df_stan_error$class <- "error"


    m_df_stan_error_cusum <- fitted_model$extracted_data$cumsum_only_after[,,idx] |>
      .get_credible_interval(credibility_level=credibility_level,
                             global=is_global)

    m_df_stan_error_cusum$variable <- m_variable_name
    m_df_stan_error_cusum$type <- "cumsum"
    m_df_stan_error_cusum$class <- "error"

    if(is_global) {

      df_list_aggregate[[length(df_list_aggregate) + 1]] <- m_df_stan_pred
      df_list_aggregate[[length(df_list_aggregate) + 1]] <- m_df_real
      df_list_aggregate[[length(df_list_aggregate) + 1]] <- m_df_stan_error
      df_list_aggregate[[length(df_list_aggregate) + 1]] <- m_df_stan_error_cusum
      df_list_aggregate[[length(df_list_aggregate) + 1]] <- m_df_real_input

    } else {

      df_list[[length(df_list) + 1]] <- m_df_stan_pred
      df_list[[length(df_list) + 1]] <- m_df_real
      df_list[[length(df_list) + 1]] <- m_df_stan_error
      df_list[[length(df_list) + 1]] <- m_df_stan_error_cusum
      df_list[[length(df_list) + 1]] <- m_df_real_input

    }

  }

  plot_df <- do.call(rbind, df_list)

  plot_df$type <- plot_df$type |>
    factor(levels=c("prediction", 'error', 'cumsum'))

  plot_df$is_impact  <- plot_df$time_index  >= event_initial


  plot_df_aggregate <- do.call(rbind, df_list_aggregate)
  plot_df_aggregate$type <- plot_df_aggregate$type |>
    factor(levels=c("prediction", 'error', 'cumsum'))

  plot_df_aggregate$is_impact  <- plot_df_aggregate$time_index >= event_initial

  plot_df <- rbind(plot_df, plot_df_aggregate)

  return(plot_df)


}


.plot_df  <- function(plot_df, event_initial, plot_variable, dates_df=NULL, break_dates = NULL) {



  plot_df <- plot_df |>  dplyr::filter(variable == plot_variable)

  if(is.null(dates_df)) {
    dates_df <- data.frame(time_index = plot_df$time_index |> unique() |> sort())
    dates_df$Date <- dates_df$time_index
  }

  temp_df <-  plot_df |>
    dplyr::mutate(
      class   = if_else(type == "cumsum", "cumsum", class)
    ) |>
    rename(lower = lower_limit, upper = upper_limit) |>
    dplyr::mutate(lower = if_else( (time_index <= event_initial) , NA_real_, lower)) |>
    dplyr::mutate(upper = if_else(  (time_index <= event_initial) , NA_real_, upper)) |>
    dplyr::mutate(value = if_else(  (time_index <= event_initial) &
                               (!(class %in% c("input", "prediction", "real","error"))) , NA_real_, value)) |>
    dplyr::mutate(lower = if_else(  class == "prediction" , NA_real_, lower)) |>
    dplyr::mutate(upper = if_else(  class == "prediction" , NA_real_, upper)) |>
    left_join(dates_df)
  # browser()

  temp_df <-  temp_df |> bind_rows(

    temp_df |> dplyr::filter(class == "real") |> dplyr::mutate(class = "Y ")
  )



  plot_df <- temp_df

  max_date  <- plot_df$time_index  |> max()


  if(is.null(break_dates)) {
    # break_dates = waiver()

    break_dates  <- .get_break_dates(max_date=max_date,
                                     event_initial=event_initial)

    # break_dates_index <- break_dates

    break_dates <- dates_df |> dplyr::filter(time_index %in% break_dates) |> pull(Date)

  }

  data_event_initial <- dates_df |> dplyr::filter(time_index == event_initial) |> pull(Date) |> first()

  # browser()

  plot_df  <- plot_df  |>
    rename(param = class) |>
    dplyr::mutate(
      param = param   |> recode(real = "Y",
                                error = "Impact",
                                # prediction = "Contra factual",
                                prediction = "prediction",
                                cumsum = "cumulative impact",
                                ArCo = "ArCo",
                                input = "X"
      )
    )   |>
    dplyr::mutate( face_break =
              case_when(
                param %in% c("Y","prediction") ~ "Real series and prediction",
                param %in% c("Impact") ~ "Impact",
                param %in% c("cumulative impact") ~ "cumulative impact",
                param %in% c("ArCo") ~ "ArCo",
                #param %in% c("X", "Y") ~ "Input"
                param %in% c("X", "Y ") ~ "Input"

              ) |> factor(levels = c("Real series and prediction", "Input", "Impact", "cumulative impact", "ArCo"))


    ) |>
    dplyr::mutate(

      face_break2 = case_when(
        face_break %in% c("Real series and prediction", "Input")  ~ "Modelo",
        face_break %in% c("Impact", "cumulative impact", "ArCo") ~ "Efecto"
      ) |> factor(levels=c("Modelo", "Efecto"))

    )

  m_scale_break <- scale_x_continuous(breaks = break_dates )

  if(lubridate::is.Date(break_dates)) {

    m_scale_break <- scale_x_date(labels = scales::date_format("%m/%d"), breaks = break_dates )

  }

  plot_result1  <- plot_df  |>
    dplyr::filter(face_break2 == "Modelo")  |>
    ggplot(aes(x=Date, y=value, col=param)) +
    geom_ribbon( aes(ymin = lower, ymax = upper), fill = "grey90" ) +
    geom_line() +
    geom_vline(xintercept = data_event_initial, linetype = "dashed") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    facet_wrap(~face_break, ncol=2, scales='fixed') +
    #scale_colour_manual(values=c("#006666", "#ff7f00", "#ff7f00", "#e41a1c")) +
    #scale_color_discrete(breaks = c("X", "Y", "Predicción"), values=c("#006666", "#ff7f00", "#ff7f00",  "#e41a1c") )+
    # "#3b80b9"
    # scale_colour_manual(values=c("#006666", "#ff7f00",  "#3b80b9",  "#ff7f00") )+
    scale_colour_manual(breaks = c("X", "Y", "prediction"), values=c("#006666", "#ff7f00",  "#3b80b9",  "#ff7f00"), na.value="#ff7f00" )+
    #scale_colour_manual(values=c("#006666", "#ff7f00", "#ff7f00",  "#e41a1c")) +
    # scale_x_continuous(breaks = break_dates ) +
    m_scale_break +
    theme_bw()  +
    theme(
      legend.title = element_blank(),
      axis.text=element_text(size=12),
      axis.title=element_text(size=14),
      legend.text = element_text(size=14),
      legend.position="bottom",
      strip.text = element_text(size=11)#,
      # strip.text.x = element_blank()
    ) + ylab("") + xlab("")


  plot_result2  <- plot_df  |>
    dplyr::filter(face_break2 == "Efecto",
                  Date >= data_event_initial
    )  |>
    ggplot(aes(x=Date, y=value, col=param)) +
    geom_ribbon( aes(ymin = lower, ymax = upper), fill = "grey90" ) +
    geom_line() +
    geom_vline(xintercept = data_event_initial, linetype = "dashed") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    facet_wrap(~face_break, ncol=3, scales='free_y') +
    scale_colour_manual(values=c("#e41a1c", "#4daf4a", "#9d57a7", "#3b80b9")) +
    # scale_x_date(labels = scales::date_format("%m/%d"), breaks = break_dates ) +
    m_scale_break +
    theme_bw()  +
    theme(
      legend.title = element_blank(),
      axis.text=element_text(size=12),
      axis.title=element_text(size=14),
      legend.text = element_text(size=14),
      legend.position="bottom",
      strip.text = element_text(size=11)#,
      # strip.text.x = element_blank()
    ) + ylab("")  + xlab("Fecha")

  # plot_result  <- (plot_result1/plot_result2) + plot_layout(guides = "collect") & theme(legend.position='bottom')
  plot_result  <- patchwork:::`/.ggplot`(plot_result1,plot_result2) + patchwork::plot_layout(guides = "collect") & theme(legend.position='bottom')


  return(plot_result)

}



#' @name plot.impact_vector_model
#' @title plot.impact_vector_model
#' @description
#' plot a fitted model.
#'
#' @details
#' plots \code{ImpactModelVector} object.
#'
#' @param  x: fitted model.
#' @param  plot_variable: variable to be plotted use \code{"global"} for the aggregated result
#' @param  break_dates: vector of breaks for the x axis.
#' @return A \code{ggplot2} object
#' @examples
#' \donttest{
#' data("X_vector", package="dynamicimpact")
#' data("Y_vector", package="dynamicimpact")
#' result <- ImpactModelVector(X_data=X_vector, Y_data=Y_vector,
#'                             event_initial=97, credibility_level=0.89)
#' plot(result)
#' }
#' @export
plot.impact_vector_model <- function(x,
                                     plot_variable="global",
                                     break_dates=NULL , ...) {


  current_plot_df  <- .build_plot_df(
  fitted_model = x,
  model_data = x$stan_data,
  event_initial = x$stan_data$T0,
  group_names = x$variables_names,
  credibility_level = x$credibility_level,
  global_name=NULL
  )

  .plot_df(
    plot_df=current_plot_df,
    event_initial=x$stan_data$T0,
    plot_variable=plot_variable,
    dates_df=x$dates,
    break_dates = break_dates
  )


}

#' @name plot.impact_vector_model_cholesky
#' @title plot.impact_vector_model_cholesky
#' @description
#' plot a fitted model.
#'
#' @details
#' plots \code{ImpactModelVector} object.
#'
#' @param  x: fitted model.
#' @param  plot_variable: variable to be plotted use \code{"global"} for the aggregated result
#' @param  break_dates: vector of breaks for the x axis.
#' @return A \code{ggplot2} object
#' @examples
#' \donttest{
#' data("X_vector", package="dynamicimpact")
#' data("Y_vector", package="dynamicimpact")
#' result <- ImpactModelVector(X_data=X_vector,
#'                             Y_data=Y_vector, event_initial=97,
#'                             credibility_level=0.89, use_cholesky=TRUE)
#' plot(result)
#' }
#' @export
plot.impact_vector_model_cholesky <- function(x,
                                     plot_variable="global",
                                     break_dates=NULL , ...) {


  plot.impact_vector_model(x=x, plot_variable=plot_variable, break_dates=break_dates)


}
