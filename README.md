# dynamicimpact: Inferring Causal Effects on vector and matrix variate time series using Bayesian dynamic models.


## Installation

```{r}
install.packages("remotes")
remotes::install_github("CarlosAndres12/dynamicimpact")
```


## Basic example

```{r}
library(dynamicimpact)
data("X_vector", package="dynamicimpact")
data("X_vector", package="dynamicimpact")

result <- ImpactModelVector(X_data=X_vector, Y_data=Y_vector, 
                            event_initial=97, credibility_level=0.89)
                            
                            
plot(result)
```
