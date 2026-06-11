Research work on factors affecting longevity

Ran multiple predictive modeling using using different regression model to evaluate factors that drive longevity

Dataset was derived from world bank
  During imputation corruption was dropped due to a high missing value
  Missing values were filled in two ways, using forward fill / backward fill and filling using mean per region
  EDA was achieved using boxplot, scatter plot, relation plot, histogram, line plot and much more
  GeoMap was incorporated to visualize countries life expectancy trnd
  Multiple model including gradient boost, random forest, xgboost were used for predictive analytics
  
Key finding and result indicates that sanitation and prevalence of undernourishment are drivers of longevity

  Correlation matrix back key finding, undernourishment had the strongest positive correlation of 0.70 and sanitaation had the strongest negative correlation of -0.69
  Average lif expectancy increased across all global region, with the strongest increase tied to lower undernourishment and higher sanitation
  Tree based model are better predictors than deep learning model
