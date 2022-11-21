# Zillow Regression Project  

# Project Description:
This project is designed to identify important features and build a regression model to predict tax-assessed property value for single home properties. This report uses 'cost' or 'value' to refer to a home's tax assessed value.
.

# Project Goal:
* Discover drivers Tax Value for Single Family Homes in the Zillow Data Set.
* Use drivers to develop machine learning models to predict tax value.

# Initial Thoughts:
My initial hypothesis is that number of bedrooms and bathrooms, along with square feet, are the most important features to predict home value.

# The Plan
* Aquire data from codeup database

* Prepare data

* Explore data to discover potential drivers of home price

  * Answer the following initial questions:
    * Does property location impact home value?
    * Does an increase in bedrooms/bathrooms also increase property tax value? 
    * Does home age impact tax value?
    * Are bedrooms or bathrooms a better predictor of home value?
  
* Develop a Model to predict tax value

    * Use drivers identified through exploration to build predictive models of different types
    * Evaluate models on train and validate data
    * Select the best model based on lowest RMSE in combination with R2 scores
    * Evaluate the best model on test data
    
* Draw conclusions



# Data Dictionary

| Feature | Definition |
| --- | --- |
| Tax_Value | Value of a property computed by county utilizing tax data (TARGET VARIABLE)
| Bedrooms | Number of bedrooms listed for a property|
| Bathrooms | Number of bathrooms listed for a property |
| Square_Feet | Total square footage listed for a property |
| FIPS | Codes associated with US Counties |
| Latitude | Latitude associated to a property location |
| Longitude | Longitude associated to a property location |
| Lot_Size | Total area of a property lot |
| Year_Built | Year a property was constructed |
| Transaction_Date | Date a property was bought/sold |
| Parcel_ID | Unique identifying number associated with a property by Zillow |
| Home_Age | Feature-engineered column assigning an integer as "home age" by subtracting year built from 2017
| County| Feature-engineered column indicating the county a property is located in, utilizing FIPS code

# Steps to Reproduce

1. Clone this repository
2. Acquire the data from the Codeup Database ('Zillow')
3. Put the data in the file containing the cloned repo.
4. Create or copy your env.py file to this repo, specifying the codeup hostname, username and password
5. Run notebook.

# Takeaways and Conclusions

* Exploration of the data revealed significant relationships between features of this data and single property tax value
* More time in exploration could yield a different combination of features, lowering RMSE and R2 for future models.
* The combination of number of bedrooms, number of bathrooms, square feet, home age and location(LA County) helped created a model that lowered RMSE by 20% from the baseline. Its selection focused on replicability on testing data, which helped create a model that performed similarly well on training, validation, and testing data.
* My final model produced an R2 score of 0.36.

# Recommendations

* Continuing feature engineering for location. Classification of properties by Zip Code is likely to provide even stronger drivers of tax value. For properties in this dataset, a separation by county.
* Obtaining data of last transaction price would likely include a large predictor of tax value

# Next Steps

* Including more features into modeling without overfitting a model
* Utilizing chosen features with models better suited to fit the skewed data within the Zillow dataset