import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kruskal, levene, ttest_ind

##--------------------------Statistical Tests------------------------##

def get_kruskal_county(train):
    '''This function takes in train data and performs a Kruskal-Wallis t-test,
    comparing properties in LA County with those in Orange and Ventura'''
    #Separating LA County properties and non-LA County properties
    LA = train[train.county == 'Los Angeles']
    not_LA = train[train.county != 'Los Angeles']
    result = kruskal(LA.tax_value, not_LA.tax_value)
    return result

def kruskal_bedrooms(train):
    '''This function takes in train data and performs a Kruskal-Wallis t-test,
    comparing properties with different number of bedrooms to their tax-value
    by evaluating their separate means'''
    ## Creating new variables to make each bedroom count distinct
    one_bed = train[train.bedrooms == 1]
    two_bed = train[train.bedrooms == 2]
    three_bed = train[train.bedrooms == 3]
    four_bed = train[train.bedrooms == 4]
    five_bed = train[train.bedrooms == 5]
    six_bed = train[train.bedrooms == 6]
    #Running the kruskal test
    result = kruskal(
        one_bed.tax_value, two_bed.tax_value, three_bed.tax_value, four_bed.tax_value, five_bed.tax_value, six_bed.tax_value)
    return result

def get_ttest_bedbath(train):
    '''This function takes in the training dataset and performs a t-test comparing properties with under-median bathrooms and over-
    median bedrooms to properties with over-median bathrooms and under median bedrooms. It returns a test statistic and
    p-value for the test'''
    # Create the samples
    bathrooms_above = train[(train.bathrooms > train.bathrooms.median())&(train.bedrooms < train.bedrooms.median())].tax_value
    bathrooms_below = train[(train.bathrooms < train.bathrooms.median())&(train.bedrooms > train.bedrooms.median())].tax_value
    
    #set alpha
    alpha = 0.05

    # Check for equal variances
    s, pval = levene(bathrooms_above, bathrooms_below)

    # Run the two-sample, one-tail T-test.
    # Use the results from checking for equal variances to set equal_var
    t, p = ttest_ind(bathrooms_above, bathrooms_below, equal_var=(pval >= alpha))

    # Evaluate results:
    return print(f'Test statistic: {t.round(2)}, P-Value: {p}')


def county_viz(train):
    '''This function takes in training data and outputs a visualization representing the difference in home value
    by county, with lines indicating mean tax value for properties'''
    # Binning Home Value
    train['brackets'] = pd.cut(train.tax_value, 10, labels=[1,2,3,4,5,6,7,8,9,10])
    
    # Set Size
    plt.figure(figsize=(12,8))
    # plot it
    sns.histplot(data=train, x='tax_value', alpha=.8, hue='county', hue_order=['Ventura', 'Orange', 'Los Angeles'])
    # add lines marking the mean value at each location
    plt.axvline(x=train[train.county == 'Los Angeles'].tax_value.mean(), color='blue', linestyle='--')
    plt.axvline(x=train[train.county == 'Orange'].tax_value.mean(), color='orange', linestyle='--')
    plt.axvline(x=train[train.county == 'Ventura'].tax_value.mean(), color='green', linestyle='--')
    
    # axis tick labeling using
    plt.xticks(ticks = [0,200000,400000,600000,800000,1000000, 1200000], labels=['0', '200,000', '400,000', '600,000', '800,000', '1,000,000', '1,200,000'])
    
    # Make a title, label the axes
    plt.title('Tax Value by County, With Dashed Lines Indicating County Mean Values')
    plt.xlabel('Tax Value')
    plt.ylabel('Count of Homes')
    plt.show()
    
    print()
    
def get_hist_bedbath(train):
    '''This function takes in train and returns two histograms overlayed showing the distribution
    of bedrooms and bathrooms in the dataset'''
    # Creating histogram
    fig, ax = plt.subplots(figsize =(10, 7), tight_layout = True)
    ax.hist(train.bedrooms, color='lightsteelblue', label='bedrooms')
    ax.hist(train.bathrooms, color='lightsalmon', label='bathrooms')
    plt.xlabel("Number of Bedrooms and Bathrooms")
    plt.ylabel("Count of Homes")
    plt.title('Distribution of Bedrooms and Bathrooms within dataset')
    plt.legend()
    #Show plot
    plt.show()