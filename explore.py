import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kruskal

##--------------------------Statistical Tests------------------------##

def get_kruskal_county(train):
    '''This function takes in train data and performs a Kruskal-Wallis correlation
    test, comparing properties in LA County with those in Orange and Ventura'''
    #Separating LA County properties and non-LA County properties
    LA = train[train.county == 'Los Angeles']
    not_LA = train[train.county != 'Los Angeles']
    result = kruskal(LA.tax_value, not_LA.tax_value)
    return result


def location_viz(train):
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