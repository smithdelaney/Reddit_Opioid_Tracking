import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import DefaultDict, Set, List
from datetime import datetime
import seaborn as sns
import pmdarima as pm
from sklearn.metrics import root_mean_squared_error
from pmdarima import auto_arima
from pmdarima.model_selection import RollingForecastCV
from scipy import stats
from matplotlib.ticker import MaxNLocator
from scipy.stats import wilcoxon

#################################################
#Title: ARIMA_modelng.py
#Author: Delaney Smith
#Inputs: output pickled dataframes for reddit and cdc data from cdc_analysis.py
#Output: ARIMA models for CDC alone and CDC + Reddit and corresponding plots
#################################################

def cdc_reddit(cdc: pd.DataFrame, reddit: pd.DataFrame):

    """
    Performs ARIMA analysis on cdc data with Reddit data as exogenous features

    Args:
    - cdc (pd.DataFrame): dataframe of cdc normalized time series
    - reddit (pd.DataFrame): dataframe of reddit normalized time series

    Returns:
    - predictions (list): numeric list of predicted values
    - results (list): numeric list of absolute error values
    """

    #use code below for lagged ARIMA
    y = cdc['National'].shift(6) #lag reddit data X months
    y = y.iloc[6:] #remove introduced NA values
    feat = reddit['National'] 
    feat = feat.iloc[6:] #remove values 'paired' to NAs

    #not lagged version, currently not used in manuscript
    #y = cdc['National']
    #feat = reddit['National']

    #set up rolling-forecast CV
    cv = RollingForecastCV(initial=24, step=1, h=1)
    cv_generator = cv.split(y)
    
    #list of tuples for error analysis
    results = []
    predictions = []
    best_rmse = float('inf')
    
    #run ARIMA model selection
    for train_idx, test_idx in cv_generator:
        
        #create split data sets
        train_data = y.iloc[train_idx]
        test_data = y.iloc[test_idx]

        #create split feature data sets
        train_feat = feat.iloc[train_idx]
        test_feat = feat.iloc[test_idx]

        #create auto arima model and fit
        model = auto_arima(train_data, X = train_feat.to_frame(name='Reddit'), seasonal=False, error_action='ignore', suppress_warnings=True, trace=False)

        #Make predictions
        preds = model.predict(n_periods=test_data.shape[0], X = test_feat.to_frame(name='Reddit'))
        x = preds.iloc[-1]
        predictions.append(x)

        #evaluate predictions using RMSE (not currently used in manuscript)
        #rmse = root_mean_squared_error(test_data, preds)
        #results.append(rmse)

        #evaluate using absolute error
        z = test_data.iloc[-1]
        ae = [abs(z - x)] #true-pred
        results.append(ae)

    #print (results)
    avg = np.mean(results)
    #print(avg)
    print("Combined CDC + Reddit:")

    return predictions, results

def reddit_arima(reddit: pd.DataFrame):

    """
    Performs ARIMA analysis on reddit data alone -- not currently used in manuscript

    Args:
    - reddit (pd.DataFrame): dataframe of reddit normalized time series

    Returns:
    - results (list): numeric list of RMSE values for ARIMA performance
    """

    #select National level reddit time series data
    y = reddit['National'] 
    
    #set up rolling-forecast CV
    cv = RollingForecastCV(initial=24, step=1, h=3)
    cv_generator = cv.split(y)
    
    #list of tuples for rmse
    results = []
    best_rmse = float('inf')
    
    #run arima w/cdc data
    for train_idx, test_idx in cv_generator:
        
        #create split data sets
        train_data = y.iloc[train_idx]
        test_data = y.iloc[test_idx]

        #create auto arima model and fit
        model = auto_arima(train_data, seasonal=False, error_action='ignore', suppress_warnings=True, trace=False)

        #Make predictions
        preds = model.predict(n_periods=test_data.shape[0])

        #evaluate predictions using RMSE
        rmse = root_mean_squared_error(test_data, preds)

        #save rmse values
        results.append(rmse)

        #save best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    #print results
    #avg = np.mean(results)
    print ("Reddit only:")
    print (len(results))

    return results

def cdc_arima(cdc: pd.DataFrame):

    """
    Performs ARIMA analysis on cdc data alone

    Args:
    - cdc (pd.DataFrame): dataframe of cdc normalized time series

    Returns:
    - predictions (list): numeric list of predicted values
    - results (list): numeric list of absolute error values
    """

    #lag data
    y = cdc['National'].shift(6) #lag CDC data
    y = y.iloc[6:] #remove NA values

    #no lag -- not currently used
    #y = cdc['National']
    #y = y.iloc[:-6]

    #set up rolling-forecast CV
    cv = RollingForecastCV(initial=24, step=1, h=1)
    cv_generator = cv.split(y)
    
    #list of tuples for error analysis
    results = []
    predictions = []
    best_rmse = float('inf')
    
    #run empty arima w/cdc data
    for train_idx, test_idx in cv_generator:
        
        #create split data sets
        train_data = y.iloc[train_idx]
        test_data = y.iloc[test_idx]

        #create auto arima model and fit
        model = auto_arima(train_data, seasonal=False, error_action='ignore', suppress_warnings=True, trace=False)

        #Make predictions
        preds = model.predict(n_periods=test_data.shape[0])
        x = preds.iloc[-1]
        predictions.append(x)

        #evaluate predictions using absolute error
        z = test_data.iloc[-1]
        ae = [abs(z - x)]
        results.append(ae)

    #print results
    avg = np.mean(results)
    #print(avg)
    print ("CDC only:")
    return predictions, results

def figure_4(reddit: pd.DataFrame, cdc: pd.DataFrame, drug: str, region: str):

    """
    Creates error over time graphs for different ARIMA models

    Args:
    - cdc (pd.DataFrame): dataframe of ARIMA error values based on cdc-only model
    - reddit (pd.DataFrame): dataframe of reddit normalized time series
    - drug (str): which drug or drug category we are currently processing
    - region (str): which geographic region we are currently processing

    Returns:
    - none
    """

    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 2))
    colorblind_palette = sns.color_palette("colorblind") 

    start_date = '2017-07-01'
    end_date = '2022-12-31'
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    date_range = date_range.append(pd.DatetimeIndex(['2023-01-01']))
    #print(len(date_range))
    reddit = np.append(reddit, np.nan)
    cdc = np.append(cdc, np.nan) #for Reddit+ CDC combintation only
    #print(len(reddit))
    #print(len(cdc))
    sns.lineplot(x=date_range, y=cdc, color = colorblind_palette[0], marker='o', linestyle='--') #for combination plotting
    sns.lineplot(x=date_range, y=reddit, color=colorblind_palette[1], marker='o', linestyle='--') #for combination plotting
    #sns.lineplot(x=date_range, y=reddit, color= colorblind_palette[4], marker='o', linestyle='--') #for reddit only plots

    plt.ylabel('absolute error', fontsize=12, color='black')
    plt.xlabel('')
    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')

    buffer_days = 10  # Number of days for the buffer
    start_limit = date_range[0] - pd.Timedelta(days=buffer_days)
    end_limit = date_range[-1] + pd.Timedelta(days=buffer_days) 

    plt.xlim([start_limit, end_limit])

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)

    num_ticks = 4
    locator = MaxNLocator(nbins=num_ticks, min_n_ticks=num_ticks, prune='both')
    ax.yaxis.set_major_locator(locator)

    #plt.title(region)
    plt.tight_layout()
    plt.savefig(f"Figure4_{region}_{drug}_abs.png")
    #plt.show()

def figure_5(reddit: pd.DataFrame, cdc: pd.DataFrame, truth: pd.DataFrame, drug: str, region: str):

    """
    Creates predicted vs. true overdose death rates over time for the two ARIMA models

    Args:
    - reddit (pd.DataFrame): ARIMA predictions based on combined Reddit/CDC model
    - cdc (pd.DataFrame): ARIMA predictions based on CDC model alone
    - truth (pd.DataFrame): dataframe of true CDC normalized overdose death rates
    - drug (str): which drug or drug category we are currently processing
    - region (str): which geographic region we are currently processing

    Returns:
    - none
    """

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 4)) 
    colorblind_palette = sns.color_palette("colorblind") 

    truth = truth['National'].iloc[30:]

    start_date = '2017-07-01'
    end_date = '2022-12-31'
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    date_range = date_range.append(pd.DatetimeIndex(['2023-01-01']))
    reddit = np.append(reddit, np.nan)
    cdc = np.append(cdc, np.nan)
    truth = np.append(truth, np.nan)
    sns.lineplot(x=date_range, y=cdc, color = colorblind_palette[0], marker='o', linestyle='--')
    sns.lineplot(x=date_range, y=reddit, color= colorblind_palette[1], marker='o', linestyle='--')
    #sns.lineplot(x=date_range, y=truth, color='black', marker='o', linestyle='-')
    plt.bar(date_range, truth, color='gray', alpha=0.5, width=20)
    
    plt.ylabel('Overdose Death Rates', fontsize=12, color='black')
    plt.xlabel('')
    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')

    buffer_days = 10  # Number of days for the buffer
    start_limit = date_range[0] - pd.Timedelta(days=buffer_days)
    end_limit = date_range[-1] + pd.Timedelta(days=buffer_days) 

    plt.xlim([start_limit, end_limit])
    bar_width = 0.6

    ax = plt.gca()
    #ax.bar(date_range, truth, color='grey', alpha=0.7, width=bar_width)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)

    num_ticks = 4
    locator = MaxNLocator(nbins=num_ticks, min_n_ticks=num_ticks, prune='both')
    ax.yaxis.set_major_locator(locator)

    #plt.title(region)
    plt.tight_layout()
    plt.savefig(f"Figure5_{region}_{drug}.png")
    #plt.show() 

def figure_6(reddit: pd.DataFrame, cdc: pd.DataFrame, drug: str, region: str):

    """
    Creates histogram distribution of errors for different ARIMA models

    Args:
    - cdc (pd.DataFrame): dataframe of cdc-only ARIMA error values
    - reddit (pd.DataFrame): dataframe of reddit/CDC ARIMA error values
    - drug (str): which drug or drug category we are currently processing
    - region (str): which geographic region we are currently processing

    Returns:
    - none
    """
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 2))
    colorblind_palette = sns.color_palette("colorblind") 

    reddit = [item for sublist in reddit for item in sublist]
    cdc = [item for sublist in cdc for item in sublist]
    bins = np.linspace(0, 0.2, 11) #change for abs. vs regular
    sns.histplot(cdc, bins=bins, color=colorblind_palette[0], alpha=0.5)
    sns.histplot(reddit, bins=bins, color=colorblind_palette[1], alpha=0.5)

    plt.xlabel('absolute error', fontsize=12, color='black')
    plt.ylabel('count', fontsize=12, color='black')
    plt.xlabel('')
    plt.xticks(ticks=bins, fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)

    #plt.title(region)
    plt.tight_layout()
    plt.savefig(f"Figure6_{region}_{drug}_abs.png")
    #plt.show()

def main():

    #options for drug categories are: 'Heroin (T40.1)', 'Natural & semi-synthetic opioids (T40.2)', 'Synthetic opioids, excl. methadone (T40.4)''

    #load pickled cdc dataframe from cdc_analysis.py
    term_file = "/Users/Delaney_Smith_1/Desktop/redditCorpus/Analysis/cdc_data_formatted_2014-2022_Natural & semi-synthetic opioids (T40.2).p"
    with open(term_file, "rb+") as f:
            cdc = pd.read_pickle(f)
    #print(cdc)

    #load pickled reddit dataframe from processing from cdc_analysis.py
    file = "/Users/Delaney_Smith_1/Desktop/redditCorpus/Analysis/reddit_data_formatted_2014-2022_Natural & semi-synthetic opioids (T40.2).p"
    with open(file, "rb+") as f:
            reddit = pd.read_pickle(f)
    reddit = reddit.rename(columns={'North East': 'Northeast'})
    #print(reddit)

    #call cdc ARIMA function
    empty, ae_e = cdc_arima(cdc)

    #combined ARIMA w/ Reddit as feature
    combined, ae_c = cdc_reddit(cdc, reddit)

    #evluate comparative avg. error
    #print(ae_e)
    #print(ae_c)
    t,p = wilcoxon(ae_e, ae_c) #for absolute error
    #t,p = stats.ttest_rel(ae_e, ae_c) #for error
    print("p-value:")
    print(p)

    #Below is additional analysis to check pre-2020 abo
    ae_e_2020 = ae_e[:-36]
    ae_c_2020 = ae_c[:-36]
    t,p = wilcoxon(ae_e_2020, ae_c_2020) #for absolute error pre-2020
    print("p-value pre-2020:")
    print(p)

    print("average absolute error CDC alone pre-2020:")
    print(np.mean(ae_e_2020))

    print("average absolute error CDC w/Reddit pre-2020:")
    print(np.mean(ae_c_2020))

    #create figures
    figure_4(ae_e, ae_c, 'N&S', 'National')
    figure_5(combined, empty, cdc, 'N&S', 'National')
    figure_6(ae_c, ae_e, 'N&S', 'National')

if __name__ == "__main__":
    
    main()