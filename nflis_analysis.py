import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from typing import DefaultDict, Set, List
import seaborn as sns
import statsmodels.tsa.stattools as smt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from matplotlib.ticker import MaxNLocator

#################################################
#Title: CDC_analysis.py
#Author: Delaney Smith
#Inputs: NFLIS data (handmade spreadsheet from published NFLIS pdf files), NST data (census), Reddit data (pickled formatted files only)
#Output: Pickled normalized and re-aggregated NFLIS and reddit timeseries dataframes and corresponding plot
#################################################

def extract_year(date_string):
    """
    Extracts datetime year object from date string

    Args:
    - date_string (str): string with date
 
    Returns:
    - date_string (dt_object): datetime object for year
    """
    return date_string[:4]

def replace_agg_period(df):
    """
    Replace monthly agg_period with semi

    Args:
    - df (pd.DataFrame): dataframe of reddit comments with monthly agg periods

    Returns:
    - df (pd.DataFrame): dataframe of reddit comments with semi agg periods
    """ 

    df['agg_period'] = pd.to_datetime(df['agg_period'], format='%Y-%m')
    df['year'] = df['agg_period'].dt.year
    df['month'] = df['agg_period'].dt.month
    df['suffix'] = df['month'].apply(lambda x: 1 if x <= 6 else 2)
    df['agg_period'] = df['year'].astype(str) + '_' + df['suffix'].astype(str)
    df.drop(columns=['year', 'month', 'suffix'], inplace=True)
    #df.groupby(df['agg_period']).sum()
    #df.set_index('agg_period', inplace=True)
    
    return df

def norm_nflis(nflis: pd.DataFrame, nst: pd.DataFrame, years: List[str], drug: str):
    """
    Standardize NFLIS data by population size

    Args:
    - nflis (pd.DataFrame): dataframe of NFLIS report data
    - nst (pd.DataFrame): dataframe of NST population estimate data
    - years (list): list of years to analyze
    - drug (str): name of the drug of interest
 
    Returns:
    - norm (pd.DataFrame): Dataframe of 
    """ 
    regions = ["Year", "United States", "West Region", "Midwest Region","Northeast Region", "South Region" ]

    #format nst data
    nst = nst.T
    nst.reset_index(inplace=True)
    nst.columns = nst.iloc[0]
    nst = nst[1:]
    nst.reset_index(drop=True, inplace=True)
    nst = nst[regions]
    nst.rename(columns={'United States': 'National', 'West Region': 'West', 'Midwest Region': 'Midwest', "Northeast Region" : "Northeast", "South Region" : "South"}, inplace=True)
    nst = pd.concat([nst, nst], ignore_index=True)
    nst = nst[nst['Year'].isin(years)]
    nst = nst.sort_values(by='Year')
    nst.set_index('Year', inplace=True)
    
    #format nflis data
    nflis['drug'] = nflis['drug'].str.lower()
    nflis_drug = nflis[nflis['drug'] == drug]
    nflis_drug['Year'] = nflis_drug['agg_period'].str[:4]
    nflis_drug = nflis_drug.sort_values(by='agg_period')
    nflis_drug = nflis_drug[nflis_drug['Year'].isin(years)]
    
    #index magic
    temp = nflis_drug.drop(columns=['drug'], inplace=True)
    temp = nflis_drug.drop(columns=['agg_period'])
    temp.set_index('Year', inplace=True)
    nflis_drug.set_index('Year', inplace=True)
    
    #normalize data
    result = temp/nst * 100000
    #result = temp #non-norm data
    result = result.drop_duplicates()
    result = result.dropna()
    result['agg_period'] = nflis_drug['agg_period']
    result.reset_index(drop=True, inplace=True)
    result.set_index('agg_period', inplace=True)
    
    return result

def norm_reddit(data: pd.DataFrame, ref: pd.DataFrame):

    """
    Standardize reddit data by NFLIS protocol

    Args:
    - data (pd.DataFrame): dataframe of reddit comments
    - ref (pd.DataFrame): dataframe of all cohort reddit comments
    - states (list): states to keep

    Returns:
    - result_values (pd.DataFrame): Dataframe of normalized values
    """   
    result_values = data.div(ref) * 100000

    return result_values
    #return data #use for non-norm pipeline

def plot_cc(ccf, category):
    
    """
    Visualizes NFLIS and Reddit cross-correlation values over time

    Args:
    - ccf (numpy vector): results of cross correlation analysis
    - category (str): type of drug category being analyzed

    Returns:
    - none
    """ 

    plt.figure(figsize=(4, 4))
    sns.set_style("whitegrid")

    plt.gca().set_facecolor('#f0f0f0')

    lags = np.arange(-6, 7)
    data = pd.DataFrame({'lag': lags, 'cross_corr': ccf})

    sns.lineplot(data=data, x='lag', y='cross_corr', marker='o', linestyle='None',markerfacecolor='black', markeredgecolor='black')

    # Customize the plot
    plt.xlabel('Lag Offset (months)', color = 'black', fontsize = 14)
    plt.ylabel('Cross Correlation', color = 'black', fontsize = 14)
    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')

    # Customize grid
    plt.grid(True, color='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['right'].set_color('white')

    # Set the background color for the entire plot
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"Figure2_ccf_NFLIS_National_{category}.png")

def figure_3(nflis: pd.DataFrame, reddit: pd.DataFrame, drug: str):

    """
    Visualizes nflis analysis data

    Args:
    - nflis (pd.DataFrame): dataframe of nflis normalized time series
    - ref (pd.DataFrame): dataframe of reddit normalized time series
    - drug (str): which drug we are currently processing

    Returns:
    - fig (ggplot figure): figure object
    """  

    #re-format data  
    columns = ['National', 'West', 'Midwest', 'North East', 'South']
    reddit = reddit[columns]
    merged = pd.concat([nflis, reddit], axis=1)
    level0 = ['National', 'West', 'Midwest', 'Northeast', 'South'] + ['National', 'West', 'Midwest', 'Northeast', 'South']
    level1 = ['NFLIS'] * 5 + ['Reddit'] * 5
    multi_index = pd.MultiIndex.from_arrays([level0, level1])
    merged.columns = multi_index
    
    merged = merged.reset_index()
    stacked = merged.set_index('agg_period').stack().reset_index()
    stacked.columns = ['agg_period','Datasource', 'National', 'West' , 'Midwest' , 'Northeast', 'South']
    stacked = stacked.sort_values(by='agg_period')
    #print(stacked)

    regions = ['National', 'West', 'Midwest', 'South', 'Northeast' ]
    num_rows = 1 
    num_cols = 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 2))
    axes = axes.flatten()

    colorblind_palette = sns.color_palette("colorblind")

    #Iterate over unique regions and create separate plots
    for i, region in enumerate(regions):
        ax = axes[i]
        n = stacked[stacked['Datasource'] == 'NFLIS']
        r = stacked[stacked['Datasource'] == 'Reddit']

        sns.lineplot(data=stacked, x='agg_period', y=n[region], color = colorblind_palette[0], ax=ax,  marker='o', linestyle='-') #blue

        ax2 = ax.twinx()
        sns.lineplot(data=stacked, x='agg_period', y=r[region], color = colorblind_palette[1], ax=ax2, marker='o', linestyle='-') #orange

        ax.grid()
        ax.set_title(region, fontsize=14, color='black')
        ax.set_ylabel('')
        ax2.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=10, color='black')
        ax2.tick_params(axis='both', which='major', labelsize=10, color='black')

        num_ticks = 3
        locator = MaxNLocator(nbins=num_ticks, min_n_ticks=num_ticks, prune='both')
        ax.yaxis.set_major_locator(locator)
        ax2.yaxis.set_major_locator(locator)

        ticks = [tick for tick in stacked['agg_period'] if '_1' in tick]
        even_ticks = [tick for tick in ticks if (int(tick.split('_')[0])%2) ==0]
        x_tick_labels = [tick.split('_')[0] for tick in even_ticks]
        ax.set_xticks(ticks=even_ticks, labels=x_tick_labels)

        for spine in ax.spines.values():
                spine.set_edgecolor('white')

    plt.subplots_adjust(right=0.99)

    #fig.text(0.5, 0.0, 'Date', ha='center',fontsize=14, color='black')
    fig.text(0.001, 0.5, 'NFLIS Rate', va='center', rotation='vertical',fontsize=12, color='black')
    fig.text(0.99, 0.5, 'Reddit Rate', va='center', rotation='vertical',fontsize=12, color='black')
    #plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.tight_layout(pad=1.5, rect=[0, 0, 0.99, 0.99])

    plt.savefig(f"Figure3_{drug}.png")
    #plt.show()

    return (fig)

def ccf(x, y, lag_max = 6):

    """
    Calculate cross-correlation values between Reddit and NFLIS rates using the ccf (R package) method

    Args:
    - x (pd.DataFrame): dataframe of reddit national normalized time series
    - y (pd.DataFrame): dataframe of NFLIS national normalized time series
    - lag_max (int): maximum number of months to check cross correlation values

    Returns:
    - ccf_output (numpy object): cross correlation values
    """ 

    backwards = smt.ccf(y, x, unbiased=False, nlags = lag_max + 1)[::-1]
    forwards = smt.ccf(x, y, unbiased=False, nlags = lag_max + 1)
    ccf_output = np.r_[backwards[:-1], forwards]

    return ccf_output

def main(years: List[str]):

    #map states to regions
    state_to_region = {
    'Alabama': 'South',
    'Alaska': 'West',
    'Arizona': 'West',
    'Arkansas': 'South',
    'California': 'West',
    'Colorado': 'West',
    'Connecticut': 'North East',
    'Delaware': 'South',
    'Florida': 'South',
    'Georgia': 'South',
    'Hawaii': 'West',
    'Idaho': 'West',
    'Illinois': 'Midwest',
    'Indiana': 'Midwest',
    'Iowa': 'Midwest',
    'Kansas': 'Midwest',
    'Kentucky': 'South',
    'Louisiana': 'South',
    'Maine': 'North East',
    'Maryland': 'South',
    'District of Columbia': 'South',
    'Massachusetts': 'North East',
    'Michigan': 'Midwest',
    'Minnesota': 'Midwest',
    'Mississippi': 'South',
    'Missouri': 'Midwest',
    'Montana': 'West',
    'Nebraska': 'Midwest',
    'Nevada': 'West',
    'New Hampshire': 'North East',
    'New Jersey': 'North East',
    'New Mexico': 'West',
    'New York': 'North East',
    'North Carolina': 'South',
    'North Dakota': 'Midwest',
    'Ohio': 'Midwest',
    'Oklahoma': 'South',
    'Oregon': 'West',
    'Pennsylvania': 'North East',
    'Rhode Island': 'North East',
    'South Carolina': 'South',
    'South Dakota': 'Midwest',
    'Tennessee': 'South',
    'Texas': 'South',
    'Utah': 'West',
    'Vermont': 'North East',
    'Virginia': 'South',
    'Washington': 'West',
    'West Virginia': 'South',
    'Wisconsin': 'Midwest',
    'Wyoming': 'West'
    }

    #read in needed files
    nflis_data = pd.read_csv("/Users/Delaney_Smith_1/Desktop/redditCorpus/Analysis/mappings/overdose/NFLIS_ALL.csv")
    nst = pd.read_csv("/Users/Delaney_Smith_1/Desktop/redditCorpus/Analysis/mappings/overdose/nst_2012-2023.csv") 
    
    #manually add drugs of interest to this list
    #drugs = ["heroin", "oxycodone", "hydrocodone", "fentanyl", "buprenorphine"]
    drugs = ["fentanyl"] #manuall edit here

    # Load data
    term_data = f'/Users/Delaney_Smith_1/Desktop/redditCorpus/Data/everything/complete_data/drug_data_2014-2022.p'
    with open(term_data, "rb+") as f:
            drug_data = pickle.load(f)

    total = f'/Users/Delaney_Smith_1/Desktop/redditCorpus/Data/everything/complete_data/total_comments_2005-2022.p'
    with open(total, "rb+") as f:
            total_comments = pickle.load(f)

    #convert to semi-annual
    drug_data = replace_agg_period(drug_data)
    total_comments.reset_index(inplace=True)
    total_comments = replace_agg_period(total_comments)
    total_comments.set_index('agg_period', inplace=True)

    #specify years for analysis
    years = ['2015','2016', '2017', '2018', '2019', '2020', '2021', '2022']
    
    #For this analysis, we keep all states's Reddit signal because we are unable to remove low signal state values from the NFLIS benchmark
    
    states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
    'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
    'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
    'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Rhode Island',
    'South Carolina', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia'
    ]

    #check for zero values below
    #zero_columns = total_comments.columns[(total_comments == 0).any()]
    #print(total_comments[zero_columns])

    for drug in drugs:
        #normalize nflis data
        nflis = norm_nflis(nflis_data, nst, years, drug)
        nflis.reset_index(inplace=True)
        nflis = nflis[nflis['agg_period'].apply(extract_year).isin(years)]
        nflis.set_index('agg_period', inplace=True)
        #nflis.to_pickle(f'nflis_data_formatted_2015-2022_{drug}_no_norm.p')
        #print(nflis)
        
        #format reddit drug data
        drug_data = drug_data[drug_data['drug'] == drug]
        drug_data = drug_data.groupby(['agg_period', 'state']).sum().unstack().fillna(0)
        drug_data.columns.names = [None, None]
        drug_data = drug_data['count']
        #print(drug_data)

        #convert states to regions
        drug_data = drug_data[states]
        drug_data.rename(columns=state_to_region, inplace=True)
        drug_data = drug_data.groupby(axis=1, level=0).sum()
        drug_data['National'] = drug_data.sum(axis=1)
        
        total_comments = total_comments.groupby(level=0).sum()
        total_comments = total_comments[states]
        total_comments.rename(columns=state_to_region, inplace=True)
        total_comments = total_comments.groupby(axis=1, level=0).sum()
        total_comments['National'] = total_comments.sum(axis=1)
        
        #normalize reddit data
        norm_drug_data = norm_reddit(drug_data, total_comments)
        norm_drug_data.reset_index(inplace=True)
        norm_drug_data = norm_drug_data[norm_drug_data['agg_period'].apply(extract_year).isin(years)]
        norm_drug_data.set_index('agg_period', inplace=True)

        #remove zeros
        zero_columns = norm_drug_data.columns[(norm_drug_data == 0).any()]
        zeros = norm_drug_data[zero_columns]
        #print(zeros[zeros['West Virginia'] == 0])

        norm_drug_data = norm_drug_data.groupby(level=0).sum()
        norm_drug_data.to_pickle(f'reddit(nflis)_data_formatted_2015-2022_{drug}_no_norm.p') #save file if needed
        #print(norm_drug_data)

        fig = figure_3(nflis, norm_drug_data, drug)

        #calculate cross correlations, not in final manuscript
        x = norm_drug_data['National']
        y = nflis['National']
        
        #differences series, we check both differenced and raw cross correlations (not in final manuscript)
        y_dif = y.diff(periods=1).dropna() 
        x_dif = x.diff(periods=1).dropna() 

        #unit root testing, not in final manuscript
        result = adfuller(x_dif)
        print('p-value:', result[1])

        result = ccf(x,y,6)
        #plot_cc(result, drug)
        #print(result)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to analyze the redddit and CDC data.")
    parser.add_argument("--years", type=str, nargs="+", required=True, help="List of years to process, e.g. --years 2012 2013 2014")

    args = parser.parse_args()
    
    main(args.years)