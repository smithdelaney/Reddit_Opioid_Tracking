import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import DefaultDict, Set, List
from datetime import datetime
import seaborn as sns
import dask.dataframe as dd
from matplotlib.ticker import MaxNLocator
import statsmodels.tsa.stattools as smt
from statsmodels.tsa.stattools import adfuller

#################################################
#Title: CDC_analysis.py
#Author: Delaney Smith
#Inputs: CDC data (VSRR provisional data), NST data (census), Reddit data (either pickled or raw comment/term files)
#Output: Pickled normalized CDC and reddit timeseries dataframes and corresponding plots (2)
#################################################

def drug_to_category(category: str):
    """
    Map drugs in CDC categories. The drugs are those listed in Adam's script.

    Args:
    - category (str): cdc drug category of interest
 
    Returns:
    - drugs (str): list of drugs in the category
    """   
    drugs = []

    if category == 'Heroin (T40.1)':
        drugs = ["heroin"]
    elif category == 'Natural & semi-synthetic opioids (T40.2)':
        drugs = ["morphine", "codeine", "hydrocodone", "oxycodone", "oxymorphone", "hydromorphone", "naloxone", "buprenorphine", "naltrexone"]
    elif category == 'Synthetic opioids, excl. methadone (T40.4)':
        drugs = ["tramadol", "fentanyl", "meperidine"]
    elif category == 'Opioids (T40.0-T40.4,T40.6)':
        drugs = ["morphine", "codeine", "oxycodone", "pethidine", "diamorphine", "hydromorphone", "levorphanol", "methadone", "fentanyl", 
                 "sufentanyl", "remifentanyl", "tramadol", "tapedolol", "oxymorphone", "hydrocodone", "buprenorphine", "meptazinol", "loperamide",
                 "nalorphine","pentazocine","nalbuphine", "butorphanol", "dezocine", "naloxone", "naltrexone", "nalmefene", "diprenorphine", "heroin", "kratom"]

    return drugs

def extract_year(date_string):
    """
    Extracts datetime year object from date string

    Args:
    - date_string (str): string with date
 
    Returns:
    - date_string (dt_object): datetime object for year
    """
    return date_string[:4]

def month_to_number(month_name):
    """
    Convert month names to numbers (ex: January to 01)

    Args:
    - month_name (str): month name
 
    Returns:
    - (dt_object): datetime object for month
    """
    dt_object = datetime.strptime(month_name, "%B")
    
    return dt_object.strftime("%m")

def state_converter(state):
    """
    Convert state acronyms to full names

    Args:
    - state (str): state acronym
 
    Returns:
    - state(str): full name if found
    """
    #currently not used in this script
    state_map = {
        'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine',
        'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
        'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee',
        'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin',
        'WV': 'West Virginia', 'WY': 'Wyoming' ,'DC': 'District of Columbia'
    }
    return state_map.get(state, state) 

def norm_cdc(cdc: pd.DataFrame, nst: pd.DataFrame, years: List[str], category: str, states: List[str], good_states: List[str]):
    """
    Standardize CDC data by population size

    Args:
    - cdc (pd.DataFrame): dataframe of CDC overdose data
    - nst (pd.DataFrame): dataframe of NST population estimate data
    - years (list): list of years to analyze
    - category (str): name of the drug of interest
    - states (list): list of states to include for National calculations
    -  good_states (list): list of states to include for regional mapping (which have sufficient signal)
 
    Returns:
    - norm (pd.DataFrame): Dataframe of 
    """
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

    #format nst census dataframe
    nst = nst.T
    nst.reset_index(inplace=True)
    nst.columns = nst.iloc[0]
    nst = nst[1:]
    nst.reset_index(drop=True, inplace=True)
    nst = nst[nst['Year'].isin(years)]
    nst = nst.sort_values(by='Year')
    nst.set_index('Year', inplace=True)
    nst = nst[states]
    nat = nst.sum(axis=1)
    nst = nst[good_states]
    #print(nst)
    
    #use the code below to convert states to regions
    nst.rename(columns=state_to_region, inplace=True)
    nst = nst.groupby(axis=1, level=0).sum()
    nst['National'] = nat

    #format the cdc overdose dataframe
    cdc = cdc[["Year", "Month", "Indicator", "Data Value", "State Name", "area"]]
    cdc = cdc[cdc['State Name'].isin(states)]
    years = [int(x) for x in years]
    cdc = cdc[cdc['Year'].isin(years)]
    cdc = cdc[cdc['Indicator'] == category]
    print(len(cdc['State Name'].unique()))

    cdc['Data Value'] = cdc['Data Value'].fillna(0)
    cdc['Data Value'] = cdc['Data Value'].str.replace(',', '').astype(float)
    cdc['Year'] = cdc['Year'].astype(str)
    cdc['Month'] = cdc['Month'].apply(lambda x: month_to_number(x))
    cdc = cdc.sort_values(by=['Year', 'Month'])
    #print(cdc)
  
    temp = cdc
    temp['agg_period'] = temp['Year'].astype(str) + '-' + temp['Month'].str[:2]
    other = temp
    temp = temp.groupby(['agg_period','State Name'])
    temp = temp['Data Value'].sum().unstack()
    agg = temp.index
    #print (temp)
    
    other = other.groupby(['Year','Month','State Name'])
    other = other['Data Value'].sum().unstack()
    other= other[good_states]
    nat = other.sum(axis=1)

    #use the code below to convert states to regions
    other.rename(columns=state_to_region, inplace=True)
    other = other.groupby(axis=1, level=0).sum()
    other['National'] = nat

    #commented code snippet below checks for 0 values
    #print(other) #--> looks ok
    #zero_columns = other.columns[(other == 0).any()] #check for columns with any zero values (can replace with .all())
    #zero = other[zero_columns]
    #print(other[zero_columns])

    #for column in zero:
        #zero_count = (zero[column] == 0).sum()
        #print(f"Number of zeros in column {column}: {zero_count}")

    norm = other/nst * 10000 #use this for normalized datasets
    #norm = other #using this to make non-normalized datasets
    norm['agg_period'] = agg
    norm.set_index('agg_period', inplace=True)
    
    #print(norm)
    return (norm)

def norm_reddit_lag(data: pd.DataFrame, ref: pd.DataFrame):
    """
    Normalize Reddit data with 12 month lag.

    Args:
    - data (pd.DataFrame): comments from reddit cohort filtered for a specific category or drug and grouped by location and month
    - ref (pd.DataFrame): ref is all comments grouped by location and month

    Returns:
    - result_values (pd.DataFrame): Dataframe of normalized and lagged reddit data
    """
    
    lagged_data = data.rolling(window=12, min_periods=1).sum()
    lagged_ref = ref.rolling(window=12, min_periods=1).sum()

    result_values = lagged_data.div(lagged_ref) * 10000
    
    #return lagged_data
    return result_values

def figure_2(cdc: pd.DataFrame, reddit: pd.DataFrame, category: str):
    """
    Visualizes CDC and Reddit overdose/comment rates over time

    Args:
    - cdc (pd.DataFrame): dataframe of cdc normalized time series
    - ref (pd.DataFrame): dataframe of reddit normalized time series
    - category (str): which category we are currently processing

    Returns:
    - none
    """    
    #reformat dataframe
    columns = ['National', 'West', 'Midwest', 'North East', 'South']
    reddit = reddit[columns]
    cdc = cdc[columns]

    merged = pd.concat([cdc, reddit], axis=1)
    level0 = ['National', 'West', 'Midwest', 'Northeast', 'South'] + ['National', 'West', 'Midwest', 'Northeast', 'South']
    level1 = ['CDC'] * 5 + ['Reddit'] * 5
    multi_index = pd.MultiIndex.from_arrays([level0, level1])
    merged.columns = multi_index
  
    merged = merged.reset_index()
    stacked = merged.set_index('agg_period').stack().reset_index()
    stacked.columns = ['agg_period','Datasource', 'National', 'West' , 'Midwest' , 'Northeast', 'South']
    stacked = stacked.sort_values(by='agg_period')

    regions = ['National', 'West' , 'Midwest' , 'Northeast', 'South']
    datasource = ['CDC', 'Reddit']

    #iterate over unique regions and create separate plots
    colorblind_palette = sns.color_palette("colorblind")
    palette = [colorblind_palette[0], colorblind_palette[1]]

    #make new plot for each region
    for region in regions:

        fig, axes = plt.subplots(len(datasource), 1, figsize=(10, 4))
        sns.set_style("whitegrid")

        for i, d in enumerate(datasource):

            ax = axes[i]

            color = palette[i]
            values = stacked[stacked['Datasource'] == d]
            sns.lineplot(data=stacked, x='agg_period', y=values[region], ax=ax, color = color, marker='o', linestyle='-')

            #ax.set_title(d, color='black', pad=-25, loc='left', x=0.05, y=0.95)
            ax.set_title('')
            ax.set_ylabel('Rate', fontsize=14, color='black')
            ax.set_xlabel('')
      
            #format x axis tick marks
            x_ticks = [tick for tick in stacked['agg_period'] if '-01' in tick]
            x_ticks.append('2023-01')
            x_ticks.append('2023-01')
            x_tick_labels = [tick.split('-')[0] for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.tick_params(axis='both', which='major', labelsize=12, color='black')

            num_ticks = 4
            locator = MaxNLocator(nbins=num_ticks, min_n_ticks=num_ticks, prune='both')
            ax.yaxis.set_major_locator(locator)
            ax.grid(axis = 'y', linestyle='--', linewidth=1, color='grey')
            ax.set_xlim([x_ticks[0], x_ticks[-1]])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(1)
    
        #plt.suptitle(region)
        plt.tight_layout()
        plt.savefig(f"Figure2_{region}_{category}.png")
        #plt.show()

def ccf(x, y, lag_max = 6):

    """
    Calculate cross-correlation values between Reddit and CDC rates using the ccf (R package) method

    Args:
    - x (pd.DataFrame): dataframe of reddit national normalized time series
    - y (pd.DataFrame): dataframe of cdc national normalized time series
    - lag_max (int): maximum number of months to check cross correlation values

    Returns:
    - ccf_output (numpy object): cross correlation values
    """    

    backwards = smt.ccf(y, x, unbiased=False, nlags = lag_max + 1)[::-1]
    forwards = smt.ccf(x, y, unbiased=False, nlags = lag_max + 1)
    ccf_output = np.r_[backwards[:-1], forwards]

    return ccf_output

def plot_cc(ccf, category):

    """
    Visualizes CDC and Reddit cross-correlation values over time +/- 6 months

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
    plt.savefig(f"Figure2_ccf_National_{category}.png")

def main(years: List[str], agg_frequency: str):

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
    
    #read in files for CDC overdose data and NST census data
    cdc = pd.read_csv("/Users/Delaney_Smith_1/Desktop/redditCorpus/Analysis/mappings/overdose/VSRR_Provisional_Drug_Overdose_Death_Counts.csv")
    nst = pd.read_csv("/Users/Delaney_Smith_1/Desktop/redditCorpus/Analysis/mappings/overdose/nst_2012-2023.csv") 

    """
    #Option for loading raw Reddit term and comment data using dask (due to large volume)
    #This only need to be run once to create the pickle files
    #If running script only for analysis of normalized data or plotting use next code snipped to load pickled and processed data

    term_file = f"/Users/Delaney_Smith_1/Desktop/redditCorpus/Data/everything/complete_data/aggregated_term_data_pickle_{agg_frequency}.p"
    with open(term_file, "rb+") as f:
            reddit_terms = pickle.load(f)

    #load comment data using Dask
    file_paths = []
    for year in years:
        comment_file = f"/Users/Delaney_Smith_1/Desktop/redditCorpus/Data/everything/complete_data/aggregated_cohort_comments_data_pickle_{agg_frequency}_{year}.csv"  
        file_paths.append(comment_file)

    total_comments = dd.read_csv(file_paths)

    #format total comment dataframe
    total_comments = total_comments[['agg_period', 'TotalCommentsForAuthor', 'area', 'state']]
    total_comments = total_comments.rename(columns={'TotalCommentsForAuthor': 'N'})
    total_comments['state'] = total_comments['state'].astype('category')
    total_comments = total_comments.groupby(['agg_period', 'state'])['N'].sum().reset_index()
    total_comments = total_comments.compute() #converts back to pandas
    total_comments = total_comments.pivot(index='agg_period', columns='state', values='N')
    total_comments.columns.name = None #saved for NFLIS
    #total_comments.to_pickle(f'total_comments_2014-2022.p')
    print (total_comments)  #looks ok
    
    #format total commenters dataframe new way to normalize -- be careful here, we tried normalizing the data differently than in the manuscript
    #new norm files are not what we used in subsequent analysis
    #we normalized by total number of comments in the final manuscript

    total_comments = total_comments[['agg_period', 'state']]
    total_comments['state'] = total_comments['state'].astype('category')
    total_comments = total_comments.groupby(['agg_period', 'state']).size().reset_index()
    #total_comments = total_comments.groupby(['agg_period', 'state']).sum().reset_index()
    total_comments = total_comments.compute() #converts back to pandas
    total_comments = total_comments.pivot(index='agg_period', columns='state')
    total_comments.columns.names = [None, None]
    total_comments.columns.name = None
    total_comments.columns = total_comments.columns.droplevel(level=0)
    print (total_comments)
    total_comments.to_pickle(f'total_comments_new_norm_2014-2022.p')
    
    #format term dataframe
    columns = ['agg_period', 'drug', 'category', 'area', 'state']
    reddit_terms = reddit_terms[columns]

    #standardize agg_period format to YYYY-MM
    def safe_convert(date_str):
        try:
            return pd.to_datetime(date_str, format='%y-%m')
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            return pd.NaT
        except Exception as e:
            print(f"Error converting {date_str}: {e}")
            return pd.NaT
    reddit_terms['agg_period'] = reddit_terms['agg_period'].apply(safe_convert)
    reddit_terms = reddit_terms.dropna(subset=['agg_period'])

    #reddit_terms['agg_period'] = pd.to_datetime(reddit_terms['agg_period'], format='%y-%m') 
    reddit_terms['agg_period'] = reddit_terms['agg_period'].dt.strftime('%Y-%m')
    reddit_terms = reddit_terms[reddit_terms['agg_period'].apply(extract_year).isin(years)]
    reddit_terms = reddit_terms[reddit_terms['state'] != 'Uncategorized']
    grouped_reddit_terms = reddit_terms.groupby(['agg_period', 'drug', 'state']).size().reset_index(name='count')
    drug_data = grouped_reddit_terms #Save this too for NFLIS
    drug_data.to_pickle(f'drug_data_new_norm_2014-2022.p')
    print(drug_data) #looks ok

    """
    
    # Load already processessed & pickled drug and total comments data data
    term_data = f'/Users/Delaney_Smith_1/Desktop/redditCorpus/Data/everything/complete_data/drug_data_2014-2022.p'
    with open(term_data, "rb+") as f:
            drug_data = pickle.load(f)

    total = f'/Users/Delaney_Smith_1/Desktop/redditCorpus/Data/everything/complete_data/total_comments_2005-2022.p'
    with open(total, "rb+") as f:
            total_comments = pickle.load(f)
    
    ##specify what years to include in analysis
    years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    #years = ['2015', '2016', '2017', '2018', '2019', '2022'] #excludes covid years
    
    #The code below filters out states with not enough signal to be included in analysis 'bad states' based on visualization code
    #States removed are Deleware, New Mexico, South Dakota and West Virginia
    good_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
    'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
    'Luisianna','Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
    'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
    'South Carolina', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington','Wisconsin', 'Wyoming', 'District of Columbia'
    ]

    #Here manually input the categories of interest as a list
    #options are: 'Heroin (T40.1)', 'Natural & semi-synthetic opioids (T40.2)', 'Synthetic opioids, excl. methadone (T40.4)', 'Opioids (T40.0-T40.4,T40.6)'

    categories = ["Natural & semi-synthetic opioids (T40.2)"] #manually update here
    
    for c in categories:
        normalized_cdc = norm_cdc(cdc, nst, years, c, states, good_states)
        normalized_cdc.to_pickle(f'cdc_data_formatted_2014-2022_{c}.p')
        #print(normalized_cdc.head(60)) #looks ok
        #zero_columns = normalized_cdc.columns[(normalized_cdc == 0).any()] #check for columns with any zero values (can replace with .all())
        #print(normalized_cdc[zero_columns])
    
        #format reddit data
        drugs = drug_to_category(c)
        drug_data = drug_data[drug_data['drug'].isin(drugs)]
        drug_data = drug_data.groupby(['agg_period', 'state']).sum().unstack().fillna(0)
        drug_data.columns.names = [None, None]
        drug_data = drug_data['count']

        #convert to regions
        drug_data = drug_data[states]
        nat = drug_data.sum(axis=1)
        drug_data = drug_data[good_states]
        drug_data.rename(columns=state_to_region, inplace=True)
        drug_data = drug_data.groupby(axis=1, level=0).sum()
        drug_data['National'] = nat

        #format total comments
        total_comments = total_comments[states]
        nat = total_comments.sum(axis=1)
        total_comments = total_comments[good_states]
        total_comments.rename(columns=state_to_region, inplace=True)
        total_comments = total_comments.groupby(axis=1, level=0).sum()
        total_comments['National'] = nat

        #normalize reddit data
        norm_drug_data = norm_reddit_lag(drug_data, total_comments)
        norm_drug_data.reset_index(inplace=True)
        norm_drug_data = norm_drug_data[norm_drug_data['agg_period'].apply(extract_year).isin(years)]
        norm_drug_data.set_index('agg_period', inplace=True)
        #norm_drug_data.to_pickle(f'reddit_data_formatted_2014-2022_{c}_new_norm.p')
        #print(norm_drug_data.head(60)) #looks ok
        
        #calculate cross correlations
        x = norm_drug_data['National']
        y = normalized_cdc['National']
        
        #differencing series
        #We did cross correlation analysis on raw data and differenced data to try and account for erroneous correlations
        y_dif = y.diff().dropna() 
        x_dif = x.diff(periods=2).dropna() 

        #unit root testing
        result = adfuller(x_dif)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:', result[4])

        #print(x.tolist())
        #print(y.tolist())
        result = ccf(x_dif,y_dif,6)
        #print(result)

        #plot figures
        plot_cc(result, c)
        figure_2(normalized_cdc, norm_drug_data, c)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to analyze the redddit and CDC data.")
    parser.add_argument("--years", type=str, nargs="+", required=True, help="List of years to process, e.g. --years 2012 2013 2014")
    parser.add_argument("--agg_frequency", type=str, choices=['day', 'month', 'year', 'week', 'semi'], default='month', help="Aggregation frequency for term mentions.")

    args = parser.parse_args()
    
    main(args.years, args.agg_frequency)