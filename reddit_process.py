import argparse
import pickle
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import os
import csv
from typing import DefaultDict, Set, List

#################################################
#Title: reddit_process.py
#Author: Delaney Smith, Aadesh Salecha
#Inputs: Raw reddit comment data, extract drug term mentions, lexicon terms file
#Output: Pickled term (extracted terms mapped to drug and drug category), total comment, and cohort (filtered by user) comment files by year
#################################################

def assign_semi(week):
    """
    Assign first or second half of year to semi agg period

    Args:
    - week (int): week of the year
 
    Returns:
    - (int) first or second half year
    """
    if week < 26:
        return 1
    else:
        return 2

def construct_term_lexicon(opBenzTermListFile: str) -> Set[str]:
    """
    Construct the term lexicon from the given file.

    Args:
    - opBenzTermListFile (str): Path to the term list file.

    Returns:
    - Set[str]: A set of terms.
    """
    with open(opBenzTermListFile) as inFile:
        terms = {line.strip().lower().replace(" ", "_") for line in inFile}

    return terms

def construct_term_to_drug_mapping(termMappingLongFormFile: str, termSet: Set[str]) -> DefaultDict[str, str]:
    """
    Construct the term to drug mapping from the given file.

    Args:
    - termMappingLongFormFile (str): Path to the term mapping file.
    - termSet (Set[str]): A set of terms.

    Returns:
    - DefaultDict[str, str]: A mapping from term to drug.
    """
    mapping = defaultdict(lambda : "Uncategorized")
    df = pd.read_csv(termMappingLongFormFile, delimiter='$', header=0)
    df['modified_term'] = df.iloc[:, 2].str.lower().str.replace(" ", "_")
    
    for _, row in df.iterrows():
        if row['modified_term'] in termSet:
            mapping[row['modified_term']] = row[1]
    
    return mapping

def construct_drug_to_category_mapping(drug_category_file: str) -> DefaultDict[str, str]:
    """
    Construct the drug to category mapping from the given file.

    Args:
    - drug_category_file (str): Path to the drug category file.

    Returns:
    - DefaultDict[str, str]: A mapping from drug to category.
    """
    mapping = defaultdict(lambda : "Uncategorized")
    df = pd.read_csv(drug_category_file)
    for _, row in df.iterrows():
        mapping[row['Drug']] = row['Category']
    
    return mapping

def construct_user_to_subreddit_mapping(user_sub_file: str) -> DefaultDict[str, str]:
    """
    Construct the user to subreddit mapping from the given file.

    Args:
    - user_subreddit_file (str): Path to the uData assignments file.

    Returns:
    - DefaultDict[str, str]: A mapping from user to subreddit.
    """
    mapping = defaultdict(lambda : "Uncategorized")
    column_headers = ["user", "subreddit"]
    df = pd.read_csv(user_sub_file, delimiter='\t', header=None, names=column_headers)
    for _, row in df.iterrows():
        mapping[row['user']] = row['subreddit']
    
    return mapping

def construct_sub_to_region_mapping(sub_region_file: str) -> DefaultDict[str, str]:
    """
    Construct the subreddit to region mapping from the given file.

    Args:
    - sub_region_file (str): Path to the subreddit locations file.

    Returns:
    - DefaultDict[str, str]: A mapping from subreddit to region (area).
    """
    mapping = defaultdict(lambda : "Uncategorized")
    df = pd.read_csv(sub_region_file)
    for _, row in df.iterrows():
        mapping[row['subreddit']] = row['area']
    
    return mapping

def construct_sub_to_state_mapping(sub_region_file: str) -> DefaultDict[str, str]:
    """
    Construct the subreddit to state mapping from the given file.

    Args:
    - sub_region_file (str): Path to the subreddit locations file.

    Returns:
    - DefaultDict[str, str]: A mapping from subreddit to state.
    """
    mapping = defaultdict(lambda : "Uncategorized")
    df = pd.read_csv(sub_region_file)
    for _, row in df.iterrows():
        mapping[row['subreddit']] = row['State']
    
    return mapping

def process_directory_terms(drugCategoryFile: str, opBenzTermListFile: str, directory: str, termMappingLongFormFile: str, userSubFile: str, subRegionFile: str, agg_frequency: str = 'day') -> pd.DataFrame:
    
    """
    Process directory for term specific comments.

    Args:
    - drugCategoryFile (str): maps drugs to drug categories
    - opBenzTermListFile (str): path to term set mapping file
    - directory (str): path to data directory 
    - termMappingLongFormFile (str): maps terms to drugs
    - userSubFile (str): adds user's subreddit to file 
    - subRegionFile (str): converts subreddit to region
    - agg_frequency (str): Aggregation frequency ('day', 'month', 'year', 'week', 'semi').

    Returns:
    - pd.DataFrame: final aggregated data.

    """
    aggregated_data = []
    files = os.listdir(directory)
    terms_files = [file for file in files if "_termsTracking" in file]
    
    termSet = construct_term_lexicon(opBenzTermListFile)
    term_to_drug_mapping = construct_term_to_drug_mapping(termMappingLongFormFile, termSet)
    drug_to_category_mapping = construct_drug_to_category_mapping(drugCategoryFile)
    user_to_sub_mapping = construct_user_to_subreddit_mapping(userSubFile)
    sub_to_region = construct_sub_to_region_mapping(subRegionFile)
    sub_to_state = construct_sub_to_state_mapping(subRegionFile)
   
    print("Total term files", len(terms_files))
    for i, terms_file in enumerate(tqdm(terms_files, desc="Processing files")):
        with open(os.path.join(directory, terms_file), 'r', encoding='ISO-8859-1') as file:
            reader = csv.reader(file)
            columns = next(reader)
            data = list(reader)
        terms_df = pd.DataFrame(data, columns=columns)
        
        termSet = set(terms_df['term'])
        if agg_frequency == 'day':
            terms_df['agg_period'] = terms_df['postTime'].str.split('-').str[:3].str.join('-')
        elif agg_frequency == 'month':
            terms_df['agg_period'] = terms_df['postTime'].str.split('-').str[:2].str.join('-')
        elif agg_frequency == 'year':
            terms_df['agg_period'] = terms_df['postTime'].str.split('-').str[0]
        elif agg_frequency == 'semi':
            terms_df['datetime'] = pd.to_datetime(terms_df['postTime'], format='%y-%m-%d-%H-%M')
            terms_df['year'], terms_df['week'] = terms_df['datetime'].dt.isocalendar().year, terms_df['datetime'].dt.isocalendar().week
            terms_df['semi'] = terms_df['week'].apply(assign_semi)
            terms_df['agg_period'] = terms_df['year'].astype(str) + "_" + terms_df['semi'].astype(str)
        elif agg_frequency == 'week':
            # Convert 'postTime' to datetime with a custom format
            terms_df['datetime'] = pd.to_datetime(terms_df['postTime'], format='%y-%m-%d-%H-%M')
            terms_df['year'], terms_df['week'] = terms_df['datetime'].dt.isocalendar().year, terms_df['datetime'].dt.isocalendar().week
            terms_df['agg_period'] = terms_df['year'].astype(str) + "-W" + terms_df['week'].astype(str)
        else:
            raise ValueError(f"Unsupported aggregation frequency: {agg_frequency}")

        # Use add the drug, subreddit, state, and area columns to dataframe from existing files
        terms_df['drug'] = terms_df['term'].map(term_to_drug_mapping)
        terms_df['category'] = terms_df['drug'].map(drug_to_category_mapping)
        terms_df['subreddit'] = terms_df['user'].map(user_to_sub_mapping)
        terms_df['area'] = terms_df['subreddit'].map(sub_to_region)
        terms_df['state'] = terms_df['subreddit'].map(sub_to_state)
        
        #aggregated = terms_df.groupby(['agg_period', 'drug', 'area']).size().reset_index(name='count')
        aggregated_data.append(terms_df)
        
    
    final_aggregated_data = pd.concat(aggregated_data, axis=0)
    return final_aggregated_data


""" 
Aadesh added this function
It builds on the previous function by de-duplicating comments
This is now useful because with the new data fill files, we sometimes have duplicate comments
"""
seen_comments_shards = {}
def process_directory_terms_v2(drugCategoryFile: str, opBenzTermListFile: str, directory: str, termMappingLongFormFile: str, userSubFile: str, subRegionFile: str, agg_frequency: str = 'day') -> pd.DataFrame:
    
    """
    Process directory for term specific comments.

    Args:
    - drugCategoryFile (str): maps drugs to drug categories
    - opBenzTermListFile (str): path to term set mapping file
    - directory (str): path to data directory 
    - termMappingLongFormFile (str): maps terms to drugs
    - userSubFile (str): adds user's subreddit to file 
    - subRegionFile (str): converts subreddit to region
    - agg_frequency (str): Aggregation frequency ('day', 'month', 'year', 'week', 'semi').

    Returns:
    - pd.DataFrame: final aggregated data.

    """
    global seen_comments_shards
    aggregated_data = []
    files = os.listdir(directory)
    terms_files = [file for file in files if "_termsTracking" in file]
    
    termSet = construct_term_lexicon(opBenzTermListFile)
    term_to_drug_mapping = construct_term_to_drug_mapping(termMappingLongFormFile, termSet)
    drug_to_category_mapping = construct_drug_to_category_mapping(drugCategoryFile)
    user_to_sub_mapping = construct_user_to_subreddit_mapping(userSubFile)
    sub_to_region = construct_sub_to_region_mapping(subRegionFile)
    sub_to_state = construct_sub_to_state_mapping(subRegionFile)
   
    print("Total term files", len(terms_files))
    for i, terms_file in enumerate(tqdm(terms_files, desc="Processing files")):
        with open(os.path.join(directory, terms_file), 'r', encoding='ISO-8859-1') as file:
            reader = csv.reader(file)
            columns = next(reader)
            data = list(reader)
        terms_df = pd.DataFrame(data, columns=columns)
        
        termSet = set(terms_df['term'])
        if agg_frequency == 'day':
            terms_df['agg_period'] = terms_df['postTime'].str.split('-').str[:3].str.join('-')
        elif agg_frequency == 'month':
            terms_df['agg_period'] = terms_df['postTime'].str.split('-').str[:2].str.join('-')
        elif agg_frequency == 'year':
            terms_df['agg_period'] = terms_df['postTime'].str.split('-').str[0]
        elif agg_frequency == 'semi':
            terms_df['datetime'] = pd.to_datetime(terms_df['postTime'], format='%y-%m-%d-%H-%M')
            terms_df['year'], terms_df['week'] = terms_df['datetime'].dt.isocalendar().year, terms_df['datetime'].dt.isocalendar().week
            terms_df['semi'] = terms_df['week'].apply(assign_semi)
            terms_df['agg_period'] = terms_df['year'].astype(str) + "_" + terms_df['semi'].astype(str)
        elif agg_frequency == 'week':
            # Convert 'postTime' to datetime with a custom format
            terms_df['datetime'] = pd.to_datetime(terms_df['postTime'], format='%y-%m-%d-%H-%M')
            terms_df['year'], terms_df['week'] = terms_df['datetime'].dt.isocalendar().year, terms_df['datetime'].dt.isocalendar().week
            terms_df['agg_period'] = terms_df['year'].astype(str) + "-W" + terms_df['week'].astype(str)
        else:
            raise ValueError(f"Unsupported aggregation frequency: {agg_frequency}")

        # Use add the drug, subreddit, state, and area columns to dataframe from existing files
        terms_df['drug'] = terms_df['term'].map(term_to_drug_mapping)
        terms_df['category'] = terms_df['drug'].map(drug_to_category_mapping)
        terms_df['subreddit'] = terms_df['user'].map(user_to_sub_mapping)
        terms_df['area'] = terms_df['subreddit'].map(sub_to_region)
        terms_df['state'] = terms_df['subreddit'].map(sub_to_state)
        
        length_before = len(terms_df)
        # Filter out rows with comment_id already in seen_comments_shards
        terms_df = terms_df[~terms_df.apply(lambda row: row['comment_id'] in seen_comments_shards.get(row['agg_period'], set()), axis=1)]
        length_after = len(terms_df)
        if (abs(length_before - length_after) > 0):
            print(f"Filtered out {length_before - length_after} duplicate term comments for {terms_file}")

        # If we dropped all data then shortcircut
        if terms_df.empty:
            continue

        #aggregated = terms_df.groupby(['agg_period', 'drug', 'area']).size().reset_index(name='count')
        aggregated_data.append(terms_df)
        for year_month, group in terms_df.groupby('agg_period'):
            if year_month not in seen_comments_shards:
                seen_comments_shards[year_month] = set()
            seen_comments_shards[year_month].update(group['comment_id'])
        
    if len(aggregated_data):
        final_aggregated_data = pd.concat(aggregated_data, axis=0)
    else:
        final_aggregated_data = pd.DataFrame()
    return final_aggregated_data

def process_cohort_comments(directory: str, userSubFile: str, subRegionFile: str) -> pd.DataFrame:
    """
    Process directory for cohort total comments.

    Args:
    - directory (str): path to data directory 
    - userSubFile (str): adds user's subreddit to file 
    - subRegionFile (str): converts subreddit to region

    Returns:
    - pd.DataFrame: final data

    """
    aggregated_data = []
    files = os.listdir(directory)
    comment_files = [file for file in files if "_cohortCommentTracking.txt" in file]
   
    print("Total cohort files", len(comment_files))
    for comment_file in tqdm(comment_files, desc="Processing files"):
        comment_df = pd.read_csv(os.path.join(directory, comment_file))
        aggregated_data.append(comment_df)
    
    data = pd.concat(aggregated_data, axis=0)

    user_to_sub_mapping = construct_user_to_subreddit_mapping(userSubFile)
    sub_to_region = construct_sub_to_region_mapping(subRegionFile)
    sub_to_state = construct_sub_to_state_mapping(subRegionFile)

    data = data.rename(columns={'Author': 'user'})
    data['subreddit'] = data['user'].map(user_to_sub_mapping)
    data['area'] = data['subreddit'].map(sub_to_region)
    data['state'] = data['subreddit'].map(sub_to_state)

    return data

# Global dictionary to track seen comment IDs for each aggregation period
seen_cohort_comments_shards = defaultdict(set)
""" 
Aadesh added this function
It builds on the previous function by de-duplicating comments
This is now useful because with the new data fill files, we sometimes have duplicate comments
"""
def process_cohort_comments_v2(directory: str, userSubFile: str, subRegionFile: str) -> pd.DataFrame:
    """
    Process directory for cohort total comments. This version of the function includes a de-duplication mechanism
    to ensure that each comment is only counted once within each aggregation period. The de-duplication is done
    using the comment_id field. A global dictionary seen_cohort_comments_shards is maintained where each key is an
    aggregation period (e.g., a specific year and month), and the value is a set of comment IDs that have been seen
    in that period. Before aggregating the data, the function filters out rows in the DataFrame where the comment_id
    is already in the set of seen comment IDs for the corresponding aggregation period.

    Args:
    - directory (str): path to data directory 
    - userSubFile (str): adds user's subreddit to file 
    - subRegionFile (str): converts subreddit to region

    Returns:
    - pd.DataFrame: final data

    """
    aggregated_data = []
    files = os.listdir(directory)
    comment_files = [file for file in files if "_cohortCommentTracking.txt" in file]
   
    print("Total cohort files", len(comment_files))
    for comment_file in tqdm(comment_files, desc="Processing files"):
        comment_df = pd.read_csv(os.path.join(directory, comment_file))

        # Extract year and month from "Date" column to create an aggregation period
        comment_df['agg_period'] = comment_df['Date'].str[:7]
        comment_df['value'] = comment_df['Date'].str[8:] + comment_df['Author'] + comment_df['TotalCommentsForAuthor'].astype(str)

        for agg_period, group in comment_df.groupby('agg_period'):
            # Filter out rows where the value is already in the set of seen values for this aggregation period
            len_before = len(group)
            group = group[~group['value'].isin(seen_cohort_comments_shards[agg_period])]
            len_after = len(group)
            if (abs(len_before - len_after) > 0):
                print(f"Filtered out {len_before - len_after} duplicate cohort comments for {comment_file}")
                    
            # If we dropped all data then shortcircut
            if group.empty:
                continue

            # Update the set of seen values for this aggregation period
            seen_cohort_comments_shards[agg_period].update(group['value'].values)

            aggregated_data.append(group)

    data = pd.concat(aggregated_data, axis=0)

    user_to_sub_mapping = construct_user_to_subreddit_mapping(userSubFile)
    sub_to_region = construct_sub_to_region_mapping(subRegionFile)
    sub_to_state = construct_sub_to_state_mapping(subRegionFile)

    data = data.rename(columns={'Author': 'user'})
    data['subreddit'] = data['user'].map(user_to_sub_mapping)
    data['area'] = data['subreddit'].map(sub_to_region)
    data['state'] = data['subreddit'].map(sub_to_state)

    return data
        

def process_total_comments(directory: str) -> pd.DataFrame:
    """
    Read in all total comment files for a given year and combine them into one dataframe

    Args:
    - directory (str): string containing file path to yearly data directory
 
    Returns:
    - pd.DataFrame: Aggregated data.
    """
    aggregated_data = []
    files = os.listdir(directory)
    comment_files = [file for file in files if "_totalCommentTracking.txt" in file]
   
    print("Total comment files", len(comment_files))
    for comment_file in tqdm(comment_files, desc="Processing files"):
        comment_df = pd.read_csv(os.path.join(directory, comment_file))
        aggregated_data.append(comment_df)
    
    final_aggregated_data = pd.concat(aggregated_data, axis=0)
    return final_aggregated_data

def aggregate_comments_by_frequency(data: pd.DataFrame, agg_frequency: str) -> pd.DataFrame:
    """
    Label the total comments based on the specified frequency.

    Args:
    - data (pd.DataFrame): Data with Date and TotalComments.
    - agg_frequency (str): Aggregation frequency ('day', 'month', 'year', 'week', 'semi').

    Returns:
    - pd.DataFrame: Agg period labeled data.
    """
    def safe_convert(date_str):
        try:
            return pd.to_datetime(date_str)
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            return pd.NaT
        except Exception as e:
            print(f"Error converting {date_str}: {e}")
            return pd.NaT
    data['Date'] = data['Date'].apply(safe_convert)
    data = data.dropna(subset=['Date'])
    
    if agg_frequency == 'day':
        data['agg_period'] = data['Date']
    elif agg_frequency == 'month':
        data['agg_period'] = data['Date'].dt.strftime('%Y-%m')
    elif agg_frequency == 'year':
        data['agg_period'] = data['Date'].dt.strftime('%Y')
    elif agg_frequency == 'week':
        data['year'], data['week'] = data['Date'].dt.isocalendar().year, data['Date'].dt.isocalendar().week
        data['agg_period'] = data['year'].astype(str) + "-W" + data['week'].astype(str)
    elif agg_frequency == 'semi':
        data['year'], data['week'] = data['Date'].dt.isocalendar().year, data['Date'].dt.isocalendar().week
        data['semi'] = data['week'].apply(assign_semi)
        data['agg_period'] = data['year'].astype(str) + "_" + data['semi'].astype(str)
    else:
        raise ValueError(f"Unsupported aggregation frequency: {agg_frequency}")

    #Only sum the 'TotalComments' column
    #aggregated_data = data.groupby('agg_period')['TotalComments'].sum().reset_index()
    return data

def aggregate_cohort_comments_by_frequency(data: pd.DataFrame, agg_frequency: str) -> pd.DataFrame:
    """
    Label the total comments based on the specified frequency.

    Args:
    - data (pd.DataFrame): Data with Date and TotalComments.
    - agg_frequency (str): Aggregation frequency ('day', 'month', 'year', 'week', 'semi').

    Returns:
    - pd.DataFrame: Data labeled with agg period.
    """
    def safe_convert(date_str):
        try:
            return pd.to_datetime(date_str)
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            return pd.NaT
        except Exception as e:
            print(f"Error converting {date_str}: {e}")
            return pd.NaT
    data['Date'] = data['Date'].apply(safe_convert)
    data = data.dropna(subset=['Date'])
    
    if agg_frequency == 'day':
        data['agg_period'] = data['Date']
    elif agg_frequency == 'month':
        data['agg_period'] = data['Date'].dt.strftime('%Y-%m')
    elif agg_frequency == 'year':
        data['agg_period'] = data['Date'].dt.strftime('%Y')
    elif agg_frequency == 'week':
        data['year'], data['week'] = data['Date'].dt.isocalendar().year, data['Date'].dt.isocalendar().week
        data['agg_period'] = data['year'].astype(str) + "-W" + data['week'].astype(str)
    elif agg_frequency == 'semi':
        data['year'], data['week'] = data['Date'].dt.isocalendar().year, data['Date'].dt.isocalendar().week
        data['semi'] = data['week'].apply(assign_semi)
        data['agg_period'] = data['year'].astype(str) + "_" + data['semi'].astype(str)
    else:
        raise ValueError(f"Unsupported aggregation frequency: {agg_frequency}")

    #aggregated_data = data.groupby(['agg_period', 'area'])['TotalCommentsForAuthor'].sum().reset_index()
    return data

def main(data_dir: str, years: List[str], agg_frequency: str, force_reprocess: bool):

#input files needed for aggregation
    termMappingLongFormFile = "./mappings/termMappingLongForm.csv"
    drug_category_file = "./mappings/drug_category.csv"
    opBenzTermListFile = "./mappings/opBenz_term_list.txt"
    userSubFile = "./mappings/uData_assignments_v2.tsv"
    subRegionFile = "./mappings/subreddit_locations_manually_cleaned.csv"

    pickle_dir = "DelaneyPickleFiles"
    os.makedirs(pickle_dir, exist_ok=True)

# performs total comment aggregation (totalCommentTracking)
    all_aggregated_totals = []
    print("Aggregate total comment data..")
    for year in years:
        directory = f"{data_dir}/{year}/"
        pickle_file =  os.path.join(pickle_dir, f"total_comments_data_{year}_pickle.p")
        if not os.path.exists(pickle_file) or force_reprocess:
            total_comments_data = process_total_comments(directory)
            with open(pickle_file, "wb") as f:
                pickle.dump(total_comments_data, f)
        else:
            with open(pickle_file, "rb+") as f:
                total_comments_data = pickle.load(f)

        all_aggregated_totals.append(total_comments_data)
    final_dataframe = pd.concat(all_aggregated_totals, ignore_index=True)
    aggregated_comments_data = aggregate_comments_by_frequency(final_dataframe, agg_frequency)
    combined_aggregated_comments = pd.concat([aggregated_comments_data], axis=0)
    agg_pickle_file =  os.path.join(pickle_dir, f"aggregated_total_comments_data_pickle_{agg_frequency}.p")
    print(combined_aggregated_comments)
    with open(agg_pickle_file, "wb") as f:
        pickle.dump(combined_aggregated_comments, f)

#  performs term-based comment aggregation (termsTracking)
    term_aggregated_data = []
    print("Aggregate term data..")
    for year in years:
        directory = f"{data_dir}/{year}/"
        pickle_file =  os.path.join(pickle_dir, f"term_data_{year}_pickle_{agg_frequency}.p")
        if not os.path.exists(pickle_file) or force_reprocess:
            aggregated_data_csv = process_directory_terms_v2(drug_category_file, opBenzTermListFile, directory, termMappingLongFormFile, userSubFile, subRegionFile, agg_frequency=agg_frequency)
            with open(pickle_file, "wb") as f:
                pickle.dump(aggregated_data_csv, f)
        else:
            with open(pickle_file, "rb+") as f:
                aggregated_data_csv = pickle.load(f)

        term_aggregated_data.append(aggregated_data_csv)
    all_term_data = pd.concat(term_aggregated_data, axis=0)
    agg_pickle_file =  os.path.join(pickle_dir, f"aggregated_term_data_pickle_{agg_frequency}.p")
    print(all_term_data)
    with open(agg_pickle_file, "wb") as f:
        pickle.dump(all_term_data, f)

    # performs cohort-specific total comment aggregation (cohortCommentTracking)
    cohort_aggregated_totals = []
    print("Aggregate cohort total comment data..")
    for year in years:
        directory = f"{data_dir}/{year}/"
        pickle_file =  os.path.join(pickle_dir, f"cohort_comments_data_{year}_pickle_{agg_frequency}.p")
        if not os.path.exists(pickle_file) or force_reprocess:
            total_comments_data = process_cohort_comments_v2(directory, userSubFile, subRegionFile)
            with open(pickle_file, "wb") as f:
                pickle.dump(total_comments_data, f)
        else:
            with open(pickle_file, "rb+") as f:
                total_comments_data = pickle.load(f)

        cohort_aggregated_totals.append(total_comments_data)
    final_dataframe = pd.concat(cohort_aggregated_totals, ignore_index=True)

    all_cohort_data = aggregate_cohort_comments_by_frequency(final_dataframe, agg_frequency)
    combined_cohort_comments = pd.concat([all_cohort_data], axis=0)
    agg_cohort_file =  os.path.join(pickle_dir, f"aggregated_cohort_comments_data_pickle_{agg_frequency}.p")
    with open(agg_cohort_file, "wb") as f:
       pickle.dump(combined_cohort_comments, f)
    print(final_dataframe)
    agg_cohort_file = f"aggregated_cohort_comments_data_pickle_{agg_frequency}.csv"
    final_dataframe.to_csv(agg_cohort_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process the redddit data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Where is the data")
    parser.add_argument("--years", type=str, nargs="+", required=True, help="List of years to process, e.g. --years 2012 2013 2014")
    parser.add_argument("--agg_frequency", type=str, choices=['day', 'month', 'year', 'week', 'semi'], default='month', help="Aggregation frequency for term mentions.")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of data even if a cached version exists.")

    args = parser.parse_args()

    print("Starting processing..")
    main(args.data_dir, args.years, args.agg_frequency, args.force_reprocess)
