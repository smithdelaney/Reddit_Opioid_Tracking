import numpy as np
import csv
import pandas as pd
import collections

def read_and_filter_csv(file_path):
    df = pd.read_csv(file_path)
    df_filtered = df[['Year', '2023', '2020']]
    return df_filtered

def construct_user_to_state_map(user_to_subreddit_file, subreddit_to_location_file):
    # Step 1: Construct a map from user to subreddit
    user_to_subreddit = {}
    with open(user_to_subreddit_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            user, subreddit = row
            user_to_subreddit[user] = subreddit

    # Step 2: Construct a map from subreddit to state
    subreddit_to_state = {}
    with open(subreddit_to_location_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            state, subreddit, _ = row
            subreddit_to_state[subreddit] = state

    # Step 3: Combine the two maps to get a map from user to state
    user_to_state = {user: subreddit_to_state.get(subreddit, None) for user, subreddit in user_to_subreddit.items()}

    return user_to_state
import matplotlib.pyplot as plt
import scipy.stats

def generate_state_user_population_table(user_to_state_map, population_df, output_csv_path):
    # Count users per state
    user_count_per_state = collections.Counter(user_to_state_map.values())

    # Prepare population data
    population_df = population_df.rename(columns={'2023': 'Population (2023)', 'Year': 'State'})
    population_df['Population (2020)'] = population_df['State'].apply(lambda x: populations[populations['Year'] == x]['2020'].values[0] if x in populations['Year'].values else 0)

    # Merge user counts with population data
    population_df['User Count'] = population_df['State'].map(user_count_per_state).fillna(0).astype(int)
    
    # Sort the DataFrame by 'User Count' in descending order
    population_df = population_df.sort_values('User Count', ascending=False)

    # Save the table to a CSV file
    population_df[['State', 'User Count', 'Population (2023)', 'Population (2020)']].to_csv(output_csv_path, index=False)

    # Plotting
    plot_state_user_population(population_df)

import seaborn as sns
from matplotlib.ticker import FuncFormatter

def plot_state_user_population(population_df):
    # sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and a set of subplots
    ax.scatter(population_df['Population (2020)'] / 1e6, population_df['User Count'], color='blue', alpha=0.5)
    # Calculate and plot linear trend line
    z = np.polyfit(population_df['Population (2020)'] / 1e6, population_df['User Count'], 1)
    p = np.poly1d(z)
    ax.plot(population_df['Population (2020)'] / 1e6, p(population_df['Population (2020)'] / 1e6), "r", alpha = 0.5)
    title_fontsize = 20
    ticks_fontsize = 15
    axis_label_fontsize = 18
    ax.set_title('Geo-located Reddit Users vs. State Population', fontsize=title_fontsize, fontweight='normal')
    ax.set_xlabel('State Population - 2020 (Millions)', fontsize=axis_label_fontsize, fontweight='normal')
    ax.set_ylabel('Reddit Users in Cohort', fontsize=axis_label_fontsize, fontweight='normal')
    ax.set_xticks([10, 20, 30])
    ax.set_yticks([0, 50000, 100000, 150000])
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # ax.tick_params(labelsize=ticks_fontsize)
    ax.set_xlim(left=0, right =40)
    ax.set_ylim(bottom=0)
    # ax.set_ylim(bottom=-population_df['User Count'].max() * 0.02, top=population_df['User Count'].max() * 1.05)
    ax.grid(True)
    # ax.margins(y=0.01)
    # ax.minorticks_on()
    ax.tick_params(labelsize=ticks_fontsize, which='both', length=6, width=2, grid_color='gray', grid_alpha=0.5)
    # plt.axhline(0, color='black', linewidth=1)
    ax.spines['right'].set_color('grey')
    # ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_color('grey')
    fig.subplots_adjust(left=0.15)
    # ax.spines['left'].set_linewidth(1)
    fig.savefig('./state_population_vs_reddit_users.svg', format='svg', dpi=1200)
    plt.show()

    # Kendall Tau calculation
    tau, p_value = scipy.stats.kendalltau(population_df['Population (2020)'], population_df['User Count'])
    print(f"Kendall Tau correlation: {tau}, P-value: {p_value}")

if __name__ == "__main__":
    global_dir = "/home/asalecha/AdamPNAS/EverythingPipeline/Analysis/"
    # populations = read_and_filter_csv('./mappings/overdose/NST_pop_estimate_2012-2023.csv')
    populations = read_and_filter_csv('./mappings/overdose/NST_pop_estimate_statesonly.csv')
    userToState = construct_user_to_state_map(f"{global_dir}/mappings/userToSubreddit.tsv", f"{global_dir}/mappings/subredditToLocation.csv")
    
    output_csv_path = './state_user_population_table.csv'
    generate_state_user_population_table(userToState, populations, output_csv_path)