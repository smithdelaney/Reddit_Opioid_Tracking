# Monitoring the opioid epidemic via social media discussions

# Abstract
Opioid-involved overdose deaths have risen significantly in the United States since 1999 with over 80,000 deaths annually since 2021, primarily driven by synthetic opioids, like fentanyl. Responding to the rapidly changing opioid crisis requires reliable and timely information. One possible source of such data is the social media platforms with billions of user-generated posts, a fraction of which are about drug use. We therefore assessed the utility of Reddit data for surveillance of the opioid epidemic, covering prescription, heroin, and synthetic drugs (as of September 2024, up-to-date Reddit data was still accessible on the open web). Specifically, we built a natural language processing pipeline to identify opioid-related comments and created a cohort of 1,689,039 Reddit users, each assigned to a state based on their previous Reddit activity. We followed these users from 2010 through 2022, measured their opioid-related posting activity over time, and compared this posting activity against CDC overdose and National Forensic Laboratory Information System (NFLIS) drug report rates. To simulate the real-world prediction of synthetic drug overdose rates, we added near real-time Reddit data to a model relying on CDC mortality data with a typical 6-month reporting lag and found that Reddit data significantly improved the prediction accuracy of overdose rates. We observed drastic, largely unpredictable changes in both Reddit and overdose patterns during the COVID-19 pandemic. Reddit discussions covered a wide variety of drug types that are currently missed by official reporting. This work suggests that social media can help identify and monitor known and emerging drug epidemics and that this data is a public health “common good” to which researchers should continue to have access.

# Compressed Folders
The The google drive link below contains a compressed input file folder with the pickled 2019 Reddit raw data for input files for running the scripts in this reposiroty. It does not contain all the Reddit data used in this analysis (2015-2022)

https://drive.google.com/drive/folders/19PLuSUQZ5062FY7lg0QwIxou4LAg2ky4?usp=sharing


The compressed mappings file contains a compressed mappings folder contains all external data (CDC, NFLIS, NST) used for benchmark data.

# Analysis Scripts
The order of operations for the scripts is as follows. Outputs from script 1 are used in script 2, then outputs from 2 are used in 3 and 4.
1. reddit_process.py
2. cdc_analysis.py
3. nflis_analysis.py
4. ARIMA_modeling.py
5. Visualization code --> folder containing code used for supplemental Table (not needed in remaining pipeline)
