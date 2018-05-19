import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import time

print("***************************************")
print("Start Data Preparation For the Asylum ")
print("Appeals:")
print("Import & Merge Data")
print("***************************************")

# Imports 3 datasets used for project: (1) Master Proceedings as processed by Dunn, (2) Judge Bios, (3) Appeals
DATAFOLDER = "/data/Dropbox/Data/Asylum_Courts/"

# Master Proceedings (processed by Sagent/Dunn)
master_dunn = pd.read_csv(os.path.join(DATAFOLDER,
                                       'AsylumAdj/data_for_model/_decision_scheduling_merge_final_converted.csv'),
                          encoding='latin-1', low_memory=False)  # gets UnicodeDecodeError otherwise
master_dunn.rename(columns={'dec_type_string': 'original_dec_type_string',
                            'dec_string': 'original_dec_string',
                            'grant': 'original_granted'},
                   inplace=True)
master_dunn.drop('Unnamed: 0', axis=1, inplace=True)

### Judge Bios
judge_bio = pd.read_csv(os.path.join(DATAFOLDER, 'AsylumAdj/data/cleaned_judge_bios.csv'))

# import main table
tblAppeal = pd.read_csv(os.path.join(DATAFOLDER, 'raw/tblAppeal.csv'), low_memory=False)

# import lookup tables (Python 2.7)
bia_appeal_category = pd.read_excel(os.path.join(DATAFOLDER, 'raw/BIA Appeal Data File code translations.xlsx'),
                                    sheetname='BIA Appeal Category', skip_footer=7)
bia_appeal_type = pd.read_excel(os.path.join(DATAFOLDER, 'raw/BIA Appeal Data File code translations.xlsx'),
                                sheetname='BIA Appeal Type', skip_footer=3)
bia_decision_type = pd.read_excel(os.path.join(DATAFOLDER, 'raw/BIA Appeal Data File code translations.xlsx'),
                                  sheetname='BIA decision type', skip_footer=2)
bia_decision_code = pd.read_excel(os.path.join(DATAFOLDER, 'raw/BIA Appeal Data File code translations.xlsx'),
                                  sheetname='BIA decision code', skip_footer=2)

# join them
tblAppeal_df = tblAppeal.merge(bia_appeal_category, how='left', left_on='strAppealCategory', right_on='Code') \
    .rename(columns={'Description': 'strAppealCategoryDesc'}).drop('Code', axis=1) \
    .merge(bia_appeal_type, how='left', left_on='strAppealType', right_on='Code') \
    .rename(columns={'Description': 'strAppealTypeDesc'}).drop('Code', axis=1) \
    .merge(bia_decision_code, how='left', left_on='strBIADecision', right_on='Code') \
    .rename(columns={'Description': 'strBIADecisionDesc'}).drop('Code', axis=1) \
    .merge(bia_decision_type, how='left', left_on='strBIADecisionType', right_on='Code') \
    .rename(columns={'Description': 'strBIADecisionTypeDesc'}).drop('Code', axis=1)

# drop appeals with no case number, proceeding number, or decision
tblAppeal.dropna(subset=['idncase', 'idnProceeding', 'strBIADecision'], inplace=True)
tblAppeal = tblAppeal[tblAppeal['idnProceeding'] != 0]  # drop zeros

# some strBIADecision don't have corresponding code translations; also drop
print(tblAppeal_df[pd.isnull(tblAppeal_df['strBIADecisionDesc'])]['strBIADecision'].value_counts())
tblAppeal_df.dropna(subset=['strBIADecisionDesc'], inplace=True)

print(tblAppeal_df.info())
tblAppeal_df.sample(3)

### Merge Them

# convert appeal indexes to integers for joins
tblAppeal['idncase'] = tblAppeal['idncase'].astype(int)
tblAppeal['idnProceeding'] = tblAppeal['idnProceeding'].astype(int)

# joins datasets
df = master_dunn.merge(tblAppeal_df, how='left',
                       left_on=['idncase', 'idnproceeding'], right_on=['idncase', 'idnProceeding'])
df = df.merge(judge_bio, how='left', on='ij_code')
print(df.columns.tolist())
df.info()

# Summarize Funnel Stats

print("***************************************")
print("Summarize Funnel Stats")
print("***************************************")

# add/reformat common variables between appeals and non-appeals before splitting them

df['judge_missing_bio'] = np.where(pd.isnull(df['Male_judge']), 1, 0)
df['ij_code_nat'] = df['ij_code'] + '_' + df['nat_string']
df['comp_dt'] = pd.to_datetime(dict(year=df['comp_year'], month=df['comp_month'], day=df['comp_day']))
df['comp_year_month'] = 12 * (df['comp_year'] - 1970) + df['comp_month']

df['datAppealFiled_dt'] = pd.to_datetime(df['datAppealFiled'], errors='coerce')
df['datAppealFiled_year'] = df['datAppealFiled_dt'].dt.year
df['datAppealFiled_month'] = df['datAppealFiled_dt'].dt.month
df['datAppealFiled_year_month'] = 12 * (df['datAppealFiled_year'] - 1970) + df['datAppealFiled_month']

df['datBIADecision_dt'] = pd.to_datetime(df['datBIADecision'], errors='coerce')
df['datBIADecision_year'] = df['datBIADecision_dt'].dt.year
df['datBIADecision_month'] = df['datBIADecision_dt'].dt.month
df['datBIADecision_year_month'] = 12 * (df['datBIADecision_year'] - 1970) + df['datBIADecision_month']

# check % of original proceedings that were granted
original_cases_total = df['idnproceeding'].nunique()
original_cases_granted = df[df['original_granted'] == 1]['idnproceeding'].nunique()
original_cases_granted_pct = float(original_cases_granted) / original_cases_total
print("Of the {} original proceedings, {} ({:.1f}%) were granted asylum.".format(original_cases_total,
                                                                                 original_cases_granted,
                                                                                 100 * original_cases_granted_pct))

# check % of denied proceedings that appealed
denied_cases = df[df['original_granted'] == 0].copy()
denied_cases['appealed'] = np.where(pd.notnull(denied_cases['idnAppeal']), 1, 0)
denied_cases_total = denied_cases['idnproceeding'].nunique()
denied_cases_appealed = denied_cases[denied_cases['appealed'] == 1]['idnproceeding'].nunique()
denied_cases_appealed_pct = float(denied_cases_appealed) / denied_cases_total
print("Of the {} denied proceedings, {} ({:.1f}%) appealed the decision.".format(denied_cases_total,
                                                                                 denied_cases_appealed,
                                                                                 100 * denied_cases_appealed_pct))

# get appeals vs. non-appeals datasets
appeals = denied_cases[denied_cases['appealed'] == 1].copy()
non_appeals = denied_cases[denied_cases['appealed'] == 0].copy()

print("***************************************")
print("Finalize Scope of Appeals")
print("***************************************")

"""
In this section we will: 
- Drop appeals outside relevant scope, defined to be ('Appeal of IJ MTR', 'Case Appeal', 'Circuit Court Remand', 
  'Interlocutory Appeal', 'MTR BIA') 
- Deduplicate multiple appeals tied to the same proceeding (by taking the appeal with the last BIA Decision) 
- Drop appeals without mandatory features ('datAppealFiled_year', 'case_type_string') 
- Group appeal outcomes into 'positive' vs. 'negative' binary labels; a small subset deemed to be 'neutral' 
  (e.g. dismissal due to incomplete paperwork) is also dropped. 

Note that we also implicitly dropped appeals made by government by subsetting 
appeals from the denied proceedings (i.e. government is likely to contest 
verdicts in favor of respondents rather than the opposite). 
"""

# check appeal and case types
appeals.groupby(['strAppealTypeDesc', 'case_type_string']).size().unstack().fillna(0)

# filter for relevant appeal types
selected_appeal_types = ['Appeal of IJ MTR', 'Case Appeal', 'Circuit Court Remand', 'Interlocutory Appeal', 'MTR BIA']
appeals = appeals[appeals['strAppealTypeDesc'].isin(selected_appeal_types)]
print("After filtering for relevant appeal types, {} rows remain".format(len(appeals)))

# de-duplicate multiple appeals (each case-proceeding should be unique) by retaining the last appeal
appeals = appeals.sort_values(by=['idncase', 'idnProceeding', 'datBIADecision_dt'],
                              ascending=[True, True, False])
appeals.drop_duplicates(subset=['idncase', 'idnProceeding'], keep='first', inplace=True)
print("After deduplicating multiple appeals, {} rows remain".format(len(appeals)))

# drop appeals without mandatory features
mandatory_features = ['datAppealFiled_year', 'case_type_string']
appeals.dropna(subset=mandatory_features, inplace=True)
print("After dropping appeals without mandatory features, {} rows remain".format(len(appeals)))

# designate appeal decision type
positive_labels = ['Background Check Remand', 'Grant With No Remand', 'Granted', 'Remand',
                   'Sustain', 'Temporary Protected Status', 'Termination']
negative_labels = ['Denied', "Dismiss Appeal/Affirm IJ's Decision", 'Dismissed (Grant V/D 30 days)',
                   'Dismissed (Voluntary Departure Granted)', 'Rejection', 'SUMMARY AFFIRMANCE/VD',
                   'Summary Affirmance', 'Summary Dismiss', 'Summary Dismissal (O) Other',
                   'Summary Dismissal (a) inad reason on appeal', 'Summary Dismissal - Both (a) & (e)']
appeals['granted'] = np.where(appeals['strBIADecisionDesc'].isin(positive_labels), 1,
                              np.where(appeals['strBIADecisionDesc'].isin(negative_labels), 0, None))
appeals.dropna(subset=['granted'], inplace=True)
appeals['granted'] = appeals['granted'].astype(int)
print("After dropping appeals with neutral outcomes, {} rows remain".format(len(appeals)))

# summarize appeal outcomes
total_appeals = len(appeals)
successful_appeals = appeals['granted'].sum()
successful_appeals_pct = float(successful_appeals) / total_appeals
print("Of the {} appeals, {} ({:.1f}%) were successful.".format(total_appeals, successful_appeals,
                                                                successful_appeals_pct * 100))

print("***************************************")
print("Additional Feature Engineering")
print("***************************************")


def get_feature_values_to_retain(df, feature_name, min_samples):
    """ Returns a list of feature values that meet min_samples """
    distinct_values = df[feature_name].value_counts()
    retain_values = distinct_values[distinct_values >= min_samples].index.tolist()
    print("{} distinct values of {} will be retained as unique values, remaining {} will be grouped as other.".format(
        len(retain_values), feature_name, len(distinct_values) - len(retain_values)))
    return retain_values


# Group nationalities and judges with few samples
# apply to judges
ij_code_to_retain = get_feature_values_to_retain(appeals, feature_name='ij_code', min_samples=50)
appeals['ij_code_grouped'] = np.where(appeals['ij_code'].isin(ij_code_to_retain), appeals['ij_code'], 'other')
non_appeals['ij_code_grouped'] = np.where(non_appeals['ij_code'].isin(ij_code_to_retain), non_appeals['ij_code'],
                                          'other')

# apply to nationalities
nat_string_to_retain = get_feature_values_to_retain(appeals, feature_name='nat_string', min_samples=50)
appeals['nat_grouped'] = np.where(appeals['nat_string'].isin(nat_string_to_retain), appeals['nat_string'], 'other')
non_appeals['nat_grouped'] = np.where(non_appeals['nat_string'].isin(nat_string_to_retain), non_appeals['nat_string'],
                                      'other')

# apply to judge-nationalities
ij_code_nat_to_retain = get_feature_values_to_retain(appeals, feature_name='ij_code_nat', min_samples=50)
appeals['ij_code_nat_grouped'] = np.where(appeals['ij_code_nat'].isin(ij_code_nat_to_retain), appeals['ij_code_nat'],
                                          'other')
non_appeals['ij_code_nat_grouped'] = np.where(non_appeals['ij_code_nat'].isin(ij_code_nat_to_retain),
                                              non_appeals['ij_code_nat'], 'other')

# apply to lang
lang_to_retain = get_feature_values_to_retain(appeals, feature_name='lang', min_samples=50)
appeals['lang_grouped'] = np.where(appeals['lang'].isin(lang_to_retain), appeals['lang'], 'other')
non_appeals['lang_grouped'] = np.where(non_appeals['lang'].isin(lang_to_retain), non_appeals['lang'], 'other')

print("***************************************")
print("Additional Feature Engineering: judges")
print("***************************************")

"""
As proxied by two variables:  
- years_since_appointed = YEAR(Original proceeding decision) - YEAR(Judge Appointment) 
- years_since_law_school = YEAR(Original proceeding decision) - YEAR(Law School) 
"""


def get_time_delta(df, before, after, default_value=-1):
    """ Computes difference between feature_year_before and feature_year_after,
        filling NaNs and negative values with -1 """
    try:
        if (df[before].dtype == 'float' or df[before].dtype == 'int') or (
                        df[after].dtype == 'float' or df[after].dtype == 'int'):
            time_delta = df[after] - df[before]
        elif df[before].dtype == 'datetime64[ns]' and df[after].dtype == 'datetime64[ns]':
            time_delta = (df[after] - df[before]).dt.days
        time_delta = np.where((time_delta < 0) | pd.isnull(time_delta), default_value, time_delta)
    except:
        raise ValueError("Please use same datatype for 'before' and 'after'.")

    return time_delta


# years since judge appointment
appeals['years_since_judge_appointment'] = get_time_delta(appeals, 'Year_Appointed_SLR', 'comp_year')
non_appeals['years_since_judge_appointment'] = get_time_delta(non_appeals, 'Year_Appointed_SLR', 'comp_year')

# years since law school
appeals['years_since_law_school'] = get_time_delta(appeals, 'Year_Law_school_SLR', 'comp_year')
non_appeals['years_since_law_school'] = get_time_delta(non_appeals, 'Year_Law_school_SLR', 'comp_year')

# Time Elapsed Between OSC vs. Input vs. Comp vs. Appeal dates
# osc is when charge is filed, input date is when proceeding began, and comp date is when decision/ruling was made
appeals['appeal_days_elapsed_since_comp_date'] = get_time_delta(appeals, "comp_dt", "datAppealFiled_dt")
appeals['comp_days_elasped_since_input_date'] = get_time_delta(appeals, "input_date", "comp_date")
appeals['input_days_elapsed_since_osc_date'] = get_time_delta(appeals, "osc_date", "input_date")
non_appeals['comp_days_elasped_since_input_date'] = get_time_delta(non_appeals, "input_date", "comp_date")
non_appeals['input_days_elapsed_since_osc_date'] = get_time_delta(non_appeals, "osc_date", "input_date")

# Since non-appeals don't have appeal dates, we assume they would have filed 28 days (median of appeals) after comp date
non_appeals['appeal_days_elapsed_since_comp_date'] = appeals['appeal_days_elapsed_since_comp_date'].median()
non_appeals['datAppealFiled_dt'] = non_appeals['comp_dt'] + pd.to_timedelta(
    non_appeals['appeal_days_elapsed_since_comp_date'], unit='D')
non_appeals['datAppealFiled_year'] = non_appeals['datAppealFiled_dt'].dt.year
non_appeals['datAppealFiled_month'] = non_appeals['datAppealFiled_dt'].dt.month
non_appeals['datAppealFiled_year_month'] = (non_appeals['datAppealFiled_year'] - 1970) + non_appeals[
    'datAppealFiled_month']

print("***************************************")
print("Additional Feature Engineering: ")
print("Hearing and location analysis")
print("***************************************")


def check_hearing_loc_match_base(row):
    """ Checks whether base and hearing location are the same, different city, or different state """
    if pd.isnull(row['base_city_state']) | pd.isnull(row['hearing_loc_state']):
        return 'missing_info'
    elif row['base_city_code'] == row['hearing_loc_code']:
        return 'same_city'
    elif row['base_city_state'] == row['hearing_loc_state']:
        return 'diff_city_same_state'
    else:
        return 'diff_state'


appeals['hearing_loc_match_base'] = appeals.apply(check_hearing_loc_match_base, axis=1)
non_appeals['hearing_loc_match_base'] = non_appeals.apply(check_hearing_loc_match_base, axis=1)
appeals['hearing_loc_match_base'].value_counts()

print("***************************************")
print("Additional Feature Engineering: ")
print("Average Appeal Grant Rate")
print("***************************************")


def break_into_chunks(data, dimension, max_chunk):
    """ Returns a dictionary of lists to instruct breaking up dataset into suitable chunks,
        where resulting rows from self-join on ij_code does not exceed max_df_rows """
    dimensions = pd.DataFrame(data.groupby(dimension).size().sort_values(ascending=False))
    dimensions = dimensions.rename(columns={0: 'rows'}).reset_index()
    dimensions['self_join'] = dimensions['rows'] ** 2

    # stop if dimension has too many rows exceeding max_chunk
    exceeds_max_chunk = dimensions[dimensions['self_join'] > max_chunk]
    if len(exceeds_max_chunk) > 0:
        print(exceeds_max_chunk)
        raise ValueError('Dimension has too many rows!')
    else:
        pass

    dimensions['self_join_cumulative'] = dimensions['self_join'].cumsum()
    dimensions['chunk'] = np.floor(dimensions['self_join_cumulative'] / max_chunk).astype(int)
    chunk_assignments = dimensions.groupby('chunk')[dimension].apply(list).to_dict()
    print("Split {} labels in {} dimension into {} chunks...".format(len(dimensions), dimension,
                                                                     len(chunk_assignments)))
    return chunk_assignments


def compute_last_n_decisions_by_chunk(data_chunk, ref_chunk, dimension, last_n):
    """ Run compute for a given chunk of data """
    df = data_chunk.merge(ref_chunk, how='left', on=dimension)
    results = df[df['datBIADecision_dt'] < df['datAppealFiled_dt']].groupby('idnproceeding').apply(
        lambda f: f.head(last_n)['granted'].mean())
    return results


def compute_last_n_decisions(data, ref, dimension, new_feature_name, max_chunk=50000000, last_n=10):
    """ Unified method to compute last n decisions """

    # get chunk assignments
    chunk_assignments = break_into_chunks(data, dimension, max_chunk)

    # initialize empty list
    results = []
    start = time.time()

    # loop through each chunk
    for chunk, selected in chunk_assignments.iteritems():
        start_chunk = time.time()
        data_variables = ['idnproceeding', 'datAppealFiled_dt'] + [dimension]
        ref_variables = ['datBIADecision_dt', 'granted'] + [dimension]
        data_chunk = data[data[dimension].isin(selected)][data_variables]
        ref_chunk = ref[ref[dimension].isin(selected)][ref_variables].sort_values(
            by=[dimension] + ['datBIADecision_dt'], ascending=[True, False])
        result = compute_last_n_decisions_by_chunk(data_chunk, ref_chunk, dimension, last_n)
        results.append(result)
        print("Chunk {} completed in {} seconds".format(chunk, time.time() - start_chunk))

    print("DONE: Last {} decisions computed for {} dimension in {} seconds".format(last_n, dimension,
                                                                                   time.time() - start))

    return pd.DataFrame(pd.concat(results), columns=[new_feature_name])


def add_last_n_decisions(data, ref, dimension, new_feature_name, last_n=10, max_chunk=50000000):
    """ Takes full dataframe, adds last n decisions as a new column, returns new df """
    last_n_grant_rate = compute_last_n_decisions(data, ref, dimension, new_feature_name, max_chunk, last_n)
    df = data.merge(last_n_grant_rate, how='left', left_on='idnproceeding', right_index=True)
    return df


# last 10 by judge, for appeals
appeals = add_last_n_decisions(data=appeals, ref=appeals, dimension='ij_code_grouped',
                               new_feature_name='last_10_appeal_grant_by_judge', last_n=10, max_chunk=50000000)

# last 10 by judge, for non-appeals
non_appeals = add_last_n_decisions(data=non_appeals, ref=appeals, dimension='ij_code_grouped',
                                   new_feature_name='last_10_appeal_grant_by_judge', last_n=10, max_chunk=50000000)

# last 10 by judge+nat, for appeals
appeals = add_last_n_decisions(data=appeals, ref=appeals, dimension='ij_code_nat',
                               new_feature_name='last_10_appeal_grant_by_judge_nat', last_n=10, max_chunk=50000000)

# last 10 by judge+nat, for non-appeals
non_appeals = add_last_n_decisions(data=non_appeals, ref=appeals, dimension='ij_code_nat',
                                   new_feature_name='last_10_appeal_grant_by_judge_nat', last_n=10, max_chunk=50000000)

print("***************************************")
print("Outputting processed dataset to csv,")
print("pickle, and dta")
print("***************************************")
# ID features
id_features = ['idncase', 'idnproceeding', 'idnAppeal']

# Respondent features
respondent_features = ['nat_grouped', 'lang_grouped']

# Judge features
judge_features = ['ij_code_grouped', 'Male_judge', 'Year_Appointed_SLR', 'Year_College_SLR', 'Year_Law_school_SLR',
                  'Government_Years_SLR', 'Govt_nonINS_SLR', 'INS_Years_SLR', 'Military_Years_SLR', 'NGO_Years_SLR',
                  'Privateprac_Years_SLR', 'Academia_Years_SLR', 'judge_missing_bio',
                  'years_since_judge_appointment', 'years_since_law_school',
                  'last_10_appeal_grant_by_judge', 'last_10_appeal_grant_by_judge_nat']

# Proceeding features
proceeding_features = ['lawyer', 'defensive', 'affirmative', 'oral', 'written',
                       'case_type_string', 'original_dec_string']

# Appeal features
appeal_features = ['strCustody', 'strProbono']

# Location features
location_features = ['base_city_code', 'hearing_loc_match_base']

# Time features
time_features = ['datAppealFiled_year', 'datAppealFiled_year_month', 'comp_year', 'comp_year_month',
                 'comp_days_elasped_since_input_date', 'input_days_elapsed_since_osc_date',
                 'appeal_days_elapsed_since_comp_date',
                 'datBIADecision_dt', 'datAppealFiled_dt', 'comp_dt']  # timestamp used for analysis only

# Features to keep
features_to_keep = id_features + respondent_features + judge_features + proceeding_features \
                   + appeal_features + location_features + time_features
print(features_to_keep)

# output appeals dataset to csv
appeals_fp = os.path.join(DATAFOLDER, 'data_for_model/appeals_data_final.csv')
appeals_final = appeals[features_to_keep + ['granted']]
appeals_final.to_csv(appeals_fp, encoding='utf-8', index=False)
appeals_final.info()

# output non-appeals dataset to csv
non_appeals_fp = os.path.join(DATAFOLDER, 'data_for_model/non_appeals_data_final.csv')
non_appeals_final = non_appeals[features_to_keep]
non_appeals_final.to_csv(non_appeals_fp, encoding='utf-8', index=False)
non_appeals_final.info()

# also save as pkl
appeals_pkl_fp = os.path.join(DATAFOLDER, 'data_for_model/appeals_data_final.pkl')
appeals_final.to_pickle(appeals_pkl_fp)
non_appeals_pkl_fp = os.path.join(DATAFOLDER, 'data_for_model/non_appeals_data_final.pkl')
non_appeals_final.to_pickle(non_appeals_pkl_fp)

# save as .dta
appeals_dta_fp = os.path.join(DATAFOLDER, 'data_for_model/appeals_data_final.dta')
appeals_final.to_stata(appeals_dta_fp)
non_appeals_dta_fp = os.path.join(DATAFOLDER, 'data_for_model/non_appeals_data_final.dta')
non_appeals_final.to_stata(non_appeals_dta_fp)
