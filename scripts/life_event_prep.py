"""
Sample Materials, provided under license.
Licensed Materials - Property of IBM
© Copyright IBM Corp. 2019. All Rights Reserved.
US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
"""

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import sys
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import json
import os
from sklearn.externals import joblib

np.random.seed(42)

class LifeEventPrep():

    def __init__(self, target_event_type_ids, train_or_score='train', training_start_date="2010-01-01", training_end_date="2017-08-01", forecast_horizon=3,
                 observation_window=4, scoring_end_date=datetime.datetime.today(), life_event_minimum_target_count=100, norepeat_months=4, cols_to_drop=['CUSTOMER_ID']):

        self.train_or_score = train_or_score
        self.target_event_type_ids = target_event_type_ids
        self.forecast_horizon = forecast_horizon
        self.observation_window = observation_window
        self.training_start_date = training_start_date
        self.training_end_date = training_end_date
        self.scoring_end_date = scoring_end_date
        self.life_event_minimum_target_count = life_event_minimum_target_count
        self.norepeat_months = norepeat_months
        self.cols_to_drop = cols_to_drop
        self.latency_start = 1
        self.perc_positive_cutoff = 1.0

        self.project_path = "/user-home/" + os.environ.get("DSX_USER_ID", "990") \
                            + "/DSX_Projects/" + os.environ.get("DSX_PROJECT_NAME") + "/"

        # if a string for a particular date for end of scoring(obs month) is passed, convert to datetime
        if self.train_or_score == 'score':
            if isinstance(self.scoring_end_date, str):
                self.scoring_end_date = datetime.datetime.strptime(self.scoring_end_date, '%Y-%m-%d')
                
                
        if self.train_or_score == 'train':
        # create a dictionary with all values for user inputs. We will save this out and use it for scoring
        # to ensure that the user inputs are consistent across train and score notebooks
        # exclude variables that won't be used for scoring
          user_inputs_dict = { 'target_event_type_ids' : target_event_type_ids, 'forecast_horizon' : forecast_horizon,
              'observation_window' : observation_window, 'life_event_minimum_target_count' : life_event_minimum_target_count, 'norepeat_months' : norepeat_months}
            
          # use joblib to save the dictionary out to file
          joblib.dump(user_inputs_dict, os.environ.get("DSX_PROJECT_DIR") + '/datasets/training_user_inputs.joblib')

    # Some functions to handle adding, subtracting, randomly selecting dates that are in YYYYMM format
    def udf_n_months(self, dateMin, dateMax):
        month_dif = (relativedelta(datetime.datetime.strptime(str(dateMax), '%Y%m'),
                        datetime.datetime.strptime(str(dateMin), '%Y%m')).months +
                    relativedelta(datetime.datetime.strptime(str(dateMax), '%Y%m'),
                        datetime.datetime.strptime(str(dateMin), '%Y%m')).years * 12)
        return month_dif

    # function returns a random date in [dateMin, dateMax]
    def udf_rand_date_in_range(self, dateMin, dateMax, randnumber):
        rand_month_in_range = (datetime.datetime.strptime(str(dateMin), '%Y%m') +
                                        (relativedelta(months=int(np.floor(randnumber *(relativedelta(
                                                        datetime.datetime.strptime(str(dateMax), '%Y%m'),
                                                        datetime.datetime.strptime(str(dateMin), '%Y%m')).months +
                                                    relativedelta(
                                                        datetime.datetime.strptime(str(dateMax), '%Y%m'),
                                                        datetime.datetime.strptime(str(dateMin),'%Y%m')).years * 12 +1))))))

        return rand_month_in_range.strftime('%Y%m')

    # function to add a specified number of months to a date in YYYYMM format
    def udf_add_months(self, YEAR_MONTH, months_to_add):
        new_date = int((datetime.datetime.strptime(str(YEAR_MONTH), '%Y%m') + relativedelta(months=months_to_add)).strftime('%Y%m'))
        return new_date

    # selects the observation month
    def udf_sub_rand_latency(self, YEAR_MONTH, randnumber, latency_start, forecast_horizon):
        rand_obs_month = (datetime.datetime.strptime(str(YEAR_MONTH), '%Y%m')
                            - relativedelta(months=latency_start + int(randnumber * (forecast_horizon - latency_start + 1)))).strftime('%Y%m')
        return rand_obs_month

    # Prep functions
    def prepare_single_event_type(self, target_event_type_id, events, cym_cnt, train_or_score):
        # ensure the dataframe has data in it - can be empty if scoring low number of cases and they were filtered out based on dates
        if cym_cnt.shape[0] > 0:
            # This function preps the data for one specified target event
            # create a temporary target variable, which is 1 if the specific life event passed to the function
            # happened in that month, otherwise 0
            print('\nPrepping data for ' + target_event_type_id)
            target_col = 'E_' + target_event_type_id
            # make sure that the target event exists as a column in the dataset, if not, add it with all 0's (should only really happen when scoring)
            if target_col not in list(cym_cnt.columns):
                cym_cnt[target_col] = 0

            cym_cnt['TARGET'] = 0
            cym_cnt.loc[cym_cnt[target_col]>0, 'TARGET'] = 1

            # Call the select_customer_observation_month function - returns 1 record per customer
            # with the target variable, the observation month, the start month of the observation period
            # and the end month of the forecast horizon month
            df_cust_target = self.select_customer_observation_month(cym_cnt, train_or_score)

            # filter out customers who have no event data in the observation window
            # join the target df with cym_cnt, the df with customer and YYYYMM and column per event
            print('Number of customers before removing those with no event data in observation window : ' + str(df_cust_target.shape[0]))
            if train_or_score=='train':
                print('Number of customers target=1 before above filtering : ' + str(df_cust_target[df_cust_target['TARGET']==1].shape[0]))

            # remove the temp target var in cym_cnt as we now have our final target
            cym_cnt.drop('TARGET', axis=1, inplace=True)
            df_cust_target_cym_cnt = cym_cnt.merge(df_cust_target, on='CUSTOMER_ID', how='inner')
            df_cust_target_cym_cnt = df_cust_target_cym_cnt[(df_cust_target_cym_cnt['YEAR_MONTH'].astype(int)>=df_cust_target_cym_cnt['OBS_MONTH_MIN_OW'].astype(int))
                                    & (df_cust_target_cym_cnt['YEAR_MONTH'].astype(int)<=df_cust_target_cym_cnt['OBS_MONTH'].astype(int))]

            # update the target df to include only those customers who have had events in the observation window
            df_cust_target = df_cust_target[df_cust_target['CUSTOMER_ID'].isin(list(df_cust_target_cym_cnt['CUSTOMER_ID'].unique()))]

            print('Number of customers after removing those with no event data in observation window : ' + str(df_cust_target.shape[0]))
            if train_or_score=='train':
                print('Number of customers target=1 after above filtering : ' + str(df_cust_target[df_cust_target['TARGET']==1].shape[0]))

            # not all customers have had an event in their observation month
            # Therefore they wouldn't have a record in the data for the observation month
            # We add the record in and fill all events with 0
            # I think this helps for later in the code
            # Get the observation month for each customer, join to the wide df (cym_cnt) on customer and YYYYMM,
            # Use a right outer join so if the observation month isn't in the wide df it will be included after join

            df_temp = cym_cnt.merge(df_cust_target[['CUSTOMER_ID', 'OBS_MONTH']], left_on=['CUSTOMER_ID', 'YEAR_MONTH'],
                                            right_on=['CUSTOMER_ID', 'OBS_MONTH'], how='right')
            # only take where the result is null, ie. where the observerd month wasn't in the wide (cym_cnt) table
            df_temp = df_temp[df_temp['YEAR_MONTH'].isnull()]
            # update YEAR_MONTH to be the OBS_MONTH and drop OBS_MONTH, fill na's with 0
            df_temp['YEAR_MONTH'] = df_temp['OBS_MONTH']
            df_temp.drop('OBS_MONTH', axis=1, inplace=True)
            df_temp.fillna(0, inplace=True)

            # join back to target df so we can add the relevant columns from that df
            # then append onto the df_cust_target_cym_cnt df
            # we then just have a df with customer, YYYYMM and column for each event,
            # but always including a record for the observation month for a customer
            df_temp = df_temp.merge(df_cust_target, on='CUSTOMER_ID', how='inner')
            df_cust_target_cym_cnt = pd.concat([df_cust_target_cym_cnt, df_temp], sort=False)

            # more filtering to remove edge cases
            # for scoring, remove customers who have experienced the life event in the previous norepeat_months months
            # where start of the observation window is before the start of our data, remove the records
            # where the end of the forecase period is after the end of our data, remove the records
            YM_min = df_cust_target_cym_cnt['YEAR_MONTH'].min()
            YM_max = df_cust_target_cym_cnt['YEAR_MONTH'].max()

            #print('Number of customers before filtering : ' + str(df_cust_target_cym_cnt['CUSTOMER_ID'].nunique()))
            #print('Number of customers target=1 before filtering : ' + str(df_cust_target_cym_cnt[df_cust_target_cym_cnt['TARGET']==1]['CUSTOMER_ID'].nunique()))

            df_cust_target_cym_cnt = df_cust_target_cym_cnt[df_cust_target_cym_cnt['OBS_MONTH_MIN_OW']>=YM_min]
            if train_or_score == 'train':
                df_cust_target_cym_cnt = df_cust_target_cym_cnt[df_cust_target_cym_cnt['OBS_MONTH_PLS_LATEND']<=YM_max]
            elif train_or_score == 'score':
                norepeat_months = self.norepeat_months
                # we don't want to score customers who have experienced the life event in the previous norepeat_months
                start_norepeat_period = self.udf_add_months(YM_max, 1-norepeat_months)

                customers_norepeat = df_cust_target_cym_cnt[(df_cust_target_cym_cnt['TARGET']>0) &
                                                (df_cust_target_cym_cnt['TARGET_MONTH']<start_norepeat_period)]

                customers_norepeat = pd.DataFrame(customers_norepeat['CUSTOMER_ID'].drop_duplicates())
                customers_norepeat['LIFE_EVENT_B4_NOREP_PERIOD'] = 1

                df_cust_target_cym_cnt = df_cust_target_cym_cnt.merge(customers_norepeat, on='CUSTOMER_ID', how='left')

                # Keep records where the target is 0 (haven't experienced the life event),
                # or target is 1 and life_event_b4_norep_period is 1 (the customer experienced the life event but it was
                # more than norepeat_months ago)
                df_cust_target_cym_cnt = df_cust_target_cym_cnt[(df_cust_target_cym_cnt['TARGET']==0) | ((df_cust_target_cym_cnt['TARGET']==1) &
                                                                        (df_cust_target_cym_cnt['LIFE_EVENT_B4_NOREP_PERIOD']==1))]

                df_cust_target_cym_cnt.drop(['LIFE_EVENT_B4_NOREP_PERIOD'], axis=1, inplace=True)
                if df_cust_target_cym_cnt.shape[0] == 0:
                    print('Note: All customers filtered out as they experienced the life event within ' +
                        str(norepeat_months) + ' months (norepeat_months) of the observation date', file=sys.stderr)


            #print('Number of customers after filtering : ' + str(df_cust_target_cym_cnt['CUSTOMER_ID'].nunique()))
            #print('Number of customers target=1 after filtering : ' + str(df_cust_target_cym_cnt[df_cust_target_cym_cnt['TARGET']==1]['CUSTOMER_ID'].nunique()))

            # get data into AMT format, one line of data per customer
            # we will create variables that are a count of each event per customer over their observation window (end with '_OW')
            # We also create variables for the count of each event in the actual observation month

            # remove columns we don't need anymore
            df_cust_target_cym_cnt.drop(['TARGET_MONTH', 'OBS_MONTH_MIN_OW', 'OBS_MONTH_PLS_LATEND'], axis=1, inplace=True)

            # count the number of occurences of each event over the observation window
            # drop target as it is summed up, correct target is later
            df_per_cust_ow = df_cust_target_cym_cnt.groupby(['CUSTOMER_ID', 'OBS_MONTH']).sum().reset_index()
            df_per_cust_ow.drop(['YEAR_MONTH', 'TARGET'], axis=1, inplace=True)
            # add a variable for total number of events over observarion window

            for col in df_per_cust_ow.columns:
                if col.startswith('E_'):
                    new_col_name = col + '_OW'
                    df_per_cust_ow.rename(columns={col:new_col_name}, inplace=True)

            # get the number of occurences of each event in the observation window
            df_per_cust_om = df_cust_target_cym_cnt[df_cust_target_cym_cnt['OBS_MONTH']==df_cust_target_cym_cnt['YEAR_MONTH']].copy()
            df_per_cust_om.drop('YEAR_MONTH', axis=1, inplace=True)

            for col in df_per_cust_om.columns:
                if col.startswith('E_'):
                    new_col_name = col + '_OM'
                    df_per_cust_om.rename(columns={col:new_col_name}, inplace=True)

            df_per_cust = df_per_cust_ow.merge(df_per_cust_om, on=['CUSTOMER_ID', 'OBS_MONTH'], how='inner')

            # add a variable for observation month.
            df_per_cust['MONTH'] = df_per_cust['OBS_MONTH'].astype(str).str[4:].astype(int)

            # get total number of events per customers in observation window and in observation month
            events_ow_cols = list(df_per_cust.loc[:, df_per_cust.columns.str.endswith('_OW')].columns)
            events_om_cols = list(df_per_cust.loc[:, (~(df_per_cust.columns.str.endswith('_OW')) & (df_per_cust.columns.str.startswith('E_')))].columns)

            df_per_cust['TOT_NB_OF_EVENTS_OW'] = df_per_cust[events_ow_cols].sum(axis=1)
            df_per_cust['TOT_NB_OF_EVENTS_OM'] = df_per_cust[events_om_cols].sum(axis=1)

            if train_or_score == 'train':
                # cleaning - move target variable to end
                cols = list(df_per_cust)
                cols.insert(len(cols), cols.pop(cols.index('TARGET')))
                df_per_cust = df_per_cust.loc[:, cols]
            elif train_or_score == 'score':
                df_per_cust.drop('TARGET', axis=1, inplace=True)

            return df_per_cust

        else:
            print('Error: No customers were passed to the function. Stopping.', file=sys.stderr)
            sys.exit()

    def select_customer_observation_month(self, cym_cnt,  train_or_score):
        # Creates the observation month for each customer and the target variable.
        # Also works out first month in observation window and end of forecast horizon period
        # First creates a target month for each customer, TARGET_MONTH, the month that the life event occurs
        # for the customer. If a customer hasn't experienced a life event, they are given a random TARGET_MONTH,
        # which is a month between the first event and last event month for that customer
        # For scoring, the observation month is the 'scoring_end_date' variable

        observation_window = self.observation_window
        forecast_horizon = self.forecast_horizon
        latency_start = self.latency_start

        print('Using observation_window = ' + str(observation_window))
        print('Using forecast_horizon = ' + str(forecast_horizon))

        # create df with just customer_id, year_month and target(as created in prepare_single_event_type function)
        # df_cust_month_target was cym_tgt in original code
        df_cust_month_target = cym_cnt[['CUSTOMER_ID', 'YEAR_MONTH', 'TARGET']]

        if train_or_score == 'train':
            # Directly from codebase:
            ######################################
            # Definition of the TARGET_MONTH:
            # occurence of target over histoical months
            # C1: 0 ------------------------- 0
            # C2: ----------------- 0 1 0 -----
            # C3: ------- 0 1 0 --- 0 1 0 -----
            #
            # C1 has no occurence of the event,
            # C2 has exactly one
            # C3 had two occurences

            # C1. For customers who didn't experience a life event give them a random TARGET_MONTH
            # within the period that they were active
            # Note this can select a target_month that is the first month a customer is seen
            # that customer will have no historical event data and will be removed later in the code
            df_negatives = df_cust_month_target.groupby(['CUSTOMER_ID']).agg({'YEAR_MONTH':['min', 'max'], 'TARGET': 'sum'}).reset_index()
            df_negatives.columns = df_negatives.columns.get_level_values(1)
            df_negatives.columns = ['CUSTOMER_ID', 'CUST_YM_MIN', 'CUST_YM_MAX', 'CUST_TARGET']
            df_negatives = df_negatives[df_negatives['CUST_TARGET']==0]
            df_negatives['RAND1'] = np.random.rand(df_negatives.shape[0])
            df_negatives['TARGET_MONTH'] = df_negatives.apply(lambda x: self.udf_rand_date_in_range(int(x['CUST_YM_MIN']), int(x['CUST_YM_MAX']), x['RAND1']), axis=1)
            df_negatives = df_negatives[['CUSTOMER_ID', 'TARGET_MONTH']]
            df_negatives['TARGET'] = 0

            # C2. Customers who experienced exactly 1 life event
            # assign a specified % as positive examples
            # should this % parameter be configurable?
            # remaing are set to negative, selecting a random month at least norepeat_months after the event

            perc_positive_cutoff = self.perc_positive_cutoff

            # Create a df with target month for each customer and a random number column which is used
            # to specify if the record should be used as a positive or negative example
            df_cust_target_one_occ = df_cust_month_target[df_cust_month_target['TARGET']>0].groupby('CUSTOMER_ID').agg({'YEAR_MONTH':'min', 'TARGET': 'sum'}).reset_index()
            df_cust_target_one_occ = df_cust_target_one_occ[df_cust_target_one_occ['TARGET']==1]
            df_cust_target_one_occ.rename(columns={'YEAR_MONTH':'TARGET_MONTH', 'TARGET':'TARGET_COUNT'}, inplace=True)
            df_cust_target_one_occ['RAND1'] = np.random.rand(df_cust_target_one_occ.shape[0])

            # Take all records with random number less than cutoff as positive examples
            df_cust_target_one_occ_pos = df_cust_target_one_occ[df_cust_target_one_occ['RAND1']<=perc_positive_cutoff][['CUSTOMER_ID', 'TARGET_MONTH']]
            df_cust_target_one_occ_pos['TARGET'] = 1

            # All records greater than the cutoff are negative examples
            # For each customer, find the starting point for their TARGET_MONTH
            # This has to be between norepeat_months after the event and the date of their last event
            # if the cutoff is set to 1 it means that we don't set any of these records to 0

            if perc_positive_cutoff < 1.0:
                df_cust_target_one_occ_neg = df_cust_target_one_occ[df_cust_target_one_occ['RAND1']>perc_positive_cutoff][['CUSTOMER_ID', 'TARGET_MONTH']]
                # specify a new (temp) TARGET_MONTH that is norepeat_months after the event occurred
                df_cust_target_one_occ_neg['TARGET_MONTH'] = df_cust_target_one_occ_neg.apply(lambda x: self.udf_add_months(int(x['TARGET_MONTH']), self.norepeat_months), axis=1)
                # Select a random month between the new TARGET_MONTH and the last time the customer is seen
                # join back to df_cust_month_target which has a record for every customer and month
                df_cust_target_one_occ_neg = df_cust_target_one_occ_neg.merge(df_cust_month_target, on='CUSTOMER_ID', how='inner')
                # filter to include only months >= the new target month
                df_cust_target_one_occ_neg[df_cust_target_one_occ_neg['YEAR_MONTH']>=df_cust_target_one_occ_neg['TARGET_MONTH']]
                # I changed this, original called for a random month between first event after new target_month and last event
                # I'm changing to say the customer can have a random target month between new target month and last event
                df_cust_target_one_occ_neg = df_cust_target_one_occ_neg.groupby(['CUSTOMER_ID', 'TARGET_MONTH'])['YEAR_MONTH'].max().reset_index()
                df_cust_target_one_occ_neg.rename(columns={'TARGET_MONTH':'CUST_YM_MIN', 'YEAR_MONTH':'CUST_YM_MAX'}, inplace=True)
                df_cust_target_one_occ_neg['RAND2'] = np.random.rand(df_cust_target_one_occ_neg.shape[0])
                # Call the function to select a random target month between TARGET_MONTH and last event month
                df_cust_target_one_occ_neg['TARGET_MONTH'] = df_cust_target_one_occ_neg.apply(lambda x: self.udf_rand_date_in_range(int(x['CUST_YM_MIN']), int(x['CUST_YM_MAX']), x['RAND2']), axis=1)
                # select relevant columns and set the target value to 0
                df_cust_target_one_occ_neg = df_cust_target_one_occ_neg[['CUSTOMER_ID', 'TARGET_MONTH']]
                df_cust_target_one_occ_neg['TARGET'] = 0

            # C3. Customers who experienced the life event multiple times
            # We just take the first time they experienced the event as the TARGET_MONTH

            # filter to only include months where target=1
            df_cust_target_multi_occ = df_cust_month_target[df_cust_month_target['TARGET']>=1]
            # Get the min of YEAR_MONTH per customer to find month of occurence of first life event
            # Sum up the target so we can filter to inlcude only customers who have had multiple life events
            df_cust_target_multi_occ = df_cust_target_multi_occ.groupby('CUSTOMER_ID').agg({'YEAR_MONTH':'min', 'TARGET':'sum'}).reset_index()
            df_cust_target_multi_occ.rename(columns={'YEAR_MONTH':'TARGET_MONTH'}, inplace=True)
            df_cust_target_multi_occ = df_cust_target_multi_occ[df_cust_target_multi_occ['TARGET']>1]
            # Filter to only include columns we need and set target to 1
            df_cust_target_multi_occ = df_cust_target_multi_occ[['CUSTOMER_ID', 'TARGET_MONTH']]
            df_cust_target_multi_occ['TARGET'] = 1

            print('Training data has #Target> 1 customers: ' + str(df_cust_target_multi_occ.shape[0]))
            print('Training data has #Target==1 customers: ' + str(df_cust_target_one_occ.shape[0]))
            print('   Of those, we set ' + str(df_cust_target_one_occ_pos.shape[0]) + ' to positive')
            if perc_positive_cutoff < 1.0:
                print('   and we set ' + str(df_cust_target_one_occ_neg.shape[0]) + ' to negative')
            else:
                print('   and we set 0 to negative')
            print('Training data has #Target==0 customers: ' + str(df_negatives.shape[0]))

            # this was cdates variable in original
            if perc_positive_cutoff < 1.0:
                df_cust_target = pd.concat([df_negatives, df_cust_target_one_occ_neg, df_cust_target_one_occ_pos, df_cust_target_multi_occ])
            else:
                df_cust_target = pd.concat([df_negatives, df_cust_target_one_occ_pos, df_cust_target_multi_occ])
            print('Number of records : ' + str(df_cust_target.shape[0]))
            print('Number of unique customers : ' + str(df_cust_target['CUSTOMER_ID'].nunique()))

            if df_cust_target['CUSTOMER_ID'].nunique() != df_cust_target.shape[0]:
                print('Something went wrong. We have more than 1 row per customer')

            # I haven't included hidden feature about boosting minority class

            # select the observation month for each customer
            # the month must be within the forecast_horizon of the target_month
            # For example, if the life event occurred in 201809, the observation month must be between
            # 201806 and 201809 if the forecast_horizon is 3 months
            df_cust_target['RAND2'] = np.random.rand(df_cust_target.shape[0])
            df_cust_target['OBS_MONTH'] = df_cust_target.apply(lambda x: self.udf_sub_rand_latency(int(x['TARGET_MONTH']), x['RAND2'], latency_start, forecast_horizon), axis=1)

            # Now that we have the observation month we get the first month of our observation window (OBS_MONTH_MIN_OW)
            # Events which occurred over the observation window will be used as variables in AMT
            # Note that the observation month is included in the observation window
            df_cust_target['OBS_MONTH_MIN_OW'] = df_cust_target.apply(lambda x: self.udf_add_months(int(x['OBS_MONTH']), (1-observation_window)), axis=1)

            # We also calculate the end month in the forecasting period (OBS_MONTH_PLS_LATEND)
            df_cust_target['OBS_MONTH_PLS_LATEND'] = df_cust_target.apply(lambda x: self.udf_add_months(int(x['OBS_MONTH']), forecast_horizon), axis=1)

            df_cust_target.drop('RAND2', axis=1, inplace=True)

            # set the months to ints instead of objects
            df_cust_target['OBS_MONTH'] = df_cust_target['OBS_MONTH'].astype(int)
            df_cust_target['TARGET_MONTH'] = df_cust_target['TARGET_MONTH'].astype(int)

        elif train_or_score =='score':

            df_cust_target = df_cust_month_target.groupby('CUSTOMER_ID')['TARGET'].max().reset_index()
            df_cust_target['OBS_MONTH'] = pd.to_datetime(self.scoring_end_date.date())
            df_cust_target['OBS_MONTH'] = df_cust_target['OBS_MONTH'].dt.strftime('%Y%m').astype(int)
            df_cust_target['OBS_MONTH_MIN_OW'] = df_cust_target.apply(lambda x: self.udf_add_months(int(x['OBS_MONTH']), (1-observation_window)), axis=1)
            df_cust_target['OBS_MONTH_PLS_LATEND'] = df_cust_target.apply(lambda x: self.udf_add_months(int(x['OBS_MONTH']), forecast_horizon), axis=1)

            # for those customers who have experienced the life event, we want to know when they last experienced it
            df_month_last_lfe_event = df_cust_month_target[df_cust_month_target['TARGET']>0].groupby('CUSTOMER_ID')['YEAR_MONTH'].max().reset_index()
            df_month_last_lfe_event.rename(columns={'YEAR_MONTH':'TARGET_MONTH'}, inplace=True)
            df_cust_target = df_cust_target.merge(df_month_last_lfe_event, on='CUSTOMER_ID', how='left')
            df_cust_target['TARGET_MONTH'] = df_cust_target['TARGET_MONTH'].fillna(0)
            df_cust_target['TARGET_MONTH'] = df_cust_target['TARGET_MONTH'].astype(int)

        return df_cust_target

    def prep_data(self, df_raw, train_or_score):
        # just in case any caps are used
        train_or_score = train_or_score.lower()
        # hidden inputs
        latency_start = self.latency_start # don't like how this behaves right now - don't include as input?

        print('Before removing dates that are not in training period : ' + str(df_raw.shape))
        # remove any dates that are not in our training period
        if train_or_score == 'train':
            df_raw = df_raw[(df_raw['EVENT_DATE']>=datetime.datetime.strptime(self.training_start_date, '%Y-%m-%d'))
                    & (df_raw['EVENT_DATE']<=datetime.datetime.strptime(self.training_end_date, '%Y-%m-%d'))]
        else:
            # otherwise use same start period but all data to end of scoring period
            df_raw = df_raw[(df_raw['EVENT_DATE']>=datetime.datetime.strptime(self.training_start_date, '%Y-%m-%d'))
                    & (df_raw['EVENT_DATE']<=self.scoring_end_date)]
        print('After removing dates that are not in training period : ' + str(df_raw.shape) + '\n')

        # create a df with 1 record per customer, get date of first and last event
        # filter to include only those who have enough months of data
        # enough months = (observation + forecast) for training data
        # enough months = observation window for scoring data
        # For scoring, if we haven't seen the customer in the observation window, we filter them out here

        print('Number of customers before checking for enough history : ' + str(df_raw['CUSTOMER_ID'].nunique()))
        if train_or_score == 'train':
            n_months = self.forecast_horizon + self.observation_window

            customers_with_enough_history = df_raw.groupby('CUSTOMER_ID')['EVENT_DATE'].agg([max, min]).reset_index()
            customers_with_enough_history.columns = ['CUSTOMER_ID', 'MAX_DATE', 'MIN_DATE']

            # Convert to yyyymm and add new column for number of months
            # filter to exclude customers who don't have enough months of data
            customers_with_enough_history['MAX_DATE'] = customers_with_enough_history['MAX_DATE'].dt.strftime('%Y%m').astype(int)
            customers_with_enough_history['MIN_DATE'] = customers_with_enough_history['MIN_DATE'].dt.strftime('%Y%m').astype(int)
            customers_with_enough_history['N_MONTHS'] = customers_with_enough_history.apply(lambda x: self.udf_n_months(x['MIN_DATE'], x['MAX_DATE']), axis=1)
            customers_with_enough_history = customers_with_enough_history[customers_with_enough_history['N_MONTHS']>n_months]

        elif train_or_score == 'score':
            n_months = self.observation_window

            customers_with_enough_history = df_raw.groupby('CUSTOMER_ID')['EVENT_DATE'].max().reset_index()
            customers_with_enough_history.columns = ['CUSTOMER_ID', 'MAX_DATE']
            # add a new column for effective date
            customers_with_enough_history['EFF_DATE_LATEST'] = pd.to_datetime(self.scoring_end_date.date())
            # Convert to yyyymm and add new column for number of months
            # filter to exclude customers who haven't had an event in the observation periods
            customers_with_enough_history['MAX_DATE'] = customers_with_enough_history['MAX_DATE'].dt.strftime('%Y%m').astype(int)
            customers_with_enough_history['EFF_DATE_LATEST'] = customers_with_enough_history['EFF_DATE_LATEST'].dt.strftime('%Y%m').astype(int)
            customers_with_enough_history['N_MONTHS'] = customers_with_enough_history.apply(lambda x: self.udf_n_months(x['MAX_DATE'], x['EFF_DATE_LATEST']), axis=1)
            customers_with_enough_history = customers_with_enough_history[customers_with_enough_history['N_MONTHS']<=n_months]
            if customers_with_enough_history.shape[0] == 0:
                print('Note: No customer for scoring had any event within the observation window and all have been filtered out')

        print('Number of customers after  checking for enough history : ' + str(customers_with_enough_history.shape[0]) + '\n')

        df_events = df_raw.merge(customers_with_enough_history, on='CUSTOMER_ID', how='inner')
        print('Total number of events in the data : ' + str(df_events.shape[0]) + '\n')
        # get a list of distinct events
        events = list(df_events['EVENT_TYPE_ID'].unique())

        # get a count of number of occurences of each event by customer and month (yyyymm)
        # pivot to give one record per customer and month (yyyymm) with each event having a column
        df_events['YEAR_MONTH'] = df_events['EVENT_DATE'].dt.strftime('%Y%m').astype(int)
        df_events = df_events.groupby(['CUSTOMER_ID', 'YEAR_MONTH', 'EVENT_TYPE_ID']).size().reset_index()
        df_events.rename(columns={0:'count'}, inplace=True)

        cym_cnt = pd.pivot_table(df_events, index=['CUSTOMER_ID', 'YEAR_MONTH'], columns='EVENT_TYPE_ID', values='count').reset_index()
        cym_cnt.fillna(0, inplace=True)

        if cym_cnt.shape[0] == 0:
            print('Note: All customers were filtered out\n')

        # check to make sure target events are in the events table
        # any target event that doesn't appear in the events table is removed
        # This should only be carried out for training
        if train_or_score == 'train':
            for target_event in self.target_event_type_ids:
                if target_event not in events:
                    self.target_event_type_ids.remove(target_event)
                    print(target_event + ' does not appear in events table and has been removed')

            if len(self.target_event_type_ids) == 0:
                print('Note: event_type_ids from target_event_type_ids not found in events table')

            # if there are less than the threshold number of customers associated with the target event, remove the event
            # get a count of number of unique customers associated with each target event
            # any below the threshold are removed from the target_event_type_ids list
            df_target_cust_count = pd.DataFrame(df_events[df_events['EVENT_TYPE_ID'].isin(self.target_event_type_ids)].groupby('EVENT_TYPE_ID')['CUSTOMER_ID'].nunique().reset_index())
            df_target_cust_count.rename(columns={'CUSTOMER_ID':'customer_count'}, inplace=True)
            events_below_threshold = list(df_target_cust_count[df_target_cust_count['customer_count']<self.life_event_minimum_target_count]['EVENT_TYPE_ID'])
            target_event_type_ids = [x for x in self.target_event_type_ids if x not in events_below_threshold]
            print('\n' + str(len(self.target_event_type_ids)) + ' Target ID(s) left after removing target events below threshold (' + str(self.life_event_minimum_target_count) + ' customers)')

        # rename event columns to include 'E_'
        for e in events:
            cym_cnt.rename(columns={e:'E_' + e}, inplace=True)

        result_map = {}
        for event_type_id in self.target_event_type_ids:
            #Call the prepare_single_event_type function
            result_map[event_type_id] = self.prepare_single_event_type(event_type_id, events, cym_cnt, train_or_score)

        for event_type_id in self.target_event_type_ids:
            # prep training data, remove columns where nulls make up over 10%
            # drop constant columns (eg all 0's)

            # drop obs_month column
            result_map[event_type_id].drop('OBS_MONTH', axis=1, inplace=True)

            if train_or_score == 'train':
                columns_required = ['CUSTOMER_ID', 'TARGET', 'MONTH']
                numeric_cols = []
                for col in result_map[event_type_id].columns:
                    if is_numeric_dtype(result_map[event_type_id][col].dtype):
                        numeric_cols.append(col)

                numeric_cols = set(numeric_cols) - set(columns_required)
                print(result_map[event_type_id].shape)
                # loop through columns and check for constants or missing vals
                for col in numeric_cols:
                    # drop cols where min=max ie constants
                    curr_col = result_map[event_type_id][col]
                    if curr_col.min() == curr_col.max():
                        result_map[event_type_id].drop(col, axis=1, inplace=True)
                    # drop column if it is 10% or more null values
                    elif (curr_col.isna().sum()/curr_col.shape[0]) > 0.1:
                        result_map[event_type_id].drop(col, axis=1, inplace=True)

            # Cleanup - make sure we have same columns and order for training and scoring

            # if training, use json to save out the columns that were used for training
            if train_or_score == 'train':
                with open(self.project_path + 'datasets/' + event_type_id + '_training_cols.json', 'w') as f:
                    json.dump(list(result_map[event_type_id].columns), f)
            # if we are scoring, we import the json file so we make sure all columns used for scoring
            # are in the training data
            elif train_or_score == 'score':
                with open(self.project_path + 'datasets/' + event_type_id + '_training_cols.json', 'rb') as f:
                    cols_used_for_training = json.load(f)

                # don't need to include target variable for scoring
                cols_used_for_training.remove('TARGET')

                # if a column does not exist in scoring but is in training, add the column to scoring dataset
                for col in cols_used_for_training:
                    if col not in list(result_map[event_type_id].columns):
                        result_map[event_type_id][col] = 0

                # if a column exists in scoring but not in training, delete it from scoring dataset
                for col in list(result_map[event_type_id].columns):
                    if col not in cols_used_for_training:
                        result_map[event_type_id].drop(col, axis=1, inplace=True)

                # make sure order of scoring columns is same as training dataset
                result_map[event_type_id] = result_map[event_type_id][cols_used_for_training]

            for col in self.cols_to_drop:
                result_map[event_type_id].drop(col, axis=1, inplace=True)

        return result_map
