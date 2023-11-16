import numpy as np
import pandas as pd
import os
import itertools

input_path_huber = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New attacked data duration 4Mths/huber_tau_2014_2Ms'
input_path_cauchy = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New attacked data duration 4Mths/Cauchy_tau_2014_2Ms'
input_path_tukey = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New attacked data duration 4Mths/Tukey_tau_2014_2Ms'

test_attack_path = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New attacked data duration 4Mths/RUCs_values_2016_2Ms'
output_path = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New attacked data duration 4Mths/track_FA_2014_2Ms_diff'

# constant values
DEL_AVG_ARRAY = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
RO_MAL_ARRAY = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


# defined function:
def testing_EFA(residual_frame, tao_max, tao_min):
    false_alarm = []
    for row in residual_frame.itertuples():
        if not (((getattr(row, "ratio2016") <= getattr(row, 'margin_high')) and
                 (getattr(row, 'ratio2016') >= getattr(row, 'margin_low')))):
            if float(getattr(row, "ruc2016")) != float(0):
                if ((float(getattr(row, "ruc2016")) > float(tao_max)) or (
                        float(getattr(row, "ruc2016")) < float(tao_min))):
                    false_alarm.append(row.day)
    return false_alarm


def testing_tau(residual_frame, tao_max, tao_min):
    first_detected_org = 0
    false_alarm_tier2_org = 0
    tier1_anomaly = 0
    tier2_for_org = 0
    false_alarm = []
    for row in residual_frame.itertuples():
        if not (((getattr(row, "ratio2016") <= getattr(row, 'margin_high')) and
                 (getattr(row, 'ratio2016') >= getattr(row, 'margin_low')))):
            tier1_anomaly = tier1_anomaly + 1
            if float(getattr(row, "ruc2016")) != float(0):
                if ((float(getattr(row, "ruc2016")) > float(tao_max)) or (
                        float(getattr(row, "ruc2016")) < float(tao_min))):
                    # print("day",getattr(row, "day"),' ',getattr(row,"ruc2016"),' ', max_threshold_org,' ', 
                    # min_threshold_org) 
                    if ((getattr(row, "day") >= int(91)) and (
                            getattr(row, "day") <= int(152))):  # 91 = attack start day 273 = attack end day
                        if int(tier2_for_org) == int(0):
                            first_detected_org = getattr(row, "day")
                        tier2_for_org = tier2_for_org + 1
                    else:
                        false_alarm_tier2_org = false_alarm_tier2_org + 1
                        false_alarm.append(row.day)
    return tier1_anomaly, tier2_for_org, first_detected_org, false_alarm


for del_val, romal_val in itertools.product(DEL_AVG_ARRAY, RO_MAL_ARRAY):
    for del_val_test in DEL_AVG_ARRAY:
        if del_val_test != del_val: # to calculate just different
        #if del_val_test == del_val: # to calculate same
            # for del_val in DEL_AVG_ARRAY:
            # for romal_val in RO_MAL_ARRAY:
            # Form the file name based on the values of del and romal
            file_name = f'tau_RUCs_attacked_data_del_{del_val}_romal_{romal_val}_type_ded_2014.csv'

            # Read the three DataFrames from the respective folders
            tau_frame_RA_T = pd.read_csv(os.path.join(input_path_tukey, file_name))
            tau_frame_RA_C = pd.read_csv(os.path.join(input_path_cauchy, file_name))
            tau_frame_RA_H = pd.read_csv(os.path.join(input_path_huber, file_name))

            # rename the columns
            tau_frame_RA_T.rename(columns={'optimal_tau_max': 'tau_max_T', 'optimal_tau_min': 'tau_min_T'},
                                  inplace=True)
            tau_frame_RA_C.rename(columns={'optimal_tau_max': 'tau_max_c', 'optimal_tau_min': 'tau_min_c'},
                                  inplace=True)
            tau_frame_RA_H.rename(columns={'optimal_tau_max': 'tau_max_h', 'optimal_tau_min': 'tau_min_h'},
                                  inplace=True)

            # merge the dataframes
            tau_frame_RA = pd.merge(tau_frame_RA_T, tau_frame_RA_C, on=['beta_n', 'beta_p'])
            tau_frame_RA = pd.merge(tau_frame_RA, tau_frame_RA_H, on=['beta_n', 'beta_p'])
            # attack in test is for 2 months: 4th_6th
            attack_start_date = 91
            attack_end_date = 152
            result = []

            for index, row in tau_frame_RA.iterrows():
                # Now check with the benign Test set residual
                # As a cross validation set we can consider 6 months of data
                # Form the file name based on the values of del and romal
                test_attack_file_name = f'RUCs_attacked_data_del_{del_val_test}_romal_{romal_val}_type_ded_2016.csv'

                # Test_RUC_TR{del_val}_TE_{del_val}_RO_{romal_val}dedM4M6.csv'

                test_residual_benign = pd.read_csv('Test_RUC_Benign.csv')
                # Read the test residual attack file using the test_attack_path
                test_residual_attack = pd.read_csv(os.path.join(test_attack_path, test_attack_file_name))

                cv_residual_benign = test_residual_benign[
                    (test_residual_benign['day'] >= 91) & (test_residual_benign['day'] <= 152)]
                cv_residual_attack = test_residual_attack[
                    (test_residual_attack['day'] >= 91) & (test_residual_attack['day'] <= 152)]
                # CAUCHY
                false_alarm_c = testing_EFA(cv_residual_benign, row.tau_max_c, row.tau_min_c)
                tier1_anomaly_c, tier2_for_org_c, first_detected_org_c, false_alarm_ca = testing_tau(cv_residual_attack,
                                                                                                     row.tau_max_c,
                                                                                                     row.tau_min_c)
                # Tukey
                false_alarm_T = testing_EFA(cv_residual_benign, row.tau_max_T, row.tau_min_T)
                tier1_anomaly_t, tier2_for_org_t, first_detected_org_t, false_alarm_ta = testing_tau(cv_residual_attack,
                                                                                                     row.tau_max_T,
                                                                                                     row.tau_min_T)
                # HUBER
                false_alarm_h = testing_EFA(cv_residual_benign, row.tau_max_h, row.tau_min_h)
                tier1_anomaly_h, tier2_for_org_h, first_detected_org_h, false_alarm_ha = testing_tau(cv_residual_attack,
                                                                                                     row.tau_max_h,
                                                                                                     row.tau_min_h)

                missed_detection_c = 0
                missed_detection_h = 0
                missed_detection_t = 0
                print(first_detected_org_c, ' ', first_detected_org_h, ' ', first_detected_org_t)
                if first_detected_org_c == 0:
                    missed_detection_c = attack_end_date - attack_start_date
                else:
                    missed_detection_c = first_detected_org_c - attack_start_date

                if first_detected_org_t == 0:
                    missed_detection_t = attack_end_date - attack_start_date
                else:
                    missed_detection_t = first_detected_org_t - attack_start_date

                if first_detected_org_h == 0:
                    missed_detection_h = attack_end_date - attack_start_date
                else:
                    missed_detection_h = first_detected_org_h - attack_start_date

                object_fm = {"beta_p": row.beta_p, 'beta_n': row.beta_n,
                             'tau_max_c': row.tau_max_c, 'tau_min_c': row.tau_min_c,
                             'fa_c': len(false_alarm_c), 'md_c': missed_detection_c, 'fa_c_a': len(false_alarm_ca),
                             "beta_p": row.beta_p, 'beta_n': row.beta_n,
                             'tau_max_t': row.tau_max_T, 'tau_min_t': row.tau_min_T,
                             'fa_t': len(false_alarm_T), 'md_t': missed_detection_t, 'fa_t_a': len(false_alarm_ta),

                             'tau_max_h': row.tau_max_h, 'tau_min_h': row.tau_min_h,
                             'fa_h': len(false_alarm_h), 'md_h': missed_detection_h, 'fa_h_a': len(false_alarm_ha)
                             }
                result.append(object_fm)
            fm_result_frame = pd.DataFrame(result)
            result_file_name = f'RA_TR{del_val}_TE{del_val_test}_ROMAL{romal_val}_CHT_6Ms.csv'
            fm_result_frame.to_csv(os.path.join(output_path, result_file_name), index=False)
