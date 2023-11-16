import pandas as pd
import sys
import os
import itertools

input_path_CHT = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New attacked data duration 4Mths/track_FA_2014_6Ms_diff'
# constant values
DEL_AVG_ARRAY = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
RO_MAL_ARRAY = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


# fa_frame_ra = pd.read_csv('RA_TR150_TE100_ROMAL03_CH.csv')
# fa_frame_ra = pd.read_csv('RA_FA_MD_DEL150_ROMAL03_CR_DEL_175_C.csv')

def get_cost(fa_frame):
    fa_frame['cost_c'] = None
    fa_frame['cost_h'] = None
    fa_frame['cost_t'] = None

    for index, row in fa_frame.iterrows():
        fa_frame.at[index, 'cost_c'] = row.md_c * 0.3 + row.fa_c * 0.7
        fa_frame.at[index, 'cost_h'] = row.md_h * 0.3 + row.fa_h * 0.7
        fa_frame.at[index, 'cost_t'] = row.md_t * 0.3 + row.fa_t * 0.7
    return fa_frame


# Cauchy
def get_opt_beta_c(cost_frame):
    # flag = 1
    min_beta_c = None
    max_beta_c = None
    tau_min_c = None
    tau_max_c = None
    min_c = sys.float_info.max
    for index, row in cost_frame.iterrows():
        # if(flag == 1):
        #     min_beta_c = row.beta_p
        #     max_beta_c = row.beta_n
        #     min_c = row.cost_c
        #     flag = 0
        # else:
        if row.cost_c < min_c:  # & (row.tau_max_c != 0) & (row.tau_min_c != 0)& (row.beta_p!=0) & (row.beta_n!=0)
            print(' *** ', row.cost_c, min_c, row.tau_max_c, row.tau_min_c)
            min_c = row.cost_c
            min_beta_c = row.beta_p
            max_beta_c = row.beta_n
            tau_max_c = row.tau_max_c
            tau_min_c = row.tau_min_c
    return max_beta_c, min_beta_c, tau_max_c, tau_min_c


# Tukey
def get_opt_beta_t(cost_frame):
    # flag = 1
    min_beta_t = None
    max_beta_t = None
    tau_min_t = None
    tau_max_t = None
    min_t = sys.float_info.max
    for index, row in cost_frame.iterrows():
        # if(flag == 1):
        #     min_beta_h = row.beta_p
        #     max_beta_h = row.beta_n
        #     min_h = row.cost_h
        #     flag = 0
        # else:
        if row.cost_t < min_t:  # & (row.tau_max_h!=0) & (row.tau_min_h!=0) & (row.beta_p!=0) & (row.beta_n!=0)
            print(' *** ', row.cost_t, min_t, row.tau_max_t, row.tau_min_t)
            min_t = row.cost_t
            min_beta_t = row.beta_p
            max_beta_t = row.beta_n
            tau_max_t = row.tau_max_t
            tau_min_t = row.tau_min_t
    return max_beta_t, min_beta_t, tau_max_t, tau_min_t


# Huber
def get_opt_beta_h(cost_frame):
    # flag = 1
    min_beta_h = None
    max_beta_h = None
    tau_min_h = None
    tau_max_h = None
    min_h = sys.float_info.max
    for index, row in cost_frame.iterrows():
        # if(flag == 1):
        #     min_beta_h = row.beta_p
        #     max_beta_h = row.beta_n
        #     min_h = row.cost_h
        #     flag = 0
        # else:
        if row.cost_h < min_h:  # & (row.tau_max_h!=0) & (row.tau_min_h!=0) & (row.beta_p!=0) & (row.beta_n!=0)
            print(' *** ', row.cost_h, min_h, row.tau_max_h, row.tau_min_h)
            min_h = row.cost_h
            min_beta_h = row.beta_p
            max_beta_h = row.beta_n
            tau_max_h = row.tau_max_h
            tau_min_h = row.tau_min_h
    return max_beta_h, min_beta_h, tau_max_h, tau_min_h


result_list = []
for del_val in DEL_AVG_ARRAY:
    for romal_val in RO_MAL_ARRAY:
        # Form the file name based on the values of del and romal
        file_name = f'RA_TR{del_val}_TE{del_val}_ROMAL{romal_val}_CHT_6Ms.csv'

        # Read the three DataFrames from the respective folders
        fa_frame_ra = pd.read_csv(os.path.join(input_path_CHT, file_name))
        cost_frame_ra = get_cost(fa_frame_ra)
        max_beta_c_ra, min_beta_c_ra, tau_max_c, tau_min_c = get_opt_beta_c(cost_frame_ra)
        max_beta_h_ra, min_beta_h_ra, tau_max_h, tau_min_h = get_opt_beta_h(cost_frame_ra)
        max_beta_t_ra, min_beta_t_ra, tau_max_t, tau_min_t = get_opt_beta_t(cost_frame_ra)
        result_dict = {
            'max_beta_c_ra': max_beta_c_ra, 'min_beta_c_ra': min_beta_c_ra,
            'tau_max_c': tau_max_c, 'tau_min_c': tau_min_c,
            'max_beta_h_ra': max_beta_h_ra, 'min_beta_h_ra': min_beta_h_ra,
            'tau_max_h': tau_max_h, 'tau_min_h': tau_min_h,
            'max_beta_t_ra': max_beta_t_ra, 'min_beta_t_ra': min_beta_t_ra,
            'tau_max_t': tau_max_t, 'tau_min_t': tau_min_t,
            'TR_delta': del_val, 'TE_delta': del_val, 'ROMAL': romal_val
        }
        result_list.append(result_dict)
result_df = pd.DataFrame(result_list)

# Save the DataFrame as a CSV file in the specified directory
output_path = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New attacked data duration 4Mths/Optimal_beta_tau_diff_romal_delta_6Ms'
output_file_name = 'optimal_tau_beta_both_6Ms.csv'
output_file_path = os.path.join(output_path, output_file_name)
result_df.to_csv(output_file_path, index=False)

result_list = []
for del_val, romal_val in itertools.product(DEL_AVG_ARRAY, RO_MAL_ARRAY):
    for del_val_test in DEL_AVG_ARRAY:
        if del_val_test != del_val:
            # Form the file name based on the values of del and romal
            file_name = f'RA_TR{del_val}_TE{del_val_test}_ROMAL{romal_val}_CHT_6Ms.csv'

            # Read the three DataFrames from the respective folders
            fa_frame_ra = pd.read_csv(os.path.join(input_path_CHT, file_name))
            cost_frame_ra = get_cost(fa_frame_ra)
            max_beta_c_ra, min_beta_c_ra, tau_max_c, tau_min_c = get_opt_beta_c(cost_frame_ra)
            max_beta_h_ra, min_beta_h_ra, tau_max_h, tau_min_h = get_opt_beta_h(cost_frame_ra)
            max_beta_t_ra, min_beta_t_ra, tau_max_t, tau_min_t = get_opt_beta_t(cost_frame_ra)
            result_dict = {
                'max_beta_c_ra': max_beta_c_ra, 'min_beta_c_ra': min_beta_c_ra,
                'tau_max_c': tau_max_c, 'tau_min_c': tau_min_c,
                'max_beta_h_ra': max_beta_h_ra, 'min_beta_h_ra': min_beta_h_ra,
                'tau_max_h': tau_max_h, 'tau_min_h': tau_min_h,
                'max_beta_t_ra': max_beta_t_ra, 'min_beta_t_ra': min_beta_t_ra,
                'tau_max_t': tau_max_t, 'tau_min_t': tau_min_t,
                'TR_delta': del_val, 'TE_delta': del_val_test, 'ROMAL': romal_val
            }
            result_list.append(result_dict)
result_df = pd.DataFrame(result_list)

# Save the DataFrame as a CSV file in the specified directory
output_path = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New attacked data duration 4Mths/Optimal_beta_tau_diff_romal_delta_2Ms'
output_file_name = 'optimal_tau_beta_diff_2Ms.csv'
output_file_path = os.path.join(output_path, output_file_name)
result_df.to_csv(output_file_path, index=False)
