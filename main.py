import utils
import pandas as pd
import numpy as np
import os

loss_function = 'huber'  # l2 huber pseudo_huber

input_directory_path = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New attacked data duration 4Mths/RUCs_values_2014_benign'

output_directory_path = 'C:/Users/abedz/OneDrive - Western Michigan University/PhD/Summer2021/New folder/navid w/New ' \
                        'attacked data duration 4Mths'


# Function to read all CSV files in the directory and process them
def tau_mix_min(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            keys = ['2014']
            file_path = os.path.join(input_directory, filename)
            data_need = pd.read_csv(file_path)
            # range tau max
            maxRuc2014 = data_need['ruc2014'].max()
            max_candidate = maxRuc2014

            # range tau min
            minRuc2014 = data_need['ruc2014'].min()
            min_candidate = minRuc2014

            beta_range_max = np.arange(.00001, max_candidate, .00008)
            beta_range_min = np.arange(.00001, abs(min_candidate), .00008)

            [attack_t_max_opt, cost_list_max, tau_list_max] = getattr(utils, 'calculate_t_max_' + loss_function)(
                data_need, keys,
                tau_range=[.0, max_candidate, .000080],
                w1=.5, w2=2., beta_range_max=np.arange(.00001, max_candidate, .00008))
            [attack_t_min_opt, cost_list_min, tau_list_min] = getattr(utils, 'calculate_t_min_' + loss_function)(
                data_need, keys,
                tau_range=[min_candidate, .0, .000080],
                w1=.5, w2=2., beta_range_min=np.arange(.00001, abs(min_candidate), .00008))
            # optimal tau max
            attack_t_max_opt_arr = pd.DataFrame(np.transpose(np.array([attack_t_max_opt])), columns=["optimal_tau_max"])
            beta_range_max_arr = pd.DataFrame(np.transpose(np.array([beta_range_max])), columns=["beta_p"])
            # optimal tau min
            attack_t_min_opt_arr = pd.DataFrame(np.transpose(np.array([attack_t_min_opt])), columns=["optimal_tau_min"])
            beta_range_min_arr = pd.DataFrame(np.transpose(np.array([beta_range_min])), columns=["beta_n"])
            optimal_tau_beta_table = pd.concat(
                [attack_t_max_opt_arr, beta_range_max_arr, attack_t_min_opt_arr, beta_range_min_arr],
                axis=1)
            # Save RUCs_values for each input data in the output directory
            output_filename = 'huber_benign_tau_' + os.path.splitext(filename)[0] + '.csv'
            output_file_path = os.path.join(output_directory, output_filename)
            optimal_tau_beta_table.to_csv(output_file_path, index=False)


# Call the function to process all files and save in the specified output directory
tau_mix_min(input_directory_path, output_directory_path)

# ruc_frame_attacked = pd.read_csv('attacked_residual.csv')
# keys = ['2014', '2015']
