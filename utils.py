import numpy as np
from math import sqrt


def calculate_t_max_cauchy(ruc_frame, keys, tau_range, w1, w2, beta_range_max):
    opt_tau_max_list = list()
    for b in beta_range_max:
        tau_list = list()
        cost_list = list()
        for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):
            cost = 0
            for row in ruc_frame.itertuples():
                for key in keys:
                    if getattr(row, "ruc" + key) > 0:
                        x = getattr(row, "ruc" + key) - tau
                        if x >= 0:
                            cost += np.log(1 + (((x * w2) / b) ** 2))

                        else:
                            cost += np.log(1 + (((x * w1) / b) ** 2))

            cost_list.append(b * b * cost)
            tau_list.append(tau)
        opt_tau_max_list.append(tau_list[np.argmin(cost_list)])

    return opt_tau_max_list, cost_list, tau_list


def calculate_t_min_cauchy(ruc_frame, keys, tau_range, w1, w2, beta_range_min):
    opt_tau_min_list = list()
    for b in beta_range_min:
        tau_list = list()
        cost_list = list()
        for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):
            cost = 0
            for row in ruc_frame.itertuples():
                for key in keys:
                    if getattr(row, "ruc" + key) < 0:
                        x = getattr(row, "ruc" + key) - tau
                        if x >= 0:
                            cost += np.log(1 + (((x * w1) / b) ** 2))
                        else:
                            cost += np.log(1 + (((x * w2) / b) ** 2))
            cost_list.append(b * b * cost)
            tau_list.append(tau)
        opt_tau_min_list.append(tau_list[np.argmin(cost_list)])

    return opt_tau_min_list, cost_list, tau_list


def calculate_t_max_huber(ruc_frame, keys, tau_range, w1, w2, beta_range_max):
    opt_tau_max_list = list()
    for b in beta_range_max:

        tau_list = list()
        cost_list = list()
        for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):
            cost = 0
            for row in ruc_frame.itertuples():
                for key in keys:
                    if getattr(row, "ruc" + key) > 0:
                        x = getattr(row, "ruc" + key) - tau
                        if x >= 0:
                            if (abs(x) * w2) <= b:
                                cost += .5 * ((abs(x) * w2) ** 2)
                            else:
                                cost += b * (abs(x) * w2) - .5 * (b ** 2)
                        else:
                            if (abs(x) * w1) <= b:
                                cost += .5 * ((abs(x) * w1) ** 2)
                            else:
                                cost += b * (abs(x) * w1) - (.5 * (b ** 2))

            cost_list.append(cost)
            tau_list.append(tau)
        opt_tau_max_list.append(tau_list[np.argmin(cost_list)])

    return opt_tau_max_list, cost_list, tau_list


def calculate_t_min_huber(ruc_frame, keys, tau_range, w1, w2, beta_range_min):
    opt_tau_min_list = list()
    for b in beta_range_min:

        tau_list = list()
        cost_list = list()
        for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):
            cost = 0
            for row in ruc_frame.itertuples():
                for key in keys:
                    if getattr(row, "ruc" + key) < 0:
                        x = getattr(row, "ruc" + key) - tau
                        if x >= 0:
                            if (abs(x) * w1) <= b:
                                cost += .5 * ((abs(x) * w1) ** 2)
                            else:
                                cost += b * (abs(x) * w1) - (.5 * (b ** 2))
                        else:
                            if (abs(x) * w2) <= b:
                                cost += .5 * ((abs(x) * w2) ** 2)
                            else:
                                cost += b * (abs(x) * w2) - (.5 * (b ** 2))
            cost_list.append(cost)
            tau_list.append(tau)
        opt_tau_min_list.append(tau_list[np.argmin(cost_list)])

    return opt_tau_min_list, cost_list, tau_list


def calculate_t_max_pseudo_huber(ruc_frame, keys, tau_range, w1, w2, beta_range_max):
    opt_tau_max_list = list()

    for b in beta_range_max:
        tau_list = list()
        cost_list = list()

        for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):
            cost = 0
            for row in ruc_frame.itertuples():
                for key in keys:
                    if getattr(row, "ruc" + key) > 0:
                        x = getattr(row, "ruc" + key) - tau
                        if x >= 0:
                            cost += sqrt(1 + ((abs(x) * w1) / b) ** 2) - 1
                        else:
                            cost += sqrt(1 + ((abs(x) * w2) / b) ** 2) - 1

            cost_list.append(b * cost)
            tau_list.append(tau)
        opt_tau_max_list.append(tau_list[np.argmin(cost_list)])

    return opt_tau_max_list, cost_list, tau_list


def calculate_t_min_pseudo_huber(ruc_frame, keys, tau_range, w1, w2, beta_range_min):
    opt_tau_min_list = list()

    for b in beta_range_min:
        tau_list = list()
        cost_list = list()
        for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):

            cost = 0
            for row in ruc_frame.itertuples():
                for key in keys:
                    if getattr(row, "ruc" + key) < 0:
                        x = getattr(row, "ruc" + key) - tau
                        if x >= 0:
                            cost += sqrt(1 + ((abs(x) * w1) / b) ** 2) - 1

                        else:
                            cost += sqrt(1 + ((abs(x) * w2) / b) ** 2) - 1

            cost_list.append(b * cost)
            tau_list.append(tau)
        opt_tau_min_list.append(tau_list[np.argmin(cost_list)])

    return opt_tau_min_list, cost_list, tau_list


def calculate_t_max_Tukey(ruc_frame, keys, tau_range, w1, w2, beta_range_max):
    opt_tau_max_list = list()

    for b in beta_range_max:
        tau_list = list()
        cost_list = list()

        for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):
            cost = 0

            for row in ruc_frame.itertuples():
                for key in keys:
                    if getattr(row, "ruc" + key) > 0:
                        x = getattr(row, "ruc" + key) - tau
                        if x >= 0:
                            if (abs(x) * w2) <= b:
                                cost += ((b ** 2) / 6) * (1 - (1 - ((abs(x) * w2) / b) ** 2) ** 3)
                            else:
                                cost += (b ** 2) / 6
                        else:
                            if (abs(x) * w1) <= b:
                                cost += ((b ** 2) / 6) * (1 - (1 - ((abs(x) * w1) / b) ** 2) ** 3)
                            else:
                                cost += (b ** 2) / 6

            cost_list.append(cost)
            tau_list.append(tau)
        opt_tau_max_list.append(tau_list[np.argmin(cost_list)])

    return opt_tau_max_list, cost_list, tau_list


def calculate_t_min_Tukey(ruc_frame, keys, tau_range, w1, w2, beta_range_min):
    opt_tau_min_list = list()

    for b in beta_range_min:
        tau_list = list()
        cost_list = list()
        for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):

            cost = 0
            for row in ruc_frame.itertuples():
                for key in keys:
                    if getattr(row, "ruc" + key) < 0:
                        x = getattr(row, "ruc" + key) - tau
                        if x >= 0:
                            if (abs(x) * w1) <= b:
                                cost += ((b ** 2) / 6) * (1 - (1 - ((abs(x) * w1) / b) ** 2) ** 3)
                            else:
                                cost += (b ** 2) / 6
                        else:
                            if (abs(x) * w2) <= b:
                                cost += ((b ** 2) / 6) * (1 - (1 - ((abs(x) * w2) / b) ** 2) ** 3)
                            else:
                                cost += (b ** 2) / 6

            cost_list.append(cost)
            tau_list.append(tau)
        opt_tau_min_list.append(tau_list[np.argmin(cost_list)])

    return opt_tau_min_list, cost_list, tau_list

# def calculate_t_max_welsch(ruc_frame, keys, tau_range, w1, w2, b=0.00009):
#     tau_list = list()
#     cost_list = list()
#     # s_less_list = list()
#     # s_high_list = list()
#     for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):
#         cost = 0
#         # cost_less = 1
#         # cost_high = 1
#         for row in ruc_frame.itertuples():
#             for key in keys:
#                 if getattr(row, "ruc" + key) > 0:
#                     x = getattr(row, "ruc" + key) - tau
#                     if x >= 0:
#                         cost += 1 - np.exp((-1) * 0.5 * ((x * w1) / b) ** 2)
#                     # cost_less += 1 - np.exp((-1) * 0.5 * ((x * w1) / b) ** 2)
#                     else:
#                         cost += 1 - np.exp((-1) * 0.5 * ((x * w2) / b) ** 2)
#                     # cost_high += 1 - np.exp((-1) * 0.5 * ((x * w1) / b) ** 2)

#         # s_high_list.append(cost_high)
#         # s_less_list.append(cost_less)
#         cost_list.append(cost)
#         tau_list.append(tau)

#     return tau_list[np.argmin(cost_list)], cost_list, tau_list


# def calculate_t_min_welsch(ruc_frame, keys, tau_range, w1, w2, b=0.00009):
#     tau_list = list()
#     cost_list = list()
#     # s_less_list = list()
#     # s_high_list = list()
#     for tau in np.arange(tau_range[0], tau_range[1], tau_range[2]):
#         cost = 0
#         # cost_less = 1
#         # cost_high = 1
#         for row in ruc_frame.itertuples():
#             for key in keys:
#                 if getattr(row, "ruc" + key) < 0:
#                     x = getattr(row, "ruc" + key) - tau
#                     if x >= 0:
#                         cost += 1 - np.exp((-1) * 0.5 * ((x * w1) / b) ** 2)
#                     # cost_less += 1 - np.exp((-1) * 0.5 * ((x * w1) / b) ** 2)
#                     else:
#                         cost += 1 - np.exp((-1) * 0.5 * ((x * w2) / b) ** 2)
#                     # cost_high += 1 - np.exp((-1) * 0.5 * ((x * w1) / b) ** 2)

#         # s_high_list.append(cost_high)
#         # s_less_list.append(cost_less)
#         cost_list.append(cost)
#         tau_list.append(tau)

#     return tau_list[np.argmin(cost_list)], cost_list, tau_list
