# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
from scipy import stats
import numpy as np
import pandas as pd


import uncert as uc

uc.logs(show_level='info', show_color=True)
logger = uc.CustomLogger(__name__)  # use custom logger

# const
# SAVE_P = True  # save pickle files with data
# LOAD_P = False  # load pickle files with data
# SAVE_CSV = True  # load csv files with data
# FILTER_DATA = True  # filter Appen and heroku data
# CLEAN_DATA = True  # clean Appen data
# REJECT_CHEATERS = False  # reject cheaters on Appen
# UPDATE_MAPPING = True  # update mapping with keypress data
# SHOW_STATS = True  # should figures be plotted
# SHOW_OUTPUT = True  # should figures be plotted

# for debugging, skip processing
SAVE_P = False  # save pickle files with data
LOAD_P = True  # load pickle files with data
SAVE_CSV = True  # load csv files with data
FILTER_DATA = False  # filter Appen and heroku data
CLEAN_DATA = False  # clean Appen data
REJECT_CHEATERS = False  # reject cheaters on Appen
UPDATE_MAPPING = False  # update mapping with keypress data
SHOW_STATS = True  # should figures be plotted
SHOW_OUTPUT = False  # should figures be plotted

file_mapping = 'mapping.p'  # file to save updated mapping

if __name__ == '__main__':
    # create object for working with heroku data
    files_heroku = uc.common.get_configs('files_heroku')
    heroku = uc.analysis.Heroku(files_data=files_heroku, save_p=SAVE_P, load_p=LOAD_P, save_csv=SAVE_CSV)
    # read heroku data
    heroku_data = heroku.read_data(filter_data=FILTER_DATA)
    # create object for working with appen data
    file_appen = uc.common.get_configs('file_appen')
    appen = uc.analysis.Appen(file_data=file_appen, save_p=SAVE_P, load_p=LOAD_P, save_csv=SAVE_CSV)
    # read appen data
    appen_data = appen.read_data(filter_data=FILTER_DATA,
                                 clean_data=CLEAN_DATA)
    # get keys in data files
    heroku_data_keys = heroku_data.keys()
    appen_data_keys = appen_data.keys()
    # flag and reject cheaters
    if REJECT_CHEATERS:
        qa = uc.analysis.QA(file_cheaters=uc.common.get_configs('file_cheaters'),
                            job_id=uc.common.get_configs('appen_job'))
        qa.reject_users()
        qa.ban_users()
    # merge heroku and appen data frames into one
    all_data = heroku_data.merge(appen_data, left_on='worker_code', right_on='worker_code')
    logger.info('Data from {} participants included in analysis.', all_data.shape[0])
    # update original data files
    heroku_data = all_data[all_data.columns.intersection(heroku_data_keys)]
    heroku_data = heroku_data.set_index('worker_code')
    heroku.set_data(heroku_data)  # update object with filtered data
    appen_data = all_data[all_data.columns.intersection(appen_data_keys)]
    appen_data = appen_data.set_index('worker_code')
    appen.set_data(appen_data)  # update object with filtered data
    appen.show_info()  # show info for filtered data
    # generate country-specific data
    countries_data = appen.process_countries()
    # update mapping with keypress data
    if UPDATE_MAPPING:
        # read in questions for stimuli
        qs_videos = heroku.read_questions_videos()
        qs_images = heroku.read_questions_images()
        # process post-trial questions and return combined df for all stimuli
        mapping = heroku.process_stimulus_questions()
        # export to pickle
        uc.common.save_to_p(file_mapping, mapping, 'mapping of stimuli')
    else:
        mapping = uc.common.load_from_p(file_mapping, 'mapping of stimuli')
    if SHOW_STATS:
        # Statistics
        # copy mapping to a temp df
        df = mapping.copy()
        # convert type of vehicle to num
        df['vehicle_type'] = df['vehicle_type'].map({'AV': 0, 'MDV': 1})
        # 1. Kolmogorov-Smirnov test
        logger.info('Kolmogorov-Smirnov test for raw answers of stimulus responses: {}.',
                    stats.kstest(list(df['raw_answers'].explode()), 'norm'))
        logger.info('Kolmogorov-Smirnov test for STD of stimulus responses: {}.', stats.kstest(df['std'], 'norm'))
        logger.info('Kolmogorov-Smirnov test for mean of stimulus responses: {}.', stats.kstest(df['mean'], 'norm'))
        logger.info('Kolmogorov-Smirnov test for median of stimulus responses: {}.',
                    stats.kstest(df['median'], 'norm'))
        # pairs of stimuli
        for index, row in df.iterrows():
            logger.info('Kolmogorov-Smirnov test for responses for stimulus {}: {}.',
                        index, stats.kstest(row['raw_answers'], 'norm'))
        # 2. A paired t-test between all the uncertainty of each sample group  (manually driven vs. fully automated).
        group_a = df.where(df.vehicle_type == 0).dropna()['mean']
        group_b = df.where(df.vehicle_type == 1).dropna()['mean']
        logger.info('A paired t-test between all the uncertainty of manually driven vs. fully automated: {}.',
                    stats.ttest_ind(group_a, group_b))
        # pairs of stimuli
        for stimulus in range(20, 30):
            group_a = df.loc[stimulus].dropna()['raw_answers']
            group_b = df.loc[stimulus+10].dropna()['raw_answers']
            logger.info('A paired t-test for stimuli {} and {}: {}.',
                        stimulus,
                        stimulus + 10,
                        stats.ttest_ind(group_a, group_b))
        # 3. Wilcoxon signed-rank test between all the uncertainty of each sample group (manually driven vs. fully
        # automated).
        group_a = df.where(df.vehicle_type == 0).dropna()['mean']
        group_b = df.where(df.vehicle_type == 1).dropna()['mean']
        logger.info('Wilcoxon signed-rank test for mean of stimulus responses: {}.', stats.wilcoxon(group_a, group_b))
        group_a = df.where(df.vehicle_type == 0).dropna()['std']
        group_b = df.where(df.vehicle_type == 1).dropna()['std']
        logger.info('Wilcoxon signed-rank test for STD of stimulus responses: {}.', stats.wilcoxon(group_a, group_b))
        group_a = df.where(df.vehicle_type == 0).dropna()['median']
        group_b = df.where(df.vehicle_type == 1).dropna()['median']
        logger.info('Wilcoxon signed-rank test for median of stimulus responses: {}.',
                    stats.wilcoxon(group_a, group_b))
        # pairs of stimuli
        for stimulus in range(20, 30):
            group_a = df.loc[stimulus].dropna()['raw_answers']
            group_b = df.loc[stimulus+10].dropna()['raw_answers']
            logger.info('Wilcoxon signed-rank test for stimuli {} and {}: {}.',
                        stimulus,
                        stimulus + 10,
                        stats.mannwhitneyu(group_a, group_b))
    if SHOW_OUTPUT:
        # Output
        analysis = uc.analysis.Analysis(save_csv=SAVE_CSV)
        logger.info('Creating figures.')
        # columns to drop in correlation matrix and scatter matrix
        columns_drop = ['short_name', 'id', 'question', 'short_name', 'label_0', 'label_100', 'stimulus',
                        'video_length', 'description', 'comments', 'partner_video', 'raw_answers']
        # copy mapping to a temp df
        df = mapping.copy()
        # convert type of vehicle to num
        df['vehicle_type'] = df['vehicle_type'].map({'AV': 0, 'MDV': 1})
        # set nan to -1
        df = df.fillna(-1)
        # create correlation matrix for mapping
        analysis.corr_matrix(df, columns_drop=columns_drop, save_file=True, filename='_corr_matrix_mapping.jpg')
        # create correlation matrix for mapping
        analysis.scatter_matrix(df,
                                columns_drop=columns_drop,
                                color='vehicle_type',
                                symbol='vehicle_type',
                                diagonal_visible=False,
                                save_file=True,
                                filename='scatter_matrix_mapping')
        # columns to drop in correlation matrix and scatter matrix for the combined dataframe
        columns_drop = ['worker_code', 'index', 'unit_id', 'end', 'id', 'start', 'tainted', 'channel', 'trust',
                        'worker_id', 'country', 'region', 'city', 'ip', 'answers_hidden', 'consent', 'instructions',
                        'place_other', 'experiences_other', 'device_other', 'place', 'year_ad', 'suggestions_ad',
                        'device', 'item', 'browser_app_name', 'browser_full_version', 'browser_major_version',
                        'browser_name', 'browser_user_agent', 'window_height', 'window_width',
                        'image_0-dur-0', 'image_0-dur-1', 'image_0-event-0', 'image_0-event-1', 'image_0-time-0', 'image_0-time-1',  # noqa: E501
                        'image_1-dur-0', 'image_1-dur-1', 'image_1-event-0', 'image_1-event-1', 'image_1-time-0', 'image_1-time-1',  # noqa: E501
                        'image_2-dur-0', 'image_2-dur-1', 'image_2-event-0', 'image_2-event-1', 'image_2-time-0', 'image_2-time-1',  # noqa: E501
                        'image_3-dur-0', 'image_3-dur-1', 'image_3-event-0', 'image_3-event-1', 'image_3-time-0', 'image_3-time-1',  # noqa: E501
                        'video_0-dur-0', 'video_0-dur-1', 'video_0-event-0', 'video_0-event-1', 'video_0-time-0', 'video_0-time-1',  # noqa: E501
                        'video_1-dur-0', 'video_1-dur-1', 'video_1-event-0', 'video_1-event-1', 'video_1-time-0', 'video_1-time-1',  # noqa: E501
                        'video_10-dur-0', 'video_10-dur-1', 'video_10-event-0', 'video_10-event-1', 'video_10-time-0', 'video_10-time-1',  # noqa: E501
                        'video_11-dur-0', 'video_11-dur-1', 'video_11-event-0', 'video_11-event-1', 'video_11-time-0', 'video_11-time-1',  # noqa: E501
                        'video_12-dur-0', 'video_12-dur-1', 'video_12-event-0', 'video_12-event-1', 'video_12-time-0', 'video_12-time-1',  # noqa: E501
                        'video_13-dur-0', 'video_13-dur-1', 'video_13-event-0', 'video_13-event-1', 'video_13-time-0', 'video_13-time-1',  # noqa: E501
                        'video_14-dur-0', 'video_14-dur-1', 'video_14-event-0', 'video_14-event-1', 'video_14-time-0', 'video_14-time-1',  # noqa: E501
                        'video_15-dur-0', 'video_15-dur-1', 'video_15-event-0', 'video_15-event-1', 'video_15-time-0', 'video_15-time-1',  # noqa: E501
                        'video_16-dur-0', 'video_16-dur-1', 'video_16-event-0', 'video_16-event-1', 'video_16-time-0', 'video_16-time-1',  # noqa: E501
                        'video_17-dur-0', 'video_17-dur-1', 'video_17-event-0', 'video_17-event-1', 'video_17-time-0', 'video_17-time-1',  # noqa: E501
                        'video_18-dur-0', 'video_18-dur-1', 'video_18-event-0', 'video_18-event-1', 'video_18-time-0', 'video_18-time-1',  # noqa: E501
                        'video_19-dur-0', 'video_19-dur-1', 'video_19-event-0', 'video_19-event-1', 'video_19-time-0', 'video_19-time-1',  # noqa: E501
                        'video_2-dur-0', 'video_2-dur-1', 'video_2-event-0', 'video_2-event-1', 'video_2-time-0', 'video_2-time-1',  # noqa: E501
                        'video_3-dur-0', 'video_3-dur-1', 'video_3-event-0', 'video_3-event-1', 'video_3-time-0', 'video_3-time-1',  # noqa: E501
                        'video_4-dur-0', 'video_4-dur-1', 'video_4-event-0', 'video_4-event-1', 'video_4-time-0', 'video_4-time-1',  # noqa: E501
                        'video_5-dur-0', 'video_5-dur-1', 'video_5-event-0', 'video_5-event-1', 'video_5-time-0', 'video_5-time-1',  # noqa: E501
                        'video_6-dur-0', 'video_6-dur-1', 'video_6-event-0', 'video_6-event-1', 'video_6-time-0', 'video_6-time-1',  # noqa: E501
                        'video_7-dur-0', 'video_7-dur-1', 'video_7-event-0', 'video_7-event-1', 'video_7-time-0', 'video_7-time-1',  # noqa: E501
                        'video_8-dur-0', 'video_8-dur-1', 'video_8-event-0', 'video_8-event-1', 'video_8-time-0', 'video_8-time-1',  # noqa: E501
                        'video_9-dur-0', 'video_9-dur-1', 'video_9-event-0', 'video_9-event-1', 'video_9-time-0', 'video_9-time-1']  # noqa: E501
        # copy mapping to a temp df
        df = all_data.copy()
        # convert columns to num values
        df['milage'] = df['milage'].map({'0_km__mi': 0,
                                         '1__1000_km_1__621_mi': 1,
                                         '1001__5000_km_622__3107_mi': 2,
                                         '5001__15000_km_3108__9321_mi': 3,
                                         '15001__20000_km_9322__12427_mi': 4,
                                         '20001__25000_km_12428__15534_mi': 5,
                                         '25001__35000_km_15535__21748_mi': 6,
                                         '35001__50000_km_21749__31069_mi': 7,
                                         '50001__100000_km_31070__62137_mi': 8,
                                         'more_than_100000_km_more_than_62137_mi': 9,
                                         'i_prefer_not_to_respond': np.nan})
        df['dbq1_anger'] = df['dbq1_anger'].map({'0_times_per_month': 0,
                                                 '1_to_3_times_per_month': 1,
                                                 '4_to_6_times_per_month': 2,
                                                 '7_to_9_times_per_month': 3,
                                                 '10_or_more_times_per_month': 4,
                                                 'i_prefer_not_to_respond': np.nan})
        df['dbq2_speed_motorway'] = df['dbq2_speed_motorway'].map({'0_times_per_month': 0,
                                                                   '1_to_3_times_per_month': 1,
                                                                   '4_to_6_times_per_month': 2,
                                                                   '7_to_9_times_per_month': 3,
                                                                   '10_or_more_times_per_month': 4,
                                                                   'i_prefer_not_to_respond': np.nan})
        df['dbq3_speed_residential'] = df['dbq3_speed_residential'].map({'0_times_per_month': 0,
                                                                         '1_to_3_times_per_month': 1,
                                                                         '4_to_6_times_per_month': 2,
                                                                         '7_to_9_times_per_month': 3,
                                                                         '10_or_more_times_per_month': 4,
                                                                         'i_prefer_not_to_respond': np.nan})
        df['dbq4_headway'] = df['dbq4_headway'].map({'0_times_per_month': 0,
                                                     '1_to_3_times_per_month': 1,
                                                     '4_to_6_times_per_month': 2,
                                                     '7_to_9_times_per_month': 3,
                                                     '10_or_more_times_per_month': 4,
                                                     'i_prefer_not_to_respond': np.nan})
        df['dbq5_traffic_lights'] = df['dbq5_traffic_lights'].map({'0_times_per_month': 0,
                                                                   '1_to_3_times_per_month': 1,
                                                                   '4_to_6_times_per_month': 2,
                                                                   '7_to_9_times_per_month': 3,
                                                                   '10_or_more_times_per_month': 4,
                                                                   'i_prefer_not_to_respond': np.nan})
        df['dbq6_horn'] = df['dbq6_horn'].map({'0_times_per_month': 0,
                                               '1_to_3_times_per_month': 1,
                                               '4_to_6_times_per_month': 2,
                                               '7_to_9_times_per_month': 3,
                                               '10_or_more_times_per_month': 4,
                                               'i_prefer_not_to_respond': np.nan})
        df['dbq7_mobile'] = df['dbq7_mobile'].map({'0_times_per_month': 0,
                                                   '10_or_more_times_per_month': 1,
                                                   '1_to_3_times_per_month': 2,
                                                   '4_to_6_times_per_month': 3,
                                                   '7_to_9_times_per_month': 4,
                                                   'i_prefer_not_to_respond': np.nan})
        df['license'] = df['license'].map({'no': 0,
                                           'yes': 1})
        df['certainty_experiences'] = df['certainty_experiences'].map({'strongly_disagree': 0,
                                                                       'disagree': 1,
                                                                       'neither_agree_nor_disagree': 2,
                                                                       'agree': 3,
                                                                       'strongly_agree': 4})
        df['uncertainty_experiences'] = df['uncertainty_experiences'].map({'strongly_disagree': 0,
                                                                           'disagree': 1,
                                                                           'neither_agree_nor_disagree': 2,
                                                                           'agree': 3,
                                                                           'strongly_agree': 4})
        df['certainty_decisions'] = df['certainty_decisions'].map({'strongly_disagree': 0,
                                                                   'disagree': 1,
                                                                   'neither_agree_nor_disagree': 2,
                                                                   'agree': 3,
                                                                   'strongly_agree': 4})
        df['uncertainty_decisions'] = df['uncertainty_decisions'].map({'strongly_disagree': 0,
                                                                       'disagree': 1,
                                                                       'neither_agree_nor_disagree': 2,
                                                                       'agree': 3,
                                                                       'strongly_agree': 4})
        df['certainty_myself'] = df['certainty_myself'].map({'strongly_disagree': 0,
                                                             'disagree': 1,
                                                             'neither_agree_nor_disagree': 2,
                                                             'agree': 3,
                                                             'strongly_agree': 4})
        df['uncertainty_myself'] = df['uncertainty_myself'].map({'strongly_disagree': 0,
                                                                 'disagree': 1,
                                                                 'neither_agree_nor_disagree': 2,
                                                                 'agree': 3,
                                                                 'strongly_agree': 4})
        df['certainty_day'] = df['certainty_day'].map({'strongly_disagree': 0,
                                                       'disagree': 1,
                                                       'neither_agree_nor_disagree': 2,
                                                       'agree': 3,
                                                       'strongly_agree': 4})
        df['uncertainty_day'] = df['uncertainty_day'].map({'strongly_disagree': 0,
                                                           'disagree': 1,
                                                           'neither_agree_nor_disagree': 2,
                                                           'agree': 3,
                                                           'strongly_agree': 4})
        df['driving_freq'] = df['driving_freq'].map({'never': 0,
                                                     'less_than_once_a_month': 1,
                                                     'once_a_month_to_once_a_week': 2,
                                                     '1_to_3_days_a_week': 3,
                                                     '4_to_6_days_a_week': 4,
                                                     'every_day': 5,
                                                     'i_prefer_not_to_respond': np.nan})
        df['attitude_ad'] = df['attitude_ad'].map({'very_negative': 0,
                                                   'negative': 1,
                                                   'neutral': 2,
                                                   'positive': 3,
                                                   'very_positive': 4})
        df['gender'] = df['gender'].map({'male': 0,
                                         'female': 1,
                                         'other': 2,
                                         'i_prefer_not_to_respond': np.nan})
        df['mode_transportation'] = df['mode_transportation'].map({'private_vehicle': 0,
                                                                   'public_transportation': 1,
                                                                   'motorcycle': 2,
                                                                   'walkingcycling': 3,
                                                                   'other': 4,
                                                                   'i_prefer_not_to_respond': np.nan})
        df['experience_ad'] = df['experience_ad'].map({'i_have_encountered_automated_cars_while_driving': 0,
                                                       'i_have_experience_with_driving_a_car_with_automated_functions': 1,  # noqa: E501
                                                       'i_have_experience_with_driving_a_car_with_automated_functionsi_have_encountered_automated_cars_while_driving': 2,  # noqa: E501
                                                       'i_have_experience_with_driving_a_car_with_automated_functionsi_have_had_a_ride_in_an_automated_car_as_a_passenger': 3,  # noqa: E501
                                                       'i_have_experience_with_driving_a_car_with_automated_functionsi_have_had_a_ride_in_an_automated_car_as_a_passengeri_have_encountered_automated_cars_while_driving': 4,  # noqa: E501
                                                       'i_have_experience_with_driving_a_car_with_automated_functionsi_have_had_a_ride_in_an_automated_car_as_a_passengeri_have_encountered_automated_cars_while_drivingi_work_in_a_related_field': 5,  # noqa: E501
                                                       'i_have_experience_with_driving_a_car_with_automated_functionsi_have_had_a_ride_in_an_automated_car_as_a_passengeri_work_in_a_related_field': 6,  # noqa: E501
                                                       'i_have_experience_with_driving_a_car_with_automated_functionsother': 7,  # noqa: E501
                                                       'i_have_had_a_ride_in_an_automated_car_as_a_passenger': 8,  # noqa: E501
                                                       'i_have_had_a_ride_in_an_automated_car_as_a_passengeri_have_encountered_automated_cars_while_driving': 9,  # noqa: E501
                                                       'i_have_had_a_ride_in_an_automated_car_as_a_passengeri_have_encountered_automated_cars_while_drivingi_work_in_a_related_field': 10,  # noqa: E501
                                                       'i_have_had_a_ride_in_an_automated_car_as_a_passengeri_work_in_a_related_field': 11,  # noqa: E501
                                                       'i_have_never_used_or_been_in_automated_cars': 12,  # noqa: E501
                                                       'i_have_never_used_or_been_in_automated_carsi_have_encountered_automated_cars_while_driving': 13,  # noqa: E501
                                                       'i_have_never_used_or_been_in_automated_carsi_have_experience_with_driving_a_car_with_automated_functions': 14,  # noqa: E501
                                                       'i_have_never_used_or_been_in_automated_carsi_have_experience_with_driving_a_car_with_automated_functionsi_have_had_a_ride_in_an_automated_car_as_a_passenger': 15,  # noqa: E501
                                                       'i_have_never_used_or_been_in_automated_carsi_have_experience_with_driving_a_car_with_automated_functionsi_work_in_a_related_field': 16,  # noqa: E501
                                                       'i_have_never_used_or_been_in_automated_carsi_have_had_a_ride_in_an_automated_car_as_a_passenger': 17,  # noqa: E501
                                                       'i_have_never_used_or_been_in_automated_carsi_work_in_a_related_field': 18,  # noqa: E501
                                                       'i_work_in_a_related_field': 19,
                                                       'i_work_in_a_related_fieldother': 20,
                                                       'other': 21,
                                                       'i_prefer_not_to_respond': np.nan})
        df['capability_ad'] = df['capability_ad'].map({'a_fully_automated_car': 0,
                                                       'a_human_driver': 1,
                                                       'i_dont_know': 2,
                                                       'the_fully_automated_car_and_the_human_driver_are_equally_capable': 3})  # noqa: E501
        df['accidents'] = df['accidents'].map({'0': 0,
                                               '1': 1,
                                               '2': 2,
                                               '3': 3,
                                               '4': 4,
                                               '5': 5,
                                               'more_than_5': 6,
                                               'i_prefer_not_to_respond': np.nan})
        # calculate mean responses for individual stimuli
        df['image_0-certain_blue'] = df[['image_0-certain_blue-0', 'image_0-certain_blue-1']].mean(axis=1)
        df = df.drop(columns=['image_0-certain_blue-0', 'image_0-certain_blue-1'])
        df['image_0-uncertain_blue'] = df[['image_0-uncertain_blue-0', 'image_0-uncertain_blue-1']].mean(axis=1)
        df = df.drop(columns=['image_0-uncertain_blue-0', 'image_0-uncertain_blue-1'])
        df['image_1-certain_people'] = df[['image_1-certain_people-0', 'image_1-certain_people-1']].mean(axis=1)
        df = df.drop(columns=['image_1-certain_people-0', 'image_1-certain_people-1'])
        df['image_1-certain_white'] = df[['image_1-certain_white-0', 'image_1-certain_white-1']].mean(axis=1)
        df = df.drop(columns=['image_1-certain_white-0', 'image_1-certain_white-1'])
        df['image_1-uncertain_people'] = df[['image_1-uncertain_people-0', 'image_1-uncertain_people-1']].mean(axis=1)
        df = df.drop(columns=['image_1-uncertain_people-0', 'image_1-uncertain_people-1'])
        df['image_1-uncertain_white'] = df[['image_1-uncertain_white-0', 'image_1-uncertain_white-1']].mean(axis=1)
        df = df.drop(columns=['image_1-uncertain_white-0', 'image_1-uncertain_white-1'])
        df['image_2-certain_zebra'] = df[['image_2-certain_zebra-0', 'image_2-certain_zebra-1']].mean(axis=1)
        df = df.drop(columns=['image_2-certain_zebra-0', 'image_2-certain_zebra-1'])
        df['image_2-uncertain_zebra'] = df[['image_2-uncertain_zebra-0', 'image_2-uncertain_zebra-1']].mean(axis=1)
        df = df.drop(columns=['image_2-uncertain_zebra-0', 'image_2-uncertain_zebra-1'])
        df['image_3-certain_intersection'] = df[['image_3-certain_intersection-0', 'image_3-certain_intersection-1']].mean(axis=1)  # noqa: E501
        df = df.drop(columns=['image_3-certain_intersection-0', 'image_3-certain_intersection-1'])
        df['image_3-uncertain_intersection'] = df[['image_3-uncertain_intersection-0', 'image_3-uncertain_intersection-1']].mean(axis=1)  # noqa: E501
        df = df.drop(columns=['image_3-uncertain_intersection-0', 'image_3-uncertain_intersection-1'])
        df['video_0-driver_certain'] = df[['video_0-driver_certain-0', 'video_0-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_0-driver_certain-0', 'video_0-driver_certain-1'])
        df['video_0-driver_uncertain'] = df[['video_0-driver_uncertain-0', 'video_0-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_0-driver_uncertain-0', 'video_0-driver_uncertain-1'])
        df['video_1-driver_certain'] = df[['video_1-driver_certain-0', 'video_1-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_1-driver_certain-0', 'video_1-driver_certain-1'])
        df['video_1-driver_uncertain'] = df[['video_1-driver_uncertain-0', 'video_1-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_1-driver_uncertain-0', 'video_1-driver_uncertain-1'])
        df['video_10-ad_certain'] = df[['video_10-ad_certain-0', 'video_10-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_10-ad_certain-0', 'video_10-ad_certain-1'])
        df['video_10-ad_uncertain'] = df[['video_10-ad_uncertain-0', 'video_10-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_10-ad_uncertain-0', 'video_10-ad_uncertain-1'])
        df['video_11-ad_certain'] = df[['video_11-ad_certain-0', 'video_11-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_11-ad_certain-0', 'video_11-ad_certain-1'])
        df['video_11-ad_uncertain'] = df[['video_11-ad_uncertain-0', 'video_11-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_11-ad_uncertain-0', 'video_11-ad_uncertain-1'])
        df['video_12-ad_certain'] = df[['video_12-ad_certain-0', 'video_12-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_12-ad_certain-0', 'video_12-ad_certain-1'])
        df['video_12-ad_uncertain'] = df[['video_12-ad_uncertain-0', 'video_12-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_12-ad_uncertain-0', 'video_12-ad_uncertain-1'])
        df['video_13-ad_certain'] = df[['video_13-ad_certain-0', 'video_13-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_13-ad_certain-0', 'video_13-ad_certain-1'])
        df['video_13-ad_uncertain'] = df[['video_13-ad_uncertain-0', 'video_13-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_13-ad_uncertain-0', 'video_13-ad_uncertain-1'])
        df['video_14-ad_certain'] = df[['video_14-ad_certain-0', 'video_14-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_14-ad_certain-0', 'video_14-ad_certain-1'])
        df['video_14-ad_uncertain'] = df[['video_14-ad_uncertain-0', 'video_14-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_14-ad_uncertain-0', 'video_14-ad_uncertain-1'])
        df['video_15-ad_certain'] = df[['video_15-ad_certain-0', 'video_15-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_15-ad_certain-0', 'video_15-ad_certain-1'])
        df['video_15-ad_uncertain'] = df[['video_15-ad_uncertain-0', 'video_15-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_15-ad_uncertain-0', 'video_15-ad_uncertain-1'])
        df['video_16-ad_certain'] = df[['video_16-ad_certain-0', 'video_16-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_16-ad_certain-0', 'video_16-ad_certain-1'])
        df['video_16-ad_uncertain'] = df[['video_16-ad_uncertain-0', 'video_16-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_16-ad_uncertain-0', 'video_16-ad_uncertain-1'])
        df['video_17-ad_certain'] = df[['video_17-ad_certain-0', 'video_17-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_17-ad_certain-0', 'video_17-ad_certain-1'])
        df['video_17-ad_uncertain'] = df[['video_17-ad_uncertain-0', 'video_17-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_17-ad_uncertain-0', 'video_17-ad_uncertain-1'])
        df['video_18-ad_certain'] = df[['video_18-ad_certain-0', 'video_18-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_18-ad_certain-0', 'video_18-ad_certain-1'])
        df['video_18-ad_uncertain'] = df[['video_18-ad_uncertain-0', 'video_18-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_18-ad_uncertain-0', 'video_18-ad_uncertain-1'])
        df['video_19-ad_certain'] = df[['video_19-ad_certain-0', 'video_19-ad_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_19-ad_certain-0', 'video_19-ad_certain-1'])
        df['video_19-ad_uncertain'] = df[['video_19-ad_uncertain-0', 'video_19-ad_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_19-ad_uncertain-0', 'video_19-ad_uncertain-1'])
        df['video_2-driver_certain'] = df[['video_2-driver_certain-0', 'video_2-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_2-driver_certain-0', 'video_2-driver_certain-1'])
        df['video_2-driver_uncertain'] = df[['video_2-driver_uncertain-0', 'video_2-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_2-driver_uncertain-0', 'video_2-driver_uncertain-1'])
        df['video_3-driver_certain'] = df[['video_3-driver_certain-0', 'video_3-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_3-driver_certain-0', 'video_3-driver_certain-1'])
        df['video_3-driver_uncertain'] = df[['video_3-driver_uncertain-0', 'video_3-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_3-driver_uncertain-0', 'video_3-driver_uncertain-1'])
        df['video_4-driver_certain'] = df[['video_4-driver_certain-0', 'video_4-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_4-driver_certain-0', 'video_4-driver_certain-1'])
        df['video_4-driver_uncertain'] = df[['video_4-driver_uncertain-0', 'video_4-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_4-driver_uncertain-0', 'video_4-driver_uncertain-1'])
        df['video_5-driver_certain'] = df[['video_5-driver_certain-0', 'video_5-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_5-driver_certain-0', 'video_5-driver_certain-1'])
        df['video_5-driver_uncertain'] = df[['video_5-driver_uncertain-0', 'video_5-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_5-driver_uncertain-0', 'video_5-driver_uncertain-1'])
        df['video_6-driver_certain'] = df[['video_6-driver_certain-0', 'video_6-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_6-driver_certain-0', 'video_6-driver_certain-1'])
        df['video_6-driver_uncertain'] = df[['video_6-driver_uncertain-0', 'video_6-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_6-driver_uncertain-0', 'video_6-driver_uncertain-1'])
        df['video_7-driver_certain'] = df[['video_7-driver_certain-0', 'video_7-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_7-driver_certain-0', 'video_7-driver_certain-1'])
        df['video_7-driver_uncertain'] = df[['video_7-driver_uncertain-0', 'video_7-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_7-driver_uncertain-0', 'video_7-driver_uncertain-1'])
        df['video_8-driver_certain'] = df[['video_8-driver_certain-0', 'video_8-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_8-driver_certain-0', 'video_8-driver_certain-1'])
        df['video_8-driver_uncertain'] = df[['video_8-driver_uncertain-0', 'video_8-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_8-driver_uncertain-0', 'video_8-driver_uncertain-1'])
        df['video_9-driver_certain'] = df[['video_9-driver_certain-0', 'video_9-driver_certain-1']].mean(axis=1)
        df = df.drop(columns=['video_9-driver_certain-0', 'video_9-driver_certain-1'])
        df['video_9-driver_uncertain'] = df[['video_9-driver_uncertain-0', 'video_9-driver_uncertain-1']].mean(axis=1)
        df = df.drop(columns=['video_9-driver_uncertain-0', 'video_9-driver_uncertain-1'])
        # calculate mean responses for (un-)certainty
        df['certainty'] = df[[col for col in df.columns if '-certain_' in col]].mean(axis=1)
        df['uncertainty'] = df[[col for col in df.columns if '-uncertain_' in col]].mean(axis=1)
        # set nan to -1
        df = df.fillna(-1)
        # # create correlation matrix for all data
        analysis.corr_matrix(df, columns_drop=columns_drop, save_file=True, filename='_corr_matrix_all_data.jpg',
                             figsize=(100, 70))
        # create correlation matrix for all data
        analysis.scatter_matrix(df,
                                columns_drop=columns_drop,
                                diagonal_visible=False,
                                save_file=True,
                                filename='scatter_matrix_all_data')
        # stimulus duration
        analysis.hist(heroku_data,
                      x=heroku_data.columns[heroku_data.columns.to_series().str.contains('-dur')],
                      color='country',
                      nbins=100,
                      pretty_text=True,
                      save_file=True)
        # browser window dimensions
        analysis.scatter(heroku_data,
                         x='window_width',
                         y='window_height',
                         color='browser_name',
                         pretty_text=True,
                         save_file=True)
        # time of participation
        time = appen_data.copy()
        time['country'] = time['country'].fillna('NaN')
        time['time'] = time['time'] / 60.0  # convert to min
        analysis.hist(time,
                      x=['time'],
                      color='country',
                      pretty_text=True,
                      # marginal='box',
                      save_file=True)
        # 1. "Which option(s) best describe(s) your experience with automated cars?""
        # store counts for individual options of the checkbox
        counts = {'I have never used or been in automated cars': 0,
                  'I have experience with driving a car with automated functions': 0,
                  'I have had a ride in an automated car as a passenger': 0,
                  'I have encountered automated cars while driving': 0,
                  'I work in a related field': 0,
                  'Other': 0,
                  'I prefer not to respond': 0
                  }
        # loop through all pp
        for index, row in appen_data.iterrows():
            try:
                if 'i_have_never_used_or_been_in_automated_cars' in row['experience_ad']:
                    counts['I have never used or been in automated cars'] += 1
                if 'i_have_experience_with_driving_a_car_with_automated_functions' in row['experience_ad']:
                    counts['I have experience with driving a car with automated functions'] += 1
                if 'i_have_had_a_ride_in_an_automated_car_as_a_passenger' in row['experience_ad']:
                    counts['I have had a ride in an automated car as a passenger'] += 1
                if 'i_have_encountered_automated_cars_while_driving' in row['experience_ad']:
                    counts['I have encountered automated cars while driving'] += 1
                if 'i_work_in_a_related_field' in row['experience_ad']:
                    counts['I work in a related field'] += 1
                if 'other' in row['experience_ad']:
                    counts['Other'] += 1
                if 'i_prefer_not_to_respond' in row['experience_ad']:
                    counts['I prefer not to respond'] += 1
            except TypeError:
                continue
        counts_df = pd.DataFrame(counts.items(), columns=['option', 'count'])
        counts_df = counts_df.set_index('option')
        analysis.bar(counts_df,
                     # x='option',
                     y=['count'],
                     yaxis_title='Number of participations who chose the option',
                     pretty_text=True,
                     save_file=True)
        # 2. Please indicate your general attitude towards fully automated cars.
        analysis.hist(appen_data,
                      x=['attitude_ad'],
                      yaxis_title='Number of participations who chose the option',
                      color='capability_ad',
                      pretty_text=True,
                      marginal=None,
                      save_file=True)
        logger.info('Please indicate your general attitude towards fully automated cars?: M={:.2f}, SD={:.2f},'
                    + ' Median={:.2f}.',
                    df['attitude_ad'].mean(), df['attitude_ad'].std(), df['attitude_ad'].median())
        # 3. Who do you think is more capable of conducting driving-related tasks?
        analysis.hist(appen_data,
                      x=['capability_ad'],
                      yaxis_title='Number of participations who chose the option',
                      color='attitude_ad',
                      pretty_text=True,
                      marginal=None,
                      save_file=True)
        # scatter attitude_ad and capability_ad
        logger.info('Who do you think is more capable of conducting driving-related tasks?: M={:.2f}, SD={:.2f},'
                    + ' Median={:.2f}.',
                    df['capability_ad'].mean(), df['capability_ad'].std(), df['capability_ad'].median())
        # 4. In my day-to-day life, I often experience the feeling of (un)certainty.
        # 5. I often have feelings of (un)certainty about myself.
        # 6. I often experience the feeling of (un)certainty when making decisions.
        # 7. During new experiences, I often experience the feeling of (un)certainty.
        analysis.hist(appen_data,
                      x=['uncertainty_day',
                         'uncertainty_myself',
                         'uncertainty_decisions',
                         'uncertainty_experiences',
                         'certainty_day',
                         'certainty_myself',
                         'certainty_decisions',
                         'certainty_experiences'
                         ],
                      yaxis_title='Number of participations who chose the option',
                      color='attitude_ad',
                      pretty_text=True,
                      marginal=None,
                      save_file=True)
        logger.info('In my day-to-day life, I often experience the feeling of certainty: M={:.2f}, SD={:.2f},'
                    + ' Median={:.2f}.',
                    df['certainty_day'].mean(), df['certainty_day'].std(), df['certainty_day'].median())
        logger.info('I often have feelings of certainty about myself: M={:.2f}, SD={:.2f}, Median={:.2f}.',
                    df['certainty_myself'].mean(), df['certainty_myself'].std(), df['certainty_myself'].median())
        logger.info('I often have feelings of certainty about myself: M={:.2f}, SD={:.2f}, Median={:.2f}.',
                    df['certainty_decisions'].mean(),
                    df['certainty_decisions'].std(),
                    df['certainty_decisions'].median())
        logger.info('During new experiences, I often experience the feeling of certainty: M={:.2f}, SD={:.2f}, '
                    + 'Median={:.2f}.', df['certainty_experiences'].mean(), df['certainty_experiences'].std(),
                    df['certainty_experiences'].median())
        logger.info('In my day-to-day life, I often experience the feeling of certainty: M={:.2f}, SD={:.2f},'
                    + ' Median={:.2f}.', df['certainty_day'].mean(), df['certainty_day'].std(),
                    df['certainty_day'].median())
        logger.info('I often have feelings of certainty about myself: M={:.2f}, SD={:.2f}, Median={:.2f}.',
                    df['certainty_myself'].mean(), df['certainty_myself'].std(), df['certainty_myself'].median())
        logger.info('I often have feelings of certainty about myself: M={:.2f}, SD={:.2f}, Median={:.2f}.',
                    df['certainty_decisions'].mean(), df['certainty_decisions'].std(),
                    df['certainty_decisions'].median())
        logger.info('During new experiences, I often experience the feeling of certainty: M={:.2f}, SD={:.2f}, '
                    + 'Median={:.2f}.', df['certainty_experiences'].mean(), df['certainty_experiences'].std(),
                    df['certainty_experiences'].median())
        # 8. At which age did you obtain your first license for driving a car?
        analysis.hist(appen_data,
                      x=['year_license'],
                      pretty_text=True,
                      yaxis_title='Number of participations who chose the option',
                      color='attitude_ad',
                      marginal=None,
                      save_file=True)
        logger.info('Age of obtaining first driving license: M={:.2f}, SD={:.2f}, Median={:.2f}.',
                    df['year_license'].mean(), df['year_license'].std(), df['year_license'].median())
        # 9. On average, how often did you drive a vehicle in the last 12 months?
        analysis.hist(appen_data,
                      x=['driving_freq'],
                      yaxis_title='Number of participations who chose the option',
                      pretty_text=True,
                      color='attitude_ad',
                      marginal=None,
                      save_file=True)
        logger.info('On average, how often did you drive a vehicle in the last 12 months?: M={:.2f}, SD={:.2f}, '
                    + 'Median={:.2f}.', df['driving_freq'].mean(), df['driving_freq'].std(),
                    df['driving_freq'].median())
        # 10. About how many kilometers (miles) did you drive in the last 12 months?
        analysis.hist(appen_data,
                      x=['milage'],
                      yaxis_title='Number of participations who chose the option',
                      pretty_text=True,
                      color='attitude_ad',
                      marginal=None,
                      save_file=True)
        logger.info('About how many kilometers (miles) did you drive in the last 12 months?: M={:.2f}, SD={:.2f}, '
                    + 'Median={:.2f}.', df['milage'].mean(), df['milage'].std(), df['milage'].median())
        # 11. How many accidents were you involved in when driving a car in the last 3 years?
        analysis.hist(appen_data,
                      x=['accidents'],
                      yaxis_title='Number of participations who chose the option',
                      pretty_text=True,
                      color='attitude_ad',
                      marginal=None,
                      save_file=True)
        logger.info('How many accidents were you involved in when driving a car in the last 3 years?: M={:.2f}, '
                    + 'SD={:.2f}, Median={:.2f}.', df['accidents'].mean(), df['accidents'].std(),
                    df['accidents'].median())
        # histogram for the input device
        analysis.hist(appen_data,
                      x=['device'],
                      pretty_text=True,
                      color='country',
                      marginal=None,
                      save_file=True)
        # histogram for the place of participant
        analysis.hist(appen_data,
                      x=['place'],
                      pretty_text=True,
                      color='country',
                      marginal=None,
                      save_file=True)
        # grouped barchart of DBQ data
        analysis.hist(appen_data,
                      x=['dbq1_anger',
                         'dbq2_speed_motorway',
                         'dbq3_speed_residential',
                         'dbq4_headway',
                         'dbq5_traffic_lights',
                         'dbq6_horn',
                         'dbq7_mobile'],
                      marginal='violin',
                      yaxis_title='Number of participations who chose the option',
                      pretty_text=True,
                      color='attitude_ad',
                      save_file=True)
        # Post-trial questions on (un)certainty
        analysis.bar(mapping,
                     y=['mean'],
                     show_all_xticks=True,
                     xaxis_title='Stimulus',
                     yaxis_title='Mean of response on questions on (un)certainty of stimulus',
                     save_file=True)
        # mean uncertainty score vs certainty score
        # todo: add separate columns for AV and MDV uncertainty
        analysis.scatter(df,
                         x='uncertainty',
                         y='certainty',
                         trendline='ols',
                         hover_data=['experience_ad',
                                     'attitude_ad',
                                     'milage',
                                     'year_license',
                                     'license',
                                     'accidents'
                                     ],
                         color='attitude_ad',
                         pretty_text=True,
                         save_file=True)
        # mean uncertainty score vs capability_ad
        analysis.scatter(df,
                         x='uncertainty',
                         y='capability_ad',
                         hover_data='certainty',
                         color='year_license',
                         pretty_text=True,
                         save_file=True)
        # mean uncertainty score vs attitude_ad
        analysis.scatter(df,
                         x='uncertainty',
                         y='attitude_ad',
                         hover_data='certainty',
                         marker_size='',
                         color='age',
                         pretty_text=True,
                         save_file=True)
        # mean certainty score vs capability_ad
        analysis.scatter(df,
                         x='certainty',
                         y='capability_ad',
                         hover_data='certainty',
                         color='year_license',
                         pretty_text=True,
                         save_file=True)
        # mean certainty score vs attitude_ad
        analysis.scatter(df,
                         x='certainty',
                         y='attitude_ad',
                         hover_data='certainty',
                         marker_size='',
                         color='age',
                         pretty_text=True,
                         save_file=True)
        # map of participants
        analysis.map(countries_data, color='counts', save_file=True)
        # map of mean age per country
        analysis.map(countries_data, color='age', save_file=True)
        # map of gender per country
        analysis.map(countries_data, color='gender', save_file=True)
        # map of year of obtaining license per country
        analysis.map(countries_data, color='year_license', save_file=True)
        # map of year of automated driving per country
        analysis.map(countries_data, color='year_ad', save_file=True)
        # check if any figures are to be rendered
        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
