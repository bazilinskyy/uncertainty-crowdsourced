# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers

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
# SHOW_OUTPUT = True  # should figures be plotted

# for debugging, skip processing
SAVE_P = False  # save pickle files with data
LOAD_P = True  # load pickle files with data
SAVE_CSV = False  # load csv files with data
FILTER_DATA = False  # filter Appen and heroku data
CLEAN_DATA = False  # clean Appen data
REJECT_CHEATERS = False  # reject cheaters on Appen
UPDATE_MAPPING = False  # update mapping with keypress data
SHOW_OUTPUT = True  # should figures be plotted

file_mapping = 'mapping.p'  # file to save updated mapping

if __name__ == '__main__':
    # create object for working with heroku data
    files_heroku = uc.common.get_configs('files_heroku')
    heroku = uc.analysis.Heroku(files_data=files_heroku,
                                save_p=SAVE_P,
                                load_p=LOAD_P,
                                save_csv=SAVE_CSV)
    # read heroku data
    heroku_data = heroku.read_data(filter_data=FILTER_DATA)
    # create object for working with appen data
    file_appen = uc.common.get_configs('file_appen')
    appen = uc.analysis.Appen(file_data=file_appen,
                              save_p=SAVE_P,
                              load_p=LOAD_P,
                              save_csv=SAVE_CSV)
    # read appen data
    appen_data = appen.read_data(filter_data=FILTER_DATA,
                                 clean_data=CLEAN_DATA)
    # get keys in data files
    heroku_data_keys = heroku_data.keys()
    appen_data_keys = appen_data.keys()
    # flag and reject cheaters
    if REJECT_CHEATERS:
        qa = uc.analysis.QA(file_cheaters=uc.common.get_configs('file_cheaters'),  # noqa: E501
                            job_id=uc.common.get_configs('appen_job'))
        qa.reject_users()
        qa.ban_users()
    # merge heroku and appen data frames into one
    all_data = heroku_data.merge(appen_data,
                                 left_on='worker_code',
                                 right_on='worker_code')
    logger.info('Data from {} participants included in analysis.',
                all_data.shape[0])
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
        uc.common.save_to_p(file_mapping,
                            mapping,
                            'mapping of stimuli')
    else:
        mapping = uc.common.load_from_p(file_mapping,
                                        'mapping of stimuli')
    if SHOW_OUTPUT:
        # Output
        analysis = uc.analysis.Analysis()
        logger.info('Creating figures.')

        # columns to drop in correlation matrix and scatter matrix
        columns_drop = ['short_name', 'id', 'question', 'short_name',
                        'label_0', 'label_100', 'stimulus', 'video_length',
                        'description', 'comments', 'partner_video']
        # copy mapping to a temp df
        df = mapping
        # convert type of vehicle to num
        df['vehicle_type'] = df['vehicle_type'].map({'AV': 0, 'MVD': 1})
        # set nan to -1
        df = df.fillna(-1)
        # create correlation matrix
        analysis.corr_matrix(df,
                             columns_drop=columns_drop,
                             save_file=True)
        # create correlation matrix
        analysis.scatter_matrix(df,
                                columns_drop=columns_drop,
                                color='vehicle_type',
                                symbol='vehicle_type',
                                diagonal_visible=False,
                                save_file=True)
        # stimulus duration
        analysis.hist(heroku_data,
                      x=heroku_data.columns[heroku_data.columns.to_series().str.contains('-dur')],  # noqa: E501
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
        df = appen_data
        df['country'] = df['country'].fillna('NaN')
        df['time'] = df['time'] / 60.0  # convert to min
        analysis.hist(df,
                      x=['time'],
                      color='country',
                      pretty_text=True,
                      save_file=True)
        # questions about AD
        analysis.scatter(appen_data,
                         x='capability_ad',
                         y='experience_ad',
                         color='year_license',
                         pretty_text=True,
                         save_file=True)
        # questions about AD
        analysis.scatter(appen_data,
                         x='attitude_ad',
                         y='experience_ad',
                         color='age',
                         pretty_text=True,
                         save_file=True)
        # histogram for driving frequency
        analysis.hist(appen_data,
                      x=['driving_freq'],
                      pretty_text=True,
                      save_file=True)
        # histogram for the year of license
        analysis.hist(appen_data,
                      x=['year_license'],
                      pretty_text=True,
                      save_file=True)
        # histogram for the mode of transportation
        analysis.hist(appen_data,
                      x=['mode_transportation'],
                      pretty_text=True,
                      save_file=True)
        # histogram for the input device
        analysis.hist(appen_data,
                      x=['device'],
                      pretty_text=True,
                      save_file=True)
        # histogram for mileage
        analysis.hist(appen_data,
                      x=['milage'],
                      pretty_text=True,
                      save_file=True)
        # histogram for the place of participant
        analysis.hist(appen_data,
                      x=['place'],
                      pretty_text=True,
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
                      pretty_text=True,
                      save_file=True)
        # post-trial questions
        analysis.bar(mapping,
                     y=['mean'],
                     show_all_xticks=True,
                     xaxis_title='Stimulus',
                     yaxis_title='Mean',
                     save_file=True)
        # # scatter plot of risk / eye contact without traffic rules involved
        # analysis.scatter(mapping[(mapping['cross_look'] != 'notCrossing_Looking') &  # noqa: E501
        #                          (mapping['cross_look'] != 'notCrossing_notLooking') &  # noqa: E501
        #                          (mapping['velocity_risk'] != 'No velocity data found')],  # noqa: E501
        #                  x='EC_score',
        #                  y='risky_slider',
        #                  # color='traffic_rules',
        #                  trendline='ols',
        #                  hover_data=['risky_slider',
        #                              'EC_score',
        #                              'EC_mean',
        #                              'EC-yes',
        #                              'EC-yes_but_too_late',
        #                              'EC-no',
        #                              'EC-i_don\'t_know',
        #                              'cross_look',
        #                              'traffic_rules'],
        #                  # pretty_text=True,
        #                  xaxis_title='Eye contact score '
        #                              + '(No=0, Yes but too late=0.25, Yes=1)',  # noqa: E501
        #                  yaxis_title='The riskiness of behaviour in video'
        #                              + ' (0-100)',
        #                  # xaxis_range=[-10, 100],
        #                  # yaxis_range=[-1, 20],
        #                  save_file=True)
        # # scatter of velocity vs eye contact
        # analysis.scatter(mapping[(mapping['cross_look'] != 'notCrossing_Looking') &  # noqa: E501
        #                          (mapping['cross_look'] != 'notCrossing_notLooking') &  # noqa: E501
        #                          (mapping['velocity_risk'] != 'No velocity data found')],  # noqa: E501
        #                  x='velocity_risk',
        #                  y='risky_slider',
        #                  # color='traffic_rules',
        #                  trendline='ols',
        #                  hover_data=['risky_slider',
        #                              'EC_score',
        #                              'EC_mean',
        #                              'EC-yes',
        #                              'EC-yes_but_too_late',
        #                              'EC-no',
        #                              'EC-i_don\'t_know',
        #                              'cross_look',
        #                              'traffic_rules'],
        #                  # pretty_text=True,
        #                  xaxis_title='Velocity (avg) at keypresses',
        #                  yaxis_title='The riskiness of behaviour in video'
        #                              + ' (0-100)',
        #                  # xaxis_range=[-10, 100],
        #                  # yaxis_range=[-1, 20],
        #                  save_file=True)
        # # scatter plot of risk and eye contact without traffic rules involved
        # analysis.scatter_mult(mapping,
        #                       x=['EC-yes',
        #                          'EC-yes_but_too_late',
        #                          'EC-no',
        #                          'EC-i_don\'t_know'],
        #                       y='risky_slider',
        #                       trendline='ols',
        #                       hover_data=['risky_slider',
        #                                   'EC-yes',
        #                                   'EC-yes_but_too_late',
        #                                   'EC-no',
        #                                   'EC-i_don\'t_know',
        #                                   'cross_look',
        #                                   'traffic_rules'],
        #                       xaxis_title='Subjective eye contact (n)',
        #                       yaxis_title='Mean risk slider (0-100)',
        #                       marginal_x='rug',
        #                       marginal_y=None,
        #                       save_file=True)
        # # scatter plot of risk and percentage of participants indicating eye
        # # contact
        # analysis.scatter_mult(mapping,
        #                       x=['EC-yes',
        #                           'EC-yes_but_too_late',
        #                           'EC-no',
        #                           'EC-i_don\'t_know'],
        #                       y='avg_kp',
        #                       trendline='ols',
        #                       xaxis_title='Percentage of participants indicating eye contact (%)',  # noqa: E501
        #                       yaxis_title='Mean keypresses (%)',
        #                       marginal_y=None,
        #                       marginal_x='rug',
        #                       save_file=True)
        # # todo: add comment
        # analysis.scatter(mapping[mapping['avg_dist'] != ''],  # noqa: E501
        #                  x='avg_dist',
        #                  y='risky_slider',
        #                  trendline='ols',
        #                  xaxis_title='Mean distance to pedestrian (m)',
        #                  yaxis_title='Mean risk slider (0-100)',
        #                  marginal_x='rug',
        #                  save_file=True)
        # # todo: add comment
        # analysis.scatter(mapping[mapping['avg_dist'] != ''],  # noqa: E501
        #                  x='avg_velocity',
        #                  y='avg_kp',
        #                  trendline='ols',
        #                  xaxis_title='Mean speed of the vehicle (km/h)',
        #                  yaxis_title='Mean risk slider (0-100)',
        #                  marginal_x='rug',
        #                  save_file=True)
        # # todo: add comment
        # analysis.scatter_mult(mapping[mapping['avg_person'] != ''],     # noqa: E501
        #                       x=['avg_object', 'avg_person', 'avg_car'],
        #                       y='risky_slider',
        #                       trendline='ols',
        #                       xaxis_title='Object count',
        #                       yaxis_title='Mean risk slider (0-100)',
        #                       marginal_y=None,
        #                       marginal_x='rug',
        #                       save_file=True)
        # # todo: add comment
        # analysis.scatter_mult(mapping[mapping['avg_person'] != ''],     # noqa: E501
        #                       x=['avg_object', 'avg_person', 'avg_car'],
        #                       y='avg_kp',
        #                       trendline='ols',
        #                       xaxis_title='Object count',
        #                       yaxis_title='Mean keypresses (%)',
        #                       marginal_y=None,
        #                       marginal_x='rug',
        #                       save_file=True)
        # # todo: add comment
        # analysis.scatter(mapping[mapping['avg_obj_surface'] != ''],    # noqa: E501
        #                  x='avg_obj_surface',
        #                  y='avg_kp',
        #                  trendline='ols',
        #                  xaxis_title='Mean object surface (0-1)',
        #                  yaxis_title='Mean keypresses (%)',
        #                  marginal_x='rug',
        #                  save_file=True)
        # # todo: add comment
        # analysis.scatter(mapping[mapping['avg_obj_surface'] != ''],    # noqa: E501
        #                  x='avg_obj_surface',
        #                  y='risky_slider',
        #                  trendline='ols',
        #                  xaxis_title='Mean object surface (0-1)',
        #                  yaxis_title='Mean risk slider (0-100)',
        #                  marginal_x='rug',
        #                  save_file=True)
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
