# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import ast
import warnings

import uncert as uc

# warning about partial assignment
pd.options.mode.chained_assignment = None  # default='warn'

logger = uc.CustomLogger(__name__)  # use custom logger


class Heroku:
    # pandas dataframe with extracted data
    heroku_data = pd.DataFrame()
    # pandas dataframe with questions for videos
    qs_videos = pd.read_csv(uc.common.get_configs('questions_videos'))
    # pandas dataframe with questions for images
    qs_images = pd.read_csv(uc.common.get_configs('questions_images'))
    # pandas dataframe with combined information for videos and images
    mapping = None
    # number of video stimuli
    num_stimuli_video = uc.common.get_configs('num_stimuli_video')
    # number of image stimuli
    num_stimuli_img = uc.common.get_configs('num_stimuli_img')
    # number of repeated stimuli
    num_stimuli_repeat = uc.common.get_configs('num_stimuli_repeat')
    # total number of stimuli
    # number of repeats for each stimulus
    num_repeat = uc.common.get_configs('num_repeat')
    # pickle file for saving data
    file_p = 'heroku_data.p'
    # csv file for saving data
    file_data_csv = 'heroku_data.csv'
    # csv file for mapping of stimuli
    file_mapping_csv = 'mapping.csv'
    # keys with meta information
    meta_keys = ['worker_code',
                 'browser_user_agent',
                 'browser_app_name',
                 'browser_major_version',
                 'browser_full_version',
                 'browser_name',
                 'window_height',
                 'window_width',
                 'video_ids']
    # prefixes used for files in node.js implementation
    prefixes = {'video': 'video_',
                'img': 'image_'}
    # stimulus duration
    default_dur = 0

    def __init__(self,
                 files_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        # list of files with raw data
        self.files_data = files_data
        # save data as pickle file
        self.save_p = save_p
        # load data as pickle file
        self.load_p = load_p
        # save data as csv file
        self.save_csv = save_csv
        # merge questions for videos and images
        self.mapping = pd.concat([self.qs_videos, self.qs_images],
                                 ignore_index=True)

    def set_data(self, heroku_data):
        """Setter for the data object.
        """
        old_shape = self.heroku_data.shape  # store old shape for logging
        self.heroku_data = heroku_data
        logger.info('Updated heroku_data. Old shape: {}. New shape: {}.',
                    old_shape,
                    self.heroku_data.shape)

    def read_data(self, filter_data=True):
        """Read data into an attribute.

        Args:
            filter_data (bool, optional): flag for filtering data.

        Returns:
            dataframe: udpated dataframe.
        """
        # load data
        if self.load_p:
            df = uc.common.load_from_p(self.file_p,
                                       'heroku data')
        # process data
        else:
            # read files with heroku data one by one
            data_list = []
            data_dict = {}  # dictionary with data
            for file in self.files_data:
                logger.info('Reading heroku data from {}.', file)
                f = open(file, 'r')
                # add data from the file to the dictionary
                data_list += f.readlines()
                f.close()
            # hold info on previous row for worker
            prev_row_info = pd.DataFrame(columns=['worker_code',
                                                  'time_elapsed'])
            prev_row_info.set_index('worker_code', inplace=True)
            # read rows in data
            for row in tqdm(data_list):  # tqdm adds progress bar
                stim_no_path = ''
                # use dict to store data
                dict_row = {}
                # load data from a single row into a list
                list_row = json.loads(row)
                # last found stimulus
                stim_name = ''
                # trial last found stimulus
                stim_trial = -1
                # last time_elapsed for logging duration of trial
                elapsed_l = 0
                # record worker_code in the row. assuming that each row has at
                # least one worker_code
                worker_code = [d['worker_code'] for d in list_row['data'] if 'worker_code' in d][0]  # noqa: E501
                # go over cells in the row with data
                for data_cell in list_row['data']:
                    # extract meta info form the call
                    for key in self.meta_keys:
                        if key in data_cell.keys():
                            # piece of meta data found, update dictionary
                            dict_row[key] = data_cell[key]
                            if key == 'worker_code':
                                logger.debug('{}: working with row with data.',
                                             data_cell['worker_code'])
                    # check if stimulus data is present
                    if 'stimulus' in data_cell.keys():
                        # check if question detected
                        if not isinstance(data_cell['stimulus'], list):
                            warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression')  # noqa: E501
                            if (self.mapping['question'].str.contains(data_cell['stimulus']).any() and  # noqa: E501
                               len(stim_name) > 2):  # noqa: E501
                                # find short name
                                short_name = self.mapping.loc[self.mapping['question'] == data_cell['stimulus'], 'short_name'].iloc[0]  # noqa: E501
                                # check if values were recorded previously
                                dict_row[stim_name + '-' + short_name] = data_cell['response']  # noqa: E501
                                # print(dict_row[stim_name + '-' + short_name])
                        # extract name of stimulus after last slash
                        # list of stimuli. use 1st
                        if isinstance(data_cell['stimulus'], list):
                            stim_no_path = data_cell['stimulus'][0].rsplit('/', 1)[-1]  # noqa: E501
                        # single stimulus
                        else:
                            stim_no_path = data_cell['stimulus'].rsplit('/', 1)[-1]  # noqa: E501
                        # remove extension
                        stim_no_path = os.path.splitext(stim_no_path)[0]
                        # skip is videos from instructions
                        if 'video_instruction_' in stim_no_path:
                            continue
                        # Check if it is a block with stimulus and not an
                        # instructions block
                        if (uc.common.search_dict(self.prefixes, stim_no_path)
                                is not None):
                            # stimulus is found
                            logger.debug('Found stimulus {}.', stim_no_path)
                            if (self.prefixes['video'] in stim_no_path or
                               self.prefixes['img'] in stim_no_path):
                                # Record that stimulus was detected for the
                                # cells to follow
                                stim_name = stim_no_path
                                # record trial of stimulus
                                stim_trial = data_cell['trial_index']
                                # add trial duration
                                if 'time_elapsed' in data_cell.keys():
                                    # positive time elapsed from las cell
                                    if elapsed_l:
                                        time = elapsed_l
                                    # non-positive time elapsed. use value from
                                    # the known cell for worker
                                    else:
                                        time = prev_row_info.loc[worker_code, 'time_elapsed']  # noqa: E501
                                    # calculate duration
                                    dur = float(data_cell['time_elapsed']) - time  # noqa: E501
                                    if stim_name + '-dur' not in dict_row.keys() and dur > 0:  # noqa: E501
                                        # first value
                                        dict_row[stim_name + '-dur'] = dur
                    # questions after stimulus
                    if 'responses' in data_cell.keys() and stim_name != '':
                        # record given keypresses
                        responses = data_cell['responses']
                        logger.debug('Found responses to questions {}.',
                                     responses)
                        # extract pressed keys and rt values
                        responses = ast.literal_eval(re.search('({.+})',
                                                               responses).group(0))  # noqa: E501
                        # unpack questions and answers
                        questions = []
                        answers = []
                        for key, value in responses.items():
                            questions.append(key)
                            answers.append(value)
                        # check if values were recorded previously
                        if stim_name + '-qs' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-qs'] = questions
                        else:
                            # previous values found
                            dict_row[stim_name + '-qs'].extend(questions)
                        # Check if time spent values were recorded
                        # previously
                        if stim_name + '-as' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-as'] = answers
                        else:
                            # previous values found
                            dict_row[stim_name + '-as'].extend(answers)
                    # browser interaction events
                    if 'interactions' in data_cell.keys() and stim_name != '':
                        interactions = data_cell['interactions']
                        logger.debug('Found {} browser interactions.',
                                     len(interactions))
                        # extract events and timestamps
                        event = []
                        time = []
                        for interation in interactions:
                            if interation['trial'] == stim_trial:
                                event.append(interation['event'])
                                time.append(interation['time'])
                        # Check if inputted values were recorded previously
                        if stim_name + '-event' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-event'] = event
                        else:
                            # previous values found
                            dict_row[stim_name + '-event'].extend(event)
                        # check if values were recorded previously
                        if stim_name + '-time' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-time'] = time
                        else:
                            # previous values found
                            dict_row[stim_name + '-time'].extend(time)
                    # record last time_elapsed
                    if 'time_elapsed' in data_cell.keys():
                        elapsed_l = float(data_cell['time_elapsed'])
                # update last time_elapsed for worker
                prev_row_info.loc[dict_row['worker_code'], 'time_elapsed'] = elapsed_l  # noqa: E501
                # worker_code was encountered before
                if dict_row['worker_code'] in data_dict.keys():
                    # iterate over items in the data dictionary
                    for key, value in dict_row.items():
                        # worker_code does not need to be added
                        if key in self.meta_keys:
                            data_dict[dict_row['worker_code']][key] = value
                            continue
                        # new value
                        if key + '-0' not in data_dict[dict_row['worker_code']].keys():  # noqa: E501
                            data_dict[dict_row['worker_code']][key + '-0'] = value  # noqa: E501
                        # update old value
                        else:
                            # traverse repetition ids until get new repetition
                            for rep in range(0, 2):
                                # build new key with id of repetition
                                new_key = key + '-' + str(rep)
                                if new_key not in data_dict[dict_row['worker_code']].keys():  # noqa: E501
                                    data_dict[dict_row['worker_code']][new_key] = value  # noqa: E501
                                    break
                # worker_code is encountered for the first time
                else:
                    # iterate over items in the data dictionary and add -0
                    for key, value in list(dict_row.items()):
                        # worker_code does not need to be added
                        if key in self.meta_keys:
                            continue
                        # new value
                        dict_row[key + '-0'] = dict_row.pop(key)
                    # add row of data
                    data_dict[dict_row['worker_code']] = dict_row
            # turn into pandas dataframe
            df = pd.DataFrame(data_dict)
            df = df.transpose()
            # report people that attempted study
            unique_worker_codes = df['worker_code'].drop_duplicates()
            logger.info('People who attempted to participate: {}',
                        unique_worker_codes.shape[0])
            # filter data
            if filter_data:
                df = self.filter_data(df)
            # sort columns alphabetically
            df = df.reindex(sorted(df.columns), axis=1)
            # move worker_code to the front
            worker_code_col = df['worker_code']
            df.drop(labels=['worker_code'], axis=1, inplace=True)
            df.insert(0, 'worker_code', worker_code_col)
        # save to pickle
        if self.save_p:
            uc.common.save_to_p(self.file_p, df, 'heroku data')
        # save to csv
        if self.save_csv:
            df.to_csv(os.path.join(uc.settings.output_dir, self.file_data_csv),
                      index=False)
            logger.info('Saved heroku data to csv file {}',
                        self.file_data_csv + '.csv')
        # update attribute
        self.heroku_data = df
        # return df with data
        return df

    def read_questions_videos(self):
        """
        Read questions for videos.
        """
        # read information from a csv file
        df = pd.read_csv(uc.common.get_configs('questions_videos'))
        # set index as stimulus_id
        df.set_index('id', inplace=True)
        # update attribute
        self.qs_videos = df
        # return information as a dataframe
        return df

    def read_questions_images(self):
        """
        Read questions for images.
        """
        # read information from a csv file
        df = pd.read_csv(uc.common.get_configs('questions_images'))
        # set index as stimulus_id
        df.set_index('id', inplace=True)
        # update attribute
        self.qs_images = df
        # return information as a dataframe
        return df

    def process_stimulus_questions(self):
        """Process questions that follow each stimulus.

        Returns:
            dataframe: combined dataframe for all questions.
        """
        logger.info('Processing post-stimulus questions')
        # arrays for storing values for each stimulus
        raw_answers = []
        means = []
        stds = []
        medians = []
        # copy heroku data to temp df
        df = self.heroku_data
        # replace empty cells with nans
        df.replace(r'^\s*$', np.nan, regex=True)
        # loop through all stimuli
        for index, row in tqdm(self.mapping.iterrows(),
                               total=self.mapping.shape[0]):
            # extract stimulus name
            stim_no_path = row['stimulus'].rsplit('/', 1)[-1]
            stim_no_path = os.path.splitext(stim_no_path)[0]
            # build names of 2 repetitions in heroku_data
            name_0 = stim_no_path + '-' + row['short_name'] + '-0'
            name_1 = stim_no_path + '-' + row['short_name'] + '-1'
            # store mean, std, median values
            raw_answers_values = []
            mean_values = []
            std_values = []
            medians_values = []
            # calculate mean answers from all repetitions for numeric questions
            for index, row in df.iterrows():
                # print(name_0, " ", row[name_0])
                # print(name_1, " ", row[name_1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if not np.isnan(row[name_0]):
                        raw_answers_values.append(row[name_0])
                    if not np.isnan(row[name_1]):
                        raw_answers_values.append(row[name_1])
                    mean_values.append(np.nanmean([row[name_0], row[name_1]]))
                    std_values.append(np.nanstd([row[name_0], row[name_1]]))
                    medians_values.append(np.nanmedian([row[name_0], row[name_1]]))
            # calculate values for all pp
            raw_answers.append(raw_answers_values)
            means.append(np.nanmean(mean_values))
            stds.append(np.nanstd(std_values))
            medians.append(np.nanmedian(medians_values))
        # save values for all stimuli
        self.mapping['raw_answers'] = raw_answers
        self.mapping['mean'] = means
        self.mapping['std'] = stds
        self.mapping['median'] = medians
        # save to csv
        if self.save_csv:
            # save to csv
            self.mapping.to_csv(os.path.join(uc.settings.output_dir,
                                             self.file_mapping_csv))
        # return new mapping
        return self.mapping

    def filter_data(self, df):
        """
        Filter data.

        Args:
            df (dataframe): dataframe with data.

        Returns:
            dataframe: updated dataframe.
        """
        logger.info('No filtering of heroku data implemented.')
        return df

    def show_info(self):
        """
        Output info for data in object.
        """
        logger.info('No info to show.')
