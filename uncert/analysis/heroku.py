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


# todo: parse browser interactions
class Heroku:
    # pandas dataframe with extracted data
    heroku_data = pd.DataFrame()
    # pandas dataframe with mapping
    mapping = pd.read_csv(uc.common.get_configs('mapping_stimuli'))
    # resolution for keypress data
    res = uc.common.get_configs('kp_resolution')
    # number of stimuli
    num_stimuli = uc.common.get_configs('num_stimuli')
    # number of stimuli shown for each participant
    num_stimuli_participant = uc.common.get_configs('num_stimuli_participant')
    # number of repeats for each stimulus
    num_repeat = uc.common.get_configs('num_repeat')
    # allowed number of stimuli with detected wrong duration
    allowed_length = uc.common.get_configs('allowed_stimuli_wrong_duration')
    # allowed number of mistakes for questions with signs
    allowed_signs = uc.common.get_configs('allowed_mistakes_signs')
    # pickle file for saving data
    file_p = 'heroku_data.p'
    # csv file for saving data
    file_data_csv = 'heroku_data'
    # csv file for mapping of stimuli
    file_mapping_csv = 'mapping'
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
    prefixes = {'stimulus': 'video_'}
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
                        # extract name of stimulus after last slash
                        # list of stimuli. use 1st
                        if isinstance(data_cell['stimulus'], list):
                            stim_no_path = data_cell['stimulus'][0].rsplit('/', 1)[-1]  # noqa: E501
                        # single stimulus
                        else:
                            stim_no_path = data_cell['stimulus'].rsplit('/', 1)[-1]  # noqa: E501
                        # remove extension
                        stim_no_path = os.path.splitext(stim_no_path)[0]
                        # Check if it is a block with stimulus and not an
                        # instructions block
                        if (uc.common.search_dict(self.prefixes, stim_no_path)
                                is not None):
                            # stimulus is found
                            logger.debug('Found stimulus {}.', stim_no_path)
                            if self.prefixes['stimulus'] in stim_no_path:
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
                    # keypresses
                    if 'rts' in data_cell.keys() and stim_name != '':
                        # record given keypresses
                        responses = data_cell['rts']
                        logger.debug('Found {} points in keypress data.',
                                     len(responses))
                        # extract pressed keys and rt values
                        key = [point['key'] for point in responses]
                        rt = [point['rt'] for point in responses]
                        # check if values were recorded previously
                        if stim_name + '-key' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-key'] = key
                        else:
                            # previous values found
                            dict_row[stim_name + '-key'].extend(key)
                        # check if values were recorded previously
                        if stim_name + '-rt' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-rt'] = rt
                        else:
                            # previous values found
                            dict_row[stim_name + '-rt'].extend(rt)
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
                    # questions in the end
                    if 'responses' in data_cell.keys() and stim_name == '':
                        # record given keypresses
                        responses_end = data_cell['responses']
                        logger.debug('Found responses to final questions {}.',
                                     responses_end)
                        # extract pressed keys and rt values
                        responses_end = ast.literal_eval(re.search('({.+})',
                                                         responses_end).group(0))  # noqa: E501
                        # unpack questions and answers
                        questions = []
                        answers = []
                        for key, value in responses_end.items():
                            questions.append(key)
                            answers.append(value)
                        # Check if inputted values were recorded previously
                        if 'end-qs' not in dict_row.keys():
                            dict_row['end-qs'] = questions
                            dict_row['end-as'] = answers
                        else:
                            # previous values found
                            dict_row['end-qs'].extend(questions)
                            dict_row['end-as'].extend(answers)
                    # question order
                    if 'question_order' in data_cell.keys() \
                       and stim_name == '':
                        # unpack question order
                        qo_str = data_cell['question_order']
                        # remove brackets []
                        qo_str = qo_str[1:]
                        qo_str = qo_str[:-1]
                        # unpack to int
                        question_order = [int(x) for x in qo_str.split(',')]
                        logger.debug('Found question order for final ' +
                                     'questions {}.',
                                     question_order)
                        # Check if inputted values were recorded previously
                        if 'end-qo' not in dict_row.keys():
                            dict_row['end-qo'] = question_order
                        else:
                            # previous values found
                            dict_row['end-qo'].extend(question_order)
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
                            continue
                        # new value
                        if key + '-0' not in data_dict[dict_row['worker_code']].keys():  # noqa: E501
                            data_dict[dict_row['worker_code']][key + '-0'] = value  # noqa: E501
                        # update old value
                        else:
                            # traverse repetition ids untill get new repetition
                            for rep in range(0, self.num_repeat):
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
            # todo: check whith index=False is needed here
            df.to_csv(uc.settings.output_dir + '/' + self.file_data_csv +
                      '.csv', index=False)
            logger.info('Saved heroku data to csv file {}',
                        self.file_data_csv + '.csv')
        # update attribute
        self.heroku_data = df
        # return df with data
        return df

    def read_mapping(self):
        """
        Read mapping.
        """
        # read mapping from a csv file
        df = pd.read_csv(uc.common.get_configs('mapping_stimuli'))
        # set index as stimulus_id
        df.set_index('video_id', inplace=True)
        # update attribute
        self.mapping = df
        # return mapping as a dataframe
        return df

    def process_stimulus_questions(self, questions):
        """Process questions that follow each stimulus.

        Args:
            questions (list): list of questions with types of possible values
                              as int or suc.

        Returns:
            dataframe: updated mapping dataframe.
        """
        logger.info('Processing post-stimulus questions')
        # array in which arrays of video_as data is stored
        mapping_as = []
        # loop through all stimuli
        for num in tqdm(range(self.num_stimuli)):
            # calculate length of of array with answers
            length = 0
            for q in questions:
                # 1 column required for numeric data
                # numberic answer, create 1 column to store mean value
                if q['type'] == 'num':
                    length = length + 1
                # strings as answers, create columns to store counts
                elif q['type'] == 'str':
                    length = length + len(q['options'])
                else:
                    logger.error('Wrong type of data {} in question {}' +
                                 'provided.', q['type'], q['question'])
                    return -1
            # array in which data of a single stimulus is stored
            answers = [[[] for i in range(self.heroku_data.shape[0])]
                       for i in range(len(questions))]
            # for number of repetitions in survey, add extra number
            for rep in range(self.num_repeat):
                # add suffix with repetition ID
                video_as = 'video_' + str(num) + '-as-' + str(rep)
                video_order = 'video_' + str(num) + '-qs-' + str(rep)
                # loop over columns
                for col_name, col_data in self.heroku_data.items():
                    # when col_name equals video, then check
                    if col_name == video_as:
                        # loop over rows in column
                        for pp, row in enumerate(col_data):
                            # filter out empty values
                            if type(row) == list:
                                order = self.heroku_data.iloc[pp][video_order]  # noqa: E501
                                # check if injection question is present
                                if 'injection' in order:
                                    # delete injection
                                    del row[order.index('injection')]
                                    del order[order.index('injection')]
                                # loop through questions
                                for i, q in enumerate(questions):
                                    # extract answer
                                    ans = row[order.index(q['question'])]
                                    # store answer from repetition
                                    answers[i][pp].append(ans)
            # calculate mean answers from all repetitions for numeric questions
            for i, q in enumerate(questions):
                if q['type'] == 'num' and answers[i]:
                    # convert to float
                    answers[i] = [list(map(float, sublist))
                                  for sublist in answers[i]]
                    # calculate mean of mean of responses of each participant
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',
                                              category=RuntimeWarning)
                        answers[i] = np.nanmean([np.nanmean(j)
                                                 for j in answers[i]])
            # save question data in array
            mapping_as.append(answers)
        # add column with data to current mapping file
        for i, q in enumerate(questions):
            # extract answers for the given question
            q_ans = [item[i] for item in mapping_as]
            # for numeric question, add column with mean values
            if q['type'] == 'num':
                self.mapping[q['question']] = q_ans
            # for textual question, add columns with counts of each value
            else:
                # go over options and count answers with the option for each
                # stimulus
                for option in q['options']:
                    # store counts in list
                    count_option = []
                    # go over each answer
                    for ans in q_ans:
                        # flatten list of answers
                        ans = [item for sublist in ans for item in sublist]
                        # add count for answers for the given option
                        count_option.append(ans.count(option))
                    # build name of column
                    col_name = q['question'] + '-' + option.replace(' ', '_')
                    col_name = col_name.lower()
                    # add to mapping
                    self.mapping[col_name] = count_option
        # save to csv
        if self.save_csv:
            # save to csv
            self.mapping.to_csv(uc.settings.output_dir + '/' +
                                self.file_mapping_csv + '.csv')
        # return new mapping
        return self.mapping

    def filter_data(self, df):
        """
        Filter data based on the following criteria:
            1. People who had more than allowed_stimuli share of stimuli of
               unexpected length.
            2. People who made more than allowed_mistakes_signs mistakes with
               questions of traffic sign.

        Args:
            df (dataframe): dataframe with data.

        Returns:
            dataframe: updated dataframe.
        """
        # more than allowed number of mistake with codes for sentinel images
        # load mapping of codes and coordinates
        logger.info('No filtering of heroku data implemented.')
        return df

    def show_info(self):
        """
        Output info for data in object.
        """
        logger.info('No info to show.')
