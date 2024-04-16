# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import warnings
import unicodedata
import re

import uncert as uc

matplotlib.use('TkAgg')
logger = uc.CustomLogger(__name__)  # use custom logger


class Analysis:
    # set template for plotly output
    template = uc.common.get_configs('plotly_template')
    # number of video stimuli
    num_stimuli_video = uc.common.get_configs('num_stimuli_video')
    # number of image stimuli
    num_stimuli_img = uc.common.get_configs('num_stimuli_img')
    # number of repeated stimuli
    num_stimuli_repeat = uc.common.get_configs('num_stimuli_repeat')
    # folder for output
    folder = '/figures/'

    def __init__(self, save_csv: bool):
        # set font to Times
        plt.rc('font', family='serif')
        # save data as csv file
        self.save_csv = save_csv

    def corr_matrix(self, df, columns_drop, save_file=False, filename='_corr_matrix.jpg', figsize=(34, 20)):
        """
        Output correlation matrix.

        Args:
            df (dataframe): mapping dataframe.
            columns_drop (list): columns dataframes in to ignore.
            save_file (bool, optional): flag for saving an html file with plot.
            filename (str, optional): name of file to save.
            filename (list, optional): size of figure.
        """
        logger.info('Creating correlation matrix.')
        # drop columns
        df = df.drop(columns=columns_drop)
        # save to csv
        if self.save_csv:
            df.to_csv(os.path.join(uc.settings.output_dir, 'all_data.csv'), index=False)
        # create correlation matrix
        corr = df.corr()
        # create mask
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        # set larger font
        vs_font = 10  # very small
        s_font = 12   # small
        m_font = 16   # medium
        l_font = 18   # large
        plt.rc('font', size=s_font)         # controls default text sizes
        plt.rc('axes', titlesize=s_font)    # fontsize of the axes title
        plt.rc('axes', labelsize=s_font)   # fontsize of the axes labels
        plt.rc('xtick', labelsize=vs_font)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=vs_font)   # fontsize of the tick labels
        plt.rc('legend', fontsize=s_font)   # fontsize of the legend
        plt.rc('figure', titlesize=l_font)  # fontsize of the figure title
        plt.rc('axes', titlesize=m_font)    # fontsize of the subplot title
        # create figure
        fig = plt.figure(figsize=figsize)
        g = sns.heatmap(corr,
                        annot=True,
                        mask=mask,
                        cmap='coolwarm',
                        fmt=".2f")
        # rotate ticks
        for item in g.get_xticklabels():
            item.set_rotation(55)
        # save image
        if save_file:
            self.save_fig('all',
                          fig,
                          self.folder,
                          filename,
                          pad_inches=0.05)
        # revert font
        self.reset_font()

    def scatter_matrix(self, df, columns_drop, color=None, symbol=None,  diagonal_visible=False, xaxis_title=None,
                       yaxis_title=None, save_file=False, filename='scatter_matrix'):
        """
        Output scatter matrix.

        Args:
            df (dataframe): mapping dataframe.
            columns_drop (list): columns dataframes in to ignore.
            color (str, optional): dataframe column to assign color of points.
            symbol (str, optional): dataframe column to assign symbol of
                                    points.
            diagonal_visible (bool, optional): show/hide diagonal with
                                               correlation==1.0.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            save_file (bool, optional): flag for saving an html file with plot.
            filename (str, optional): name of file to save.
        """
        logger.info('Creating scatter matrix.')
        # drop columns
        df = df.drop(columns=columns_drop)
        # create dimensions list after dropping columns
        dimensions = df.keys()
        # plot matrix
        fig = px.scatter_matrix(df,
                                dimensions=dimensions,
                                color=color,
                                symbol=symbol)
        # update layout
        fig.update_layout(template=self.template,
                          width=5000,
                          height=5000,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # hide diagonal
        if not diagonal_visible:
            fig.update_traces(diagonal_visible=False)
        # save file
        if save_file:
            self.save_plotly(fig, filename, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def bar(self, df, y: list, x=None, stacked=False, pretty_text=False,
            orientation='v', xaxis_title=None, yaxis_title=None,
            show_all_xticks=False, show_all_yticks=False,
            show_text_labels=False, save_file=False):
        """
        Barplot for questionnaire data. Passing a list with one variable will
        output a simple barplot; passing a list of variables will output a
        grouped barplot.

        Args:
            df (dataframe): dataframe with data from appen.
            x (list): values in index of dataframe to plot for. If no value is
                      given, the index of df is used.
            y (list): column names of dataframe to plot.
            stacked (bool, optional): show as stacked chart.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitilisng each value.
            orientation (str, optional): orientation of bars. v=vertical,
                                         h=horizontal.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            show_all_xticks (bool, optional): show all ticks on x axis.
            show_all_yticks (bool, optional): show all ticks on y axis.
            show_text_labels (bool, optional): output automatically positionsed
                                               text labels.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating bar chart for x={} and y={}', x, y)
        # prettify text
        if pretty_text:
            for variable in y:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()
        # use index of df if no is given
        if not x:
            x = df.index
        # create figure
        fig = go.Figure()
        # go over variables to plot
        for variable in y:
            # showing text labels
            if show_text_labels:
                text = df[variable]
            else:
                text = None
            # plot variable
            fig.add_trace(go.Bar(x=x,
                                 y=df[variable],
                                 name=variable,
                                 orientation=orientation,
                                 text=text,
                                 textposition='auto'))
        # add tabs if multiple variables are plotted
        if len(y) > 1:
            fig.update_layout(barmode='group')
            buttons = list([dict(label='All',
                                 method='update',
                                 args=[{'visible': [True] * df[y].shape[0]},
                                       {'title': 'All',
                                        'showlegend': True}])])
            # counter for traversing through stimuli
            counter_rows = 0
            for variable in y:
                visibility = [[counter_rows == j] for j in range(len(y))]
                visibility = [item for sublist in visibility for item in sublist]  # type: ignore # noqa: E501
                button = dict(label=variable,
                              method='update',
                              args=[{'visible': visibility},
                                    {'title': variable}])
                buttons.append(button)
                counter_rows = counter_rows + 1
            updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
            fig['layout']['updatemenus'] = updatemenus
            fig['layout']['title'] = 'All'
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # format text labels
        if show_text_labels:
            fig.update_traces(texttemplate='%{text:.2s}')
        # show all ticks on x axis
        if show_all_xticks:
            fig.update_layout(xaxis=dict(dtick=1))
        # show all ticks on x axis
        if show_all_yticks:
            fig.update_layout(yaxis=dict(dtick=1))
        # stacked bar chart
        if stacked:
            fig.update_layout(barmode='stack')
        # save file
        if save_file:
            file_name = 'bar_' + '-'.join(str(val) for val in y) + '_' + \
                        '-'.join(str(val) for val in x)
            self.save_plotly(fig,
                             file_name,
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def scatter(self, df, x, y, color=None, symbol=None, size=None, text=None,
                trendline=None, hover_data=None, marker_size=None,
                pretty_text=False, marginal_x='violin', marginal_y='violin',
                xaxis_title=None, yaxis_title=None, xaxis_range=None,
                yaxis_range=None, save_file=True):
        """
        Output scatter plot of variables x and y with optional assignment of
        colour and size.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (str): dataframe column to plot on x axis.
            y (str): dataframe column to plot on y axis.
            color (str, optional): dataframe column to assign color of points.
            symbol (str, optional): dataframe column to assign symbol of
                                    points.
            size (str, optional): dataframe column to assign soze of points.
            text (str, optional): dataframe column to assign text labels.
            trendline (str, optional): trendline. Can be 'ols', 'lowess'
            hover_data (list, optional): dataframe columns to show on hover.
            marker_size (int, optional): size of marker. Should not be used
                                         together with size argument.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitilisng each value.
            marginal_x (str, optional): type of marginal on x axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating scatter plot for x={} and y={}.', x, y)
        # using size and marker_size is not supported
        if marker_size and size:
            logger.error('Arguments marker_size and size cannot be used'
                         + ' together.')
            return -1
        # using marker_size with histogram marginal(s) is not supported
        if (marker_size and
                (marginal_x == 'histogram' or marginal_y == 'histogram')):
            logger.error('Argument marker_size cannot be used together with'
                         + ' histogram marginal(s).')
            return -1
        # prettify text
        if pretty_text:
            if isinstance(df.iloc[0][x], str):  # check if string
                # replace underscores with spaces
                df[x] = df[x].str.replace('_', ' ')
                # capitalise
                df[x] = df[x].str.capitalize()
            if isinstance(df.iloc[0][y], str):  # check if string
                # replace underscores with spaces
                df[y] = df[y].str.replace('_', ' ')
                # capitalise
                df[y] = df[y].str.capitalize()
            if color and isinstance(df.iloc[0][color], str):  # check if string
                # replace underscores with spaces
                df[color] = df[color].str.replace('_', ' ')
                # capitalise
                df[color] = df[color].str.capitalize()
            if size and isinstance(df.iloc[0][size], str):  # check if string
                # replace underscores with spaces
                df[size] = df[size].str.replace('_', ' ')
                # capitalise
                df[size] = df[size].str.capitalize()
            try:
                # check if string
                if text and isinstance(df.iloc[0][text], str):
                    # replace underscores with spaces
                    df[text] = df[text].str.replace('_', ' ')
                    # capitalise
                    df[text] = df[text].str.capitalize()
            except ValueError as e:
                logger.debug('Tried to prettify {} with exception {}', text, e)
        # scatter plot with histograms
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            fig = px.scatter(df,
                             x=x,
                             y=y,
                             color=color,
                             symbol=symbol,
                             size=size,
                             text=text,
                             trendline=trendline,
                             hover_data=hover_data,
                             marginal_x=marginal_x,
                             marginal_y=marginal_y)
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # change marker size
        if marker_size:
            fig.update_traces(marker=dict(size=marker_size))
        # save file
        if save_file:
            self.save_plotly(fig,
                             'scatter_' + x + '-' + y,
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def scatter_mult(self, df, x, y, color=None, symbol=None,
                     text=None, trendline=None, hover_data=None,
                     marker_size=None, pretty_text=False, marginal_x='violin',
                     marginal_y='violin', xaxis_title=None, yaxis_title=None,
                     xaxis_range=None, yaxis_range=None, save_file=True):
        """
        Output scatter plot of multiple variables x and y with optional
        assignment of colour and size.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (str): dataframe columns to plot on x axis.
            y (str): dataframe column to plot on y axis.
            symbol (str, optional): dataframe column to assign symbol of
                                    points.
            text (str, optional): dataframe column to assign text labels.
            trendline (str, optional): trendline. Can be 'ols', 'lowess'
            hover_data (list, optional): dataframe columns to show on hover.
            marker_size (int, optional): size of marker. Should not be used
                                         together with size argument.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitilisng each value.
            marginal_x (str, optional): type of marginal on x axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        # todo: extend with multiple columns for y
        logger.info('Creating scatter plot for x={} and y={}.', x, y)
        # using marker_size with histogram marginal(s) is not supported
        if (marker_size and
                (marginal_x == 'histogram' or marginal_y == 'histogram')):
            logger.error('Argument marker_size cannot be used together with'
                         + ' histogram marginal(s).')
            return -1
        # prettify text
        if pretty_text:
            for x_col in x:
                if isinstance(df.iloc[0][x_col], str):  # check if string
                    # replace underscores with spaces
                    df[x_col] = df[x_col].str.replace('_', ' ')
                    # capitalise
                    df[x_col] = df[x_col].str.capitalize()
            if isinstance(df.iloc[0][y], str):  # check if string
                # replace underscores with spaces
                df[y] = df[y].str.replace('_', ' ')
                # capitalise
                df[y] = df[y].str.capitalize()
            try:
                # check if string
                if text and isinstance(df.iloc[0][text], str):
                    # replace underscores with spaces
                    df[text] = df[text].str.replace('_', ' ')
                    # capitalise
                    df[text] = df[text].str.capitalize()
            except ValueError as e:
                logger.debug('Tried to prettify {} with exception {}', text, e)
        # create new dataframe with the necessary data
        color = []
        val_y = []
        val_x = []
        for x_col in x:
            for index, row in df.iterrows():
                color.append(x_col)
                val_x.append(row[x_col])
                val_y.append(row[y])
        data = {'val_y': val_y,
                'color': color,
                'val_x': val_x}
        df = pd.DataFrame(data)
        # scatter plot with histograms
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            fig = px.scatter(df,
                             x='val_x',
                             y='val_y',
                             color='color',
                             symbol=symbol,
                             text=text,
                             trendline=trendline,
                             # hover_data=hover_data,
                             marginal_x=marginal_x,
                             marginal_y=marginal_y)
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range,
                          legend_title_text=' ',
                          font=dict(size=20),
                          legend=dict(orientation='h',
                                      yanchor='bottom',
                                      y=1.02,
                                      xanchor='right',
                                      x=0.78
                                      ))
        results = px.get_trendline_results(fig)
        for i in range(len(x)):
            print(results.px_fit_results.iloc[i].summary())
        # change marker size
        if marker_size:
            fig.update_traces(marker=dict(size=marker_size))
        # save file
        if save_file:
            self.save_plotly(fig,
                             'scatter_' + ','.join(x) + '-' + y,
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def hist(self, df, x, nbins=None, color=None, pretty_text=False,
             marginal='rug', xaxis_title=None, yaxis_title=None,
             show_legend=True, save_file=True):
        """
        Output histogram of time of participation.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (list): column names of dataframe to plot.
            nbins (int, optional): number of bins in histogram.
            color (str, optional): dataframe column to assign colour of bars.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitalising each value.
            marginal (str, optional): type of marginal on x axis. Can be
                                      'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            show_legend (bool, optional): showing legend.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating histogram for x={}.', x)
        # using colour with multiple values to plot not supported
        if color and len(x) > 1:
            logger.error('Color property can be used only with a single' +
                         ' variable to plot.')
            return -1
        # prettify ticks
        if pretty_text:
            for variable in x:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()
            if color and isinstance(df.iloc[0][color], str):  # check if string
                # replace underscores with spaces
                df[color] = df[color].str.replace('_', ' ')
                # capitalise
                df[color] = df[color].str.capitalize()
        # create figure
        if color:
            fig = px.histogram(df[x], nbins=nbins, marginal=marginal,
                               color=df[color])
        else:
            fig = px.histogram(df[x], nbins=nbins, marginal=marginal)
        # ticks as numbers
        fig.update_layout(xaxis=dict(tickformat='digits'))
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # show legend if more than 1 variable is outputted
        if not show_legend:
            fig.update_layout(showlegend=False)
        # save file
        if save_file:
            self.save_plotly(fig,
                             'hist_' + '-'.join(str(val) for val in x),
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def map(self, df, color, save_file=True):
        """Map of countries of participation with color based on column in
           dataframe.

        Args:
            df (dataframe): dataframe with keypress data.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating visualisation of heatmap of participants by'
                    + ' country with colour defined by {}.', color)
        # create map
        fig = px.choropleth(df,
                            locations='country',
                            color=color,
                            hover_name='country',
                            color_continuous_scale=px.colors.sequential.Plasma)
        # update layout
        fig.update_layout(template=self.template)
        # save file
        if save_file:
            self.save_plotly(fig, 'map_' + color, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def save_plotly(self, fig, name, output_subdir):
        """
        Helper function to save figure as html file.

        Args:
            fig (plotly figure): figure object.
            name (str): name of html file.
            output_subdir (str): Folder for saving file.
        """
        # build path
        path = uc.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # limit name to 255 char
        if len(path) + len(name) > 250:
            name = name[:255 - len(path) - 5]
        file_plot = os.path.join(path + name + '.html')
        # save to file
        py.offline.plot(fig, filename=file_plot)

    def save_fig(self, image, fig, output_subdir, suffix, pad_inches=0):
        """
        Helper function to save figure as file.

        Args:
            image (str): name of figure to save.
            fig (matplotlib figure): figure object.
            output_subdir (str): folder for saving file.
            suffix (str): suffix to add in the end of the filename.
            pad_inches (int, optional): padding.
        """
        # extract name of stimulus after last slash
        file_no_path = image.rsplit('/', 1)[-1]
        # remove extension
        file_no_path = os.path.splitext(file_no_path)[0]
        # turn name into valid file name
        file_no_path = self.slugify(file_no_path)
        # create path
        path = uc.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # save file
        plt.savefig(path + file_no_path + suffix,
                    bbox_inches='tight',
                    pad_inches=pad_inches)
        # clear figure from memory
        plt.close(fig)

    def autolabel(self, ax, on_top=False, decimal=True):
        """
        Attach a text label above each bar in, displaying its height.

        Args:
            ax (matplotlib axis): bas objects in figure.
            on_top (bool, optional): add labels on top of bars.
            decimal (bool, optional): add 2 decimal digits.
        """
        # todo: optimise to use the same method
        # on top of bar
        if on_top:
            for rect in ax.patches:
                height = rect.get_height()
                # show demical points
                if decimal:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{height:.0f}'
                ax.annotate(label_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center',
                            va='bottom')
        # in the middle of the bar
        else:
            # based on https://stackoverflow.com/a/60895640/46687
            # .patches is everything inside of the chart
            for rect in ax.patches:
                # Find where everything is located
                height = rect.get_height()
                width = rect.get_width()
                x = rect.get_x()
                y = rect.get_y()
                # The height of the bar is the data value and can be used as
                # the label
                # show demical points
                if decimal:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{height:.0f}'
                label_x = x + width / 2
                label_y = y + height / 2
                # plot only when height is greater than specified value
                if height > 0:
                    ax.text(label_x,
                            label_y,
                            label_text,
                            ha='center',
                            va='center')

    def reset_font(self):
        """
        Reset font to default size values. Info at
        https://matplotlib.org/tutorials/introductory/customizing.html
        """
        s_font = 8
        m_font = 10
        l_font = 12
        plt.rc('font', size=s_font)         # controls default text sizes
        plt.rc('axes', titlesize=s_font)    # fontsize of the axes title
        plt.rc('axes', labelsize=m_font)    # fontsize of the axes labels
        plt.rc('xtick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('legend', fontsize=s_font)   # legend fontsize
        plt.rc('figure', titlesize=l_font)  # fontsize of the figure title

    def get_conf_interval_bounds(self, data, conf_interval=0.95):
        """Get lower and upper bounds of confidence interval.

        Args:
            data (list): list with data.
            conf_interval (float, optional): confidence interval value.

        Returns:
            list of lsits: lower and uppoer bounds.
        """
        # calculate condidence interval
        conf_interval = st.t.interval(conf_interval,
                                      len(data) - 1,
                                      loc=np.mean(data),
                                      scale=st.sem(data))
        # calcuate bounds
        # todo: cross-check if correct
        y_lower = data - conf_interval[0]
        y_upper = data + conf_interval[1]
        return y_lower, y_upper

    def slugify(self, value, allow_unicode=False):
        """
        Taken from https://github.com/django/django/blob/master/django/utils/text.py  # noqa: E501
        Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
        dashes to single dashes. Remove characters that aren't alphanumerics,
        underscores, or hyphens. Convert to lowercase. Also strip leading and
        trailing whitespace, dashes, and underscores.
        """
        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize('NFKC', value)
        else:
            value = unicodedata.normalize('NFKD', value).encode('ascii',
                                                                'ignore').decode('ascii')  # noqa: E501
        value = re.sub(r'[^\w\s-]', '', value.lower())
        return re.sub(r'[-\s]+', '-', value).strip('-_')
