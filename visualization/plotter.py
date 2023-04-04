# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch
from torch import nn
import math
import yaml
import argparse
import os


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
"""
Input:
    - One or more .csv files
    - config file defining plot specifications

Output:
    - different types of plots 
    - pernalize details within plot


Plots:
    - One table for each dataset with oracle and cross-val losses and accuracies (plus diff). Rows are models (DONE)
    - Line-chart (x-axis: cv shots, y-axis: accuracy) with 9 lines (model x measurement methods)
    - Line-chart (x-axis: cv shots, y-axis: MAE between oracle and cv) with 3 lines (model)
    - checkpoints_path/{model}/{dataset}
    - {eval_type}
GOAL: Create plots and then functions

"""

def get_MAE(a, b):
    return (a - b).abs().mean()

def get_table_data(dataframe_list):
    output = []
    for ho, cv in dataframe_list:
        row = []
        row.append(f"{cv['accuracy'].mean():.2f}")
        row.append(f"{cv['loss'].mean():.2f}")
        row.append(f"{ho['accuracy'].mean():.2f}")
        row.append(f"{ho['loss'].mean():.2f}")
        row.append(f"{get_MAE(ho['accuracy'], cv['accuracy']):.2f}")
        row.append(f"{get_MAE(ho['loss'], cv['loss']):.2f}")
        output.append(row)

    return output

def make_bold(headers):
    bold_headers = []
    for header in headers:
      bold_headers.append(f"$\\bf{{{header}}}$")
    
    return bold_headers

def create_table(data, column_headers, row_headers, filename='comparison_table2.png', figsize=(8,4), title=None):
    # Get colormaps
    rcolors = plt.cm.Greys(np.full(len(row_headers), 0.2))
    ccolors = plt.cm.Greys(np.full(len(column_headers), 0.2))

    plt.figure(linewidth=10,
            tight_layout={'pad':1},
            figsize=figsize)
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=data,
                        rowLabels=make_bold(row_headers),
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=make_bold(column_headers),
                        loc='center')
    
    the_table.scale(1, 2)

    # Hiding axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)

    plt.suptitle(title)
    plt.draw()

    fig = plt.gcf()
    plt.savefig(filename,
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=1000)

    print(f"Table successfully created!")

def get_data(data_path, config):
    shot = []
    acc_values = []
    std = []

    # oracle_data = pd.read_csv(f"{data_path}/{config['models']}-5fold-oracle.csv")
    for idx in range(config['shot_num']):
        # path = f"{data_path}/{config['models']}-{idx}fold-cross_val.csv"

        try: 
            df = pd.read_csv(data_path[idx])
        except:
            continue

        # MAE_value = get_MAE(oracle_data['loss'], df['loss'])
        shot.append(idx+1)
        acc_values.append(df['accuracy'].mean())
        std.append(df['accuracy'].std())
    
    df_dict = {'shot' : shot, 'acc' : acc_values, 'std' : std}
    sample_df = pd.DataFrame(df_dict)

    return sample_df

def create_linechart(data_path, config_dict, save_path):
    """
    data : list of DataFrames, each containing three columns (k_shot, accuracy and loss)
    config_dict : dictionary containing plot specifications
    save_path : path where plot should be saved    
    
    - Line-chart (x-axis: cv shots, y-axis: accuracy) with 9 lines (model x measurement methods)
    - Line-chart (x-axis: cv shots, y-axis: MAE between oracle and cv) with 3 lines (model)
    - checkpoints_path/{model}/{dataset}
    - {model_name}-{k_fold}fold-{eval_type}.csv
    """

    # include function to alter data here
    data = []
    for model in config_dict['models']:
        data.append(get_data(data_path[model], config_dict))
        
    print(data)

    fig = go.Figure()

    for idx, df in enumerate(data):
        fig.add_traces(go.Scatter(
        name=config_dict['models'][idx],
        x=df['shot'],
        y=df['acc'],
        mode='lines',
        line=dict(color=config_dict['color_list'][idx], dash=config_dict['dash'])))

        if config_dict['std']:
            fig.add_traces(go.Scatter(
            name='Upper Bound',
            x=df['shot'],
            y=df['acc']+df['std'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False))

            fig.add_traces(go.Scatter(
            name='Lower Bound',
            x=df['shot'],
            y=df['acc']-df['std'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor=config_dict['color_fill_list'][idx],
            fill='tonexty',
            showlegend=False))

        fig.update_layout(
        yaxis_title='Accuracy',
        xaxis_title='Shots',
        title='Accuracy comparison plot',
        hovermode="x",
        plot_bgcolor='rgba(192,192,192, 0.25)',
        autosize=False,
        width=500,
        height=300)

    print(f"Saving image...")
    fig.write_image(save_path, width=800, height=400)
    # print(fig.to_image(format="png"))
    print(f"Chart created successfully!")

def main():
    with open("test.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
        print(config)
    
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    # models = ['Baseline', 'ProtoNet', 'MAML']
    # datasets = ['FLW_Extended']
    # nway_kshots = ['5way-3shot']

    # df_list = []
    # for model in models:
    #     for dataset in datasets:
    #         for nway_kshot in nway_kshots:
    #             data_path = os.path.join(model_path, model, dataset, nway_kshot)
    #             oracle_df = pd.read_csv(f"{data_path}/{model}-15fold-oracle.csv")
    #             cv_df = pd.read_csv(f"{data_path}/{model}-15fold-cross_val.csv")
    #         df_list.append((oracle_df, cv_df))

    # column_headers = ['CV Accuracy', 'CV Loss', 'HO Accuracy', 'HO Loss', 'MAE Accuracy', 'MAE Loss']
    # data = get_table_data(df_list)
    # create_table(data, column_headers, models)


    
    config_dict = config['plot']['specifications']
    paths = {}
    for model in config_dict['models']:
        paths[model] = []
        for dataset in config_dict['datasets']:
            for shot in range(1, config_dict['shot_num']+1):
                data_path = f"{config_dict['checkpoint_dir']}/{model}/{dataset}/5way-{shot}shot/{model}-{shot*5}fold-cross_val.csv"
                paths[model].append(data_path)

    print(paths)
    # data_path = f"{config_dict['checkpoint_dir']}/{config_dict['models']}/{config_dict['datasets']}"
    # data_files = [f for f in os.path.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    create_linechart(paths, config_dict, './test.png')


if __name__ == '__main__':
    main()