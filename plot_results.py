import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from celer.plot_utils import configure_plt


# RUN `benchopt run . --config config_small.yml`, then replace BENCH_NAME
# by the name of the produced results csv file.
BENCH_NAME = "benchopt_run_2022-05-09_15h34m25.csv"
FLOATING_PRECISION = 1e-11

configure_plt()
cmap = plt.get_cmap('tab10')

df = pd.read_csv("./outputs/" + BENCH_NAME, header=0, index_col=0)

solvers = df["solver_name"].unique()
datasets = df["data_name"].unique()
objectives = df["objective_name"].unique()

fontsize = 20
labelsize = 20
regex = re.compile('\[(.*?)\]')

plt.close('all')
fig, axarr = plt.subplots(
    len(datasets),
    len(objectives),
    sharex=False,
    sharey=True,
    figsize=[12, 4.8],
    constrained_layout=True)

# handle if there is only 1 dataset/objective:
if len(datasets) == 1:
    if len(objectives) == 1:
        axarr = np.atleast_2d(axarr)
    else:
        axarr = axarr[None, :]
elif len(objectives) == 1:
    axarr = axarr[:, None]

for idx_data, dataset in enumerate(datasets):
    df1 = df[df['data_name'] == dataset]
    for idx_obj, objective in enumerate(objectives):
        df2 = df1[df1['objective_name'] == objective]
        ax = axarr[idx_data, idx_obj]
        c_star = np.min(df2["objective_value"]) - FLOATING_PRECISION
        for i, solver_name in enumerate(solvers):
            df3 = df2[df2['solver_name'] == solver_name]
            curve = df3.groupby('stop_val').median()

            q1 = df3.groupby('stop_val')['time'].quantile(.1)
            q9 = df3.groupby('stop_val')['time'].quantile(.9)
            y = curve["objective_value"] - c_star
            color = cmap(i)
            ax.loglog(
                curve["time"], y, color=color, marker="o", markersize=3,
                label=solver_name, linewidth=3)
        # axarr[idx_data, idx_obj].set_xlim(
            # 0, dict_xlim[dataset, div_alpha])

        axarr[len(datasets)-1, idx_obj].set_xlabel(
            "Time (s)", fontsize=fontsize - 2)
        axarr[0, idx_obj].set_title(
            regex.search(objective).group(1), fontsize=fontsize - 2)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)

    axarr[idx_data, 0].set_ylabel(dataset, fontsize=fontsize)
    # axarr[idx_data, 0].set_yticks([1, 1e-7])
    # axarr[idx_data, 0].set_ylim([1e-7, 1])

fig.suptitle(regex.sub('', objective), fontsize=fontsize)
plt.show(block=False)


fig2, ax2 = plt.subplots(1, 1, figsize=(20, 4))
ncol = 3
if ncol is None:
    ncol = len(axarr[0, 0].lines)
legend = ax2.legend(ax.lines, [line.get_label() for line in ax.lines], ncol=ncol,
                    loc="upper center")
fig2.canvas.draw()
fig2.tight_layout()
legend_width = legend.get_window_extent().width
fig2.set_size_inches((legend_width // 15, legend.get_window_extent().height // 15))
plt.show(block=False)
