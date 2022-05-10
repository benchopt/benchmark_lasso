import re
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from celer.plot_utils import configure_plt


SAVEFIG = False
figname = "leukemia"

# RUN `benchopt run . --config config_small.yml`, then replace BENCH_NAME
# by the name of the produced results csv file.
# BENCH_NAME = "./dist_outputs/benchopt_run_2022-05-10_15h20m40.csv"  # leukemia
# BENCH_NAME = "./outputs/benchopt_run_2022-05-09_17h39m12.csv"  # simu 500x5k + leuk
# BENCH_NAME = "./outputs/benchopt_run_2022-05-09_18h10m13.csv"  # rcv1
# BENCH_NAME = "./outputs/benchopt_run_2022-05-09_18h23m07.csv"  # dbg
BENCH_NAME = "./outputs/benchopt_run_2022-05-10_11h29m18.csv"  # MEG 0.1 0.01
# BENCH_NAME = "benchopt_run_2022-05-10_11h48m27.csv"  # finance
FLOATING_PRECISION = 1e-11
MIN_XLIM = 1e-3
MARKERS = list(plt.Line2D.markers.keys())[:-4]

configure_plt()
cmap = plt.get_cmap('tab20')

df = pd.read_csv(BENCH_NAME, header=0, index_col=0)

solvers = df["solver_name"].unique()
solvers = np.array(sorted(solvers, key=str.lower))
datasets = df["data_name"].unique()
objectives = df["objective_name"].unique()

fontsize = 20
labelsize = 20
regex = re.compile('\[(.*?)\]')

plt.close('all')
fig1, axarr = plt.subplots(
    len(datasets),
    len(objectives),
    sharex=False,
    sharey=True,
    figsize=[12, 0.8 + 2.5 * len(datasets)],
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
        # check that at least one solver converged to compute c_star
        if df2["objective_duality_gap"].min() > FLOATING_PRECISION * df2["objective_duality_gap"].max():
            print(
                f"No solver reached a duality gap below {FLOATING_PRECISION}, "
                "cannot safely evaluate minimum objective."
            )
            continue
        c_star = np.min(df2["objective_value"]) - FLOATING_PRECISION
        for i, solver_name in enumerate(solvers):
            df3 = df2[df2['solver_name'] == solver_name]
            curve = df3.groupby('stop_val').median()

            q1 = df3.groupby('stop_val')['time'].quantile(.1)
            q9 = df3.groupby('stop_val')['time'].quantile(.9)
            y = curve["objective_value"] - c_star

            ax.loglog(
                curve["time"], y, color=cmap(i), marker=MARKERS[i], markersize=6,
                label=solver_name, linewidth=2)

        ax.set_xlim([MIN_XLIM, ax.get_xlim()[1]])
        axarr[len(datasets)-1, idx_obj].set_xlabel(
            "Time (s)", fontsize=fontsize - 2)
        axarr[0, idx_obj].set_title(
            '\n'.join(regex.search(objective).group(1).split(",")), fontsize=fontsize - 2)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)

    if regex.search(dataset) is not None:
        dataset_label = (regex.sub("", dataset) + '\n' +
                         '\n'.join(regex.search(dataset).group(1).split(',')))
    else:
        dataset_label = dataset
    axarr[idx_data, 0].set_ylabel(
        dataset_label, fontsize=fontsize - 6)
    # axarr[idx_data, 0].set_yticks([1, 1e-7])
    # axarr[idx_data, 0].set_ylim([1e-7, 1])

fig1.suptitle(regex.sub('', objective), fontsize=fontsize)
plt.show(block=False)


fig2, ax2 = plt.subplots(1, 1, figsize=(20, 4))
n_col = 3
if n_col is None:
    n_col = len(axarr[0, 0].lines)

lines_ordered = list(itertools.chain(*[ax.lines[i::n_col] for i in range(n_col)]))
legend = ax2.legend(
    lines_ordered, [line.get_label() for line in lines_ordered], ncol=n_col,
    loc="upper center")
fig2.canvas.draw()
fig2.tight_layout()
width = legend.get_window_extent().width
height = legend.get_window_extent().height
fig2.set_size_inches((width / 80,  max(height / 80, 0.5)))
plt.axis('off')
plt.show(block=False)


if SAVEFIG:
    fig1_name = f"figures/{figname}.pdf"
    fig1.savefig(fig1_name)
    os.system(f"pdfcrop {fig1_name} {fig1_name}")
    fig1.savefig(f"figures/{figname}.svg")

    fig2_name = f"figures/{figname}_legend.pdf"
    fig2.savefig(fig2_name)
    os.system(f"pdfcrop {fig2_name} {fig2_name}")
    fig2.savefig(f"figures/{figname}_legend.svg")
