from pathlib import Path
import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib
from matplotlib.colors import ListedColormap
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from examples.plot_utils import setup_plots


def plot_region(
    ax,
    domain,
    binary_vector,
    color,
    interpolation,
    background_color="#ffffff",
    alpha=1.0,
):
    alpha = jnp.zeros_like(binary_vector, dtype=float).at[binary_vector].set(alpha)
    is_only_true = jnp.all(binary_vector)
    if is_only_true:
        cmap = ListedColormap([color])
    else:
        cmap = ListedColormap([background_color, color])
    ax.imshow(
        binary_vector,
        cmap=cmap,
        alpha=alpha,
        interpolation=interpolation,
        extent=[domain[0, 0], domain[-1, 0], domain[-1, 1], domain[0, 1]],
    )


def store_and_show_fig(
    root, fig, exp_name, name, show=False, lgd=None, small_plot=False
):
    setup_plots(matplotlib, small_plot=small_plot)
    Path(f"{root}cache/{exp_name}").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        f"{root}cache/{exp_name}/{name}.svg",
        bbox_extra_artists=None if lgd is None else (lgd,),
        bbox_inches="tight",
        format="svg",
    )
    if show:
        display(fig)


def plot_single_result(
    root,
    exp_name,
    df,
    alg_names,
    col,
    title,
    xlabel,
    ylabel,
    xlim=None,
    ylim=None,
    cumsum=False,
    save=True,
):
    series = df[[f"{name}.{col}" for name in alg_names]]
    if cumsum:
        series = series.cumsum()
    fig = (
        series.rename(columns=lambda x: x.strip(f".{col}"))
        .plot(
            ylabel=ylabel,
            xlabel=xlabel,
            ylim=ylim,
            xlim=xlim,
            title=title,
            label=alg_names,
            legend=False,
        )
        .get_figure()
    )
    fig.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save:
        store_and_show_fig(root, fig, exp_name, title)


def prepare_df(
    dfs,
    alg_names,
    col,
    cumsum=False,
    negate=False,
    T=None,
):
    cols = [f"{name}.{col}" for name in alg_names]

    dfs_concat = []
    for df in dfs:
        col_int = list(set(cols) & set(df.columns))
        if len(col_int) > 0:
            df_concat = pd.concat([df[col] for col in col_int], axis=1)
            if cumsum:
                df_concat = df_concat.cumsum()
            df_concat.columns = col_int
            dfs_concat.append(df_concat)
    df_concat = pd.concat(dfs_concat)

    grouped = df_concat.groupby(df_concat.index)
    T = T if T is not None else grouped.mean().shape[0]
    mean = grouped.mean()[:T].replace([np.inf, -np.inf], np.nan)
    if negate:
        mean = -mean
    std = grouped.std()[:T] / np.sqrt(grouped.count()[:T])
    return mean, std


def plot_result(
    root,
    exp_name,
    dfs,
    alg_names,
    col,
    title,
    xlabel,
    ylabel,
    xlim=None,
    ylim=None,
    cumsum=False,
    negate=False,
    T=None,
    legend=True,
    hide_title=False,
    save=True,
    small_plot=False,
    y_fn=None,
    bucket_size=1,
    linewidths=None,
    linestyles=None,
    markers=None,
    markevery=None,
    constlines=None,
):
    y_transform = lambda x: x if y_fn is None else y_fn(x)

    cols = [f"{name}.{col}" for name in alg_names]
    mean, std = prepare_df(dfs, alg_names, col, cumsum=cumsum, negate=negate, T=T)

    # Create bucket indices for each data point
    bucket_indices = np.floor(mean.index / bucket_size).astype(int)

    # Group the data points by bucket indices and calculate the averaged values
    grouped_mean = mean.groupby(bucket_indices).mean()
    grouped_std = std.groupby(bucket_indices).mean()

    fig, ax = plt.subplots()
    for i, col in enumerate(cols):
        # ax.errorbar(grouped_mean.index * bucket_size, grouped_mean[col], yerr=grouped_std[col], label=col)
        ax.plot(
            grouped_mean.index * bucket_size,
            y_transform(grouped_mean[col]),
            label=col.split(".")[0],
            linewidth=linewidths[i] if linewidths is not None else None,
            linestyle=linestyles[i] if linestyles is not None else None,
            marker=markers[i] if markers is not None else None,
            markevery=markevery,
        )
        ax.fill_between(
            grouped_mean.index * bucket_size,
            y_transform(grouped_mean[col] - grouped_std[col]),
            y_transform(grouped_mean[col] + grouped_std[col]),
            alpha=0.2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not hide_title:
        ax.set_title(title)
    lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5)) if legend else None

    if constlines is not None:
        for constline in constlines:
            ax.axhline(y=constline, color="black", linestyle="--")

    if save:
        store_and_show_fig(root, fig, exp_name, title, lgd=lgd, small_plot=small_plot)


def plot_advantage(
    root,
    exp_name,
    dfs,
    alg_name,
    alg_names,
    col,
    title,
    xlabel,
    ylabel,
    xlim=None,
    ylim=None,
    cumsum=False,
    T=None,
    save=True,
):
    cols = [f"{name}.{col}" for name in alg_names]
    main_col = f"{alg_name}.{col}"
    mean, std = prepare_df(dfs, alg_names, col, cumsum=cumsum, T=T)
    mean = mean.div(mean[main_col], axis=0)
    std = std.div(mean[main_col], axis=0)

    fig, ax = plt.subplots()
    for col in cols:
        # ax.errorbar(mean.index, mean[col], yerr=std[col], label=col)
        ax.plot(mean.index, mean[col], label=col.split(".")[0])
        ax.fill_between(
            mean.index, mean[col] - std[col], mean[col] + std[col], alpha=0.2
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save:
        store_and_show_fig(root, fig, exp_name, title, show=False, lgd=lgd)


def plot_advantage_aggregated(
    root,
    exp_name,
    kernel_names,
    lengthscales,
    dfs_collection,
    base_alg_name,
    other_alg_name,
    col,
    title,
    xlabel,
    ylabel,
    xlim=None,
    ylim=None,
    cumsum=False,
    T=None,
    save=True,
    plot_mean=True,
    hide_title=False,
    legend=True,
    small_plot=False,
):
    result_df = pd.DataFrame(columns=kernel_names, index=lengthscales)
    result_df_std = pd.DataFrame(columns=kernel_names, index=lengthscales)
    for kernel_name, nested_dfs_collection in zip(kernel_names, dfs_collection):
        for lengthscale, dfs in zip(lengthscales, nested_dfs_collection):
            other_col = f"{other_alg_name}.{col}"
            main_col = f"{base_alg_name}.{col}"
            mean, std = prepare_df(
                dfs, [base_alg_name, other_alg_name], col, cumsum=cumsum, T=T
            )
            mean = mean.div(mean[main_col], axis=0)
            if plot_mean:
                result_df.loc[lengthscale, kernel_name] = mean[other_col].mean()
                result_df_std.loc[lengthscale, kernel_name] = mean[other_col].std()
            else:
                result_df.loc[lengthscale, kernel_name] = mean[other_col].max()
                result_df_std.loc[lengthscale, kernel_name] = 0

    for kernel_name in kernel_names:
        plt.errorbar(
            result_df.index,
            result_df[kernel_name],
            yerr=result_df_std[kernel_name],
            linestyle="--",
            marker="x",
            label=kernel_name,
        )
    fig = plt.gcf()
    ax = fig.gca()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not hide_title:
        ax.set_title(title)
    lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5)) if legend else None

    if save:
        store_and_show_fig(
            root, fig, exp_name, title, show=False, lgd=lgd, small_plot=small_plot
        )


def plot_single_advantage_aggregated(
    root,
    exp_name,
    indices,
    dfs_collection,
    base_alg_name,
    other_alg_name,
    col,
    title,
    xlabel,
    ylabel,
    xlim=None,
    ylim=None,
    cumsum=False,
    T=None,
    save=True,
    plot_mean=True,
):
    result = []
    result_std = []
    for i, dfs in zip(indices, dfs_collection):
        other_col = f"{other_alg_name}.{col}"
        main_col = f"{base_alg_name}.{col}"
        mean, std = prepare_df(
            dfs, [base_alg_name, other_alg_name], col, cumsum=cumsum, T=T
        )
        mean = mean.div(mean[main_col], axis=0)
        if plot_mean:
            result.append(mean[other_col].mean())
            result_std.append(mean[other_col].std())
        else:
            result.append(mean[other_col].max())
            result_std.append(0)

    plt.errorbar(indices, result, yerr=result_std, linestyle="--", marker="x")
    fig = plt.gcf()
    ax = fig.gca()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    # lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save:
        store_and_show_fig(root, fig, exp_name, title, show=False)


def plot_single_result_aggregated(
    root,
    exp_name,
    indices,
    dfs_collection,
    alg_names,
    col,
    title,
    xlabel,
    ylabel,
    xlim=None,
    ylim=None,
    cumsum=False,
    T=None,
    save=True,
    plot_mean=True,
):
    result = []
    result_std = []
    for i, dfs in zip(indices, dfs_collection):
        mean, std = prepare_df(dfs, alg_names, col, cumsum=cumsum, T=T)
        if plot_mean:
            result.append(mean.mean(axis=0))
            result_std.append(mean.std(axis=0))
        else:
            result.append(mean.max(axis=0))
            result_std.append(mean.std(axis=0))
    result_df = pd.DataFrame(result)
    result_std_df = pd.DataFrame(result_std)

    fig, ax = plt.subplots()
    for col in result_df.columns:
        ax.plot(indices, result_df[col], label=col.split(".")[0])
        ax.fill_between(
            indices,
            result_df[col] - result_std_df[col],
            result_df[col] + result_std_df[col],
            alpha=0.2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save:
        store_and_show_fig(root, fig, exp_name, title, lgd=lgd, show=False)


def plot_simple_regret_convergence(
    root,
    exp_name,
    dfs,
    alg_names,
    title,
    save=True,
    tol=1e-2,
    T=1_000,
    col="simple_regret",
    name_map=None,
):
    cols = [f"{name}.{col}" for name in alg_names]

    result = {col: [] for col in cols}
    for df in dfs:
        col_int = list(set(cols) & set(df.columns))
        for col in col_int:
            m = df[col].to_numpy()
            t = np.argmax(m <= tol)
            if m[t] <= tol:
                result[col].append(t)
            else:
                result[col].append(np.nan)
    max_len = np.max([len(arr) for arr in result.values()])
    for arr in result.values():
        while len(arr) < max_len:
            arr.append(np.nan)
    result_df = pd.DataFrame(result)
    result_df.columns = (
        alg_names if name_map is None else [name_map[name] for name in alg_names]
    )

    result_df.hist(bins=np.linspace(0, T, num=100))
    fig = plt.gcf()
    for ax in fig.get_axes():
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    if save:
        store_and_show_fig(root, fig, exp_name, title, show=False)


def plot_result_grouped(
    root,
    exp_name,
    dfs,
    algs,
    col,
    title,
    ylabel,
    xlabel,
    ylim=None,
    T=None,
    cumsum=False,
    hide_title=False,
    legend=True,
    small_plot=False,
    y_fn=None,
    bucket_size=1,
    constlines=None,
):
    for group_name, alg_names in algs.items():
        plot_result(
            root,
            exp_name,
            dfs,
            alg_names=alg_names,
            col=col,
            title=f"{title} of {group_name}" if group_name is not None else title,
            ylabel=ylabel,
            xlabel=xlabel,
            ylim=ylim,
            T=T,
            cumsum=cumsum,
            hide_title=hide_title,
            legend=legend,
            small_plot=small_plot,
            y_fn=y_fn,
            bucket_size=bucket_size,
            constlines=constlines,
        )
