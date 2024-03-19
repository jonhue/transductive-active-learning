def setup_plots(matplotlib, small_plot=False):
    if small_plot:
        matplotlib.rcParams.update({"axes.labelsize": "large"})
    else:
        matplotlib.rcParams.update({"axes.labelsize": "medium"})
