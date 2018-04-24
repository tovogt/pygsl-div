"""
Implementation of "An information theoretic criterion for empirical
validation of simulation models" by Francesco Lamperti.
http://dx.doi.org/10.1016/j.ecosta.2017.01.006

created by Florian Roessler and Lies Boelen
"""

import click
import numpy as np
import pandas as pd
import scipy.stats as sc
from functools import reduce


def get_symbolised_ts(ts, b, L, min_per=1, max_per=99, state_space=None):
    """Symbolise a time-series based on sensitivity b.

    Parameters
    ----------
    ts: np.array
        array of all time series to be symbolised,
        first entry is the original data
    b: int
        the amount of symbols to be used for the
        time-series representation
    L: int
        the number of symbol word lengths to be checked
        (trade of individual correlation or patterns)
    min_per: int
        percentile to be used as minimum cut-off
    max_per: int
        percentile to be used as maximum cut-off
    state_space: tuple
        state space can be given when the boundaries of the
        state space are known and not taken from the time-series data

    Return
    ------
    a list of dataframes that contain symbol timeseries for
    combinations of b and L

    Notes
    -----
    min_per and max_per are used here instead of min and max
    values of the data to avoid sensitivity to outliers.
    """
    # if no state space is defined we generate our own based on
    # either the standard percentiles or those given by the user
    if not state_space:
        min_p = np.percentile(ts, min_per)
        max_p = np.percentile(ts, max_per)
        cuts = np.linspace(min_p, max_p, b+1)
    else:
        cuts = np.linspace(state_space[0], state_space[1], b+1)
    # now we map the time series to the bins in the symbol space
    symbolised_ts = np.array([np.digitize(t, cuts) for t in ts])
    # to be able to deal with "words" or combination of symbols it is easier
    # to deal with them as strings in pandas dfs
    # TODO: Maybe better way of doing this
    sym_str = pd.DataFrame(symbolised_ts).astype(str)
    # collect all symbol dataframes based on block size given
    all_dfs = []
    all_dfs.append(sym_str)
    for l in range(1, L):
        df = pd.DataFrame()
        for x in range(int(len(sym_str.columns))-l):
            df = pd.concat([df, sym_str.loc[:, x:x+l].apply("".join, axis=1)],
                           axis=1)
        all_dfs.append(df)
    return all_dfs


def get_weights(weight_type, L):
    """
    Generate the weights.

    Parameters
    ----------
    weight_type: str ('uniform' or 'add-progressive')
        string defining which weighting to be used
    L: int
        the number of symbol word lengths to be checked
        (trade of individual correlation or patterns)

    """
    if weight_type == 'uniform':
        w = np.array([1 / L] * L)
    elif weight_type == 'add-progressive':
        w = np.array([2 / (L * (L + 1))]*L).cumsum()
    return w


def gsl_div(original, model, weights='add-progressive',
            b=5, L=6, min_per=1, max_per=99, state_space=None):
    """Calculate the gsl_div between model and reference data.

    Parameters
    ----------
    original: np.array, shape(1, len(time-series))
        original reference data
    model: np.array, shape(len(reps), len(time-series))
        array of model results (replicates)
    weights: str ('add-progressive', 'uniform', )
        Specify the weighting to be applied to different block
        lengths.
    b: int
        the amount of symbols to be used for representation
    L: int
        the number of symbol combination to be checked
    min_per: int
        percentile to be used as minimum cut-off
    max_per: int
        percentile to be used as maximum cut-off
    state_space: tuple
        state space can be given when the boundaries of the
        state space are known and not taken from the ts data

    Notes
    -----
    Implementation of the following paper:
    http://dx.doi.org/10.1016/j.ecosta.2017.01.006
    """
    all_ts = np.concatenate([original, model])
    # determine the time series length
    T = original.shape[1]
    if T < L:
        raise ValueError('Word length cant be longer than timeseries')
    # symbolise time-series
    sym_ts = get_symbolised_ts(all_ts, b=b, L=L, min_per=min_per,
                               max_per=max_per, state_space=state_space)
    raw_divergence = []
    correction = []
    # run over all word sizes
    for n, ts in enumerate(sym_ts):
        # get frequency distributions for original and replicates
        # could do it by applying a pd.value_counts to every column but this
        # way you get a 10x speed up
        fs = pd.DataFrame((ts.stack().reset_index()
                           .groupby([0, "level_0"]).count() /
                           len(ts.T)).unstack().values)

        # replace NaN with 0 so that log does not complain later
        fs = fs.replace(np.nan, 0)
        # determine the size of vocabulary for the right base in the log
        base = b**(n+1)
        # calculate the distances between the different time-series
        # give a particluar word size
        M = (fs.iloc[:, 1:].values +
             np.expand_dims(fs.iloc[:, 0].values, 1)) / 2
        temp = (2 * sc.entropy(M, base=base) -
                sc.entropy(fs.values[:, 1:], base=base))
        raw_divergence.append(reduce(np.add, temp) / (len(fs.columns) - 1))
        cardinality_of_m = fs.apply(lambda x: reduce(np.logical_or, x),
                                    axis=1).sum()
        # if there is only one replicate this has to be handled differently
        if len(fs.columns) == 2:
            cardinality_of_reps = fs.iloc[:, 1].apply(lambda x: x != 0).sum()
        else:
            cardinality_of_reps = fs.iloc[:, 1:].apply(
                lambda x: reduce(np.logical_or, x), axis=1).sum()
        # calculate correction based on formula 9 line 2 in paper
        correction.append(2*((cardinality_of_m - 1) / (4 * T)) -
                          ((cardinality_of_reps - 1) / (2 * T)))

    w = get_weights(weight_type=weights, L=L)
    weighted_res = (w * np.array(raw_divergence)).sum(axis=0)
    weighted_correction = (w * np.array(correction)).sum()

    return weighted_res + weighted_correction


@click.command(context_settings=dict(max_content_width=120))
@click.option('--original', default='original.csv',
              help='Reference time series.',
              show_default=True)
@click.option('--model', default='model.csv',
              help='Model output time-series.',
              show_default=True)
@click.option('--weights', default='add-progressive',
              type=click.Choice(['add-progressive', 'uniform']),
              help='Which type of weighting to use.',
              show_default=True)
@click.option('--b', default=5,
              help='Number of symbols to use during symbolisation.',
              show_default=True)
@click.option('--l', default=6,
              help='Maximum word length to be used.',
              show_default=True)
@click.option('--min_per', default=1,
              type=click.IntRange(0, 100),
              help='percentile to be used as minimum cut-off.',
              show_default=True)
@click.option('--max_per', default=99,
              type=click.IntRange(0, 100),
              help='percentile to be used as maximum cut-off.',
              show_default=True)
@click.option('--state_space', default=None,
              help='State space boundaries. Format example: "(0, 1)"',
              show_default=True)
def main(original, model, weights, b, l, min_per, max_per, state_space):
    original_ts = pd.read_csv(original).values.T
    model_ts = pd.read_csv(model).values.T
    res = gsl_div(original_ts, model_ts, weights=weights, b=b, L=l, min_per=min_per,
                  max_per=max_per, state_space=eval(state_space))
    print(res)
    return res


if __name__ == '__main__':
    main()
