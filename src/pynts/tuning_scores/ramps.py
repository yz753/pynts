from typing import List, Optional

import numpy as np
import pynapple as nap
import statsmodels.api as sm
from numpy.typing import ArrayLike
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

from pynts.smoothing import apply_smoothing


def classify_ramps(score, null_distribution, alpha=0.01):
    result = {}

    for region in ["outbound", "homebound"]:
        # deal with nan vals
        if not score[f"{region}_valid_bins"]:
            result[f"{region}_sig"] = np.nan
            result[f"{region}_sign"] = 'NA'
            continue
        
        # Test significance
        result[f"{region}_sig"] = fdrcorrection(
            [score[f"{region}_pval"]] + null_distribution[f"{region}_pval"].tolist(),
            alpha,
        )[0][0] and (
            score[f"{region}_slope"]
            > norm.ppf(
                1 - alpha / 2,
                loc=np.nanmean(null_distribution[f"{region}_slope"]),
                scale=np.nanstd(null_distribution[f"{region}_slope"]),
            )
            or score[f"{region}_slope"]
            < norm.ppf(
                alpha / 2,
                loc=np.nanmean(null_distribution[f"{region}_slope"]),
                scale=np.nanstd(null_distribution[f"{region}_slope"]),
            )
        )
        result[f"{region}_sign"] = (
            "/"
            if not result[f"{region}_sig"]
            else "+"
            if score[f"{region}_slope"] > 0
            else "-"
        )
    return result


def compute_ramps(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    range: ArrayLike,
    context: str,
    trial_types: List[str],
    test_blocks: List[str],
    outbound: ArrayLike,
    homebound: ArrayLike,
    num_bins: Optional[int] = None,
    bin_size: Optional[int] = 1,
    smooth_sigma="cv",
    epoch=None,
    is_shuffle=True,
):
    if epoch is None:
        epoch = cluster.time_support
    
    select_trial_type = session["trials"][session["trials"]["type"].isin(trial_types)]
    if session['training_or_test'] == 'test':
        select_test_block = session["test_block"][session["test_block"]["type"].isin(test_blocks)]
        trials = select_trial_type.intersect(select_test_block)
    else:
        trials = select_trial_type

    range = (
        [(np.nanmin(session["P"]), np.nanmax(session["P"]))] if range is None else range
    )
    bins = num_bins

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster,
            session["P"],
            bins=bins,
            range=range,
            epochs=epochs.intersect(session["moving"]).intersect(trials),
        )[0]
    
    # because blocks are within a single session, the epoch used for smoothing needs to be defined based on that
    epoch = epoch.intersect(session["moving"]).intersect(trials)
    if len(epoch) == 0:
        raise ValueError("No valid epochs found for the given session and trials.")

    tc, smooth_sigma = apply_smoothing(
        compute_tuning_curve,
        epoch=epoch,
        dim=1,
        smooth_sigma=smooth_sigma,
        sigma_range=np.linspace(1, 3, 10),
        mode="wrap",
        keep=True,
    )
    positions = tc.coords["0"].values

    # Compute ramp fits
    results = {"_smooth_sigma": smooth_sigma}
    for region, ramp_bounds in [("outbound", outbound), ("homebound", homebound)]:
        mask = (positions > ramp_bounds[0]) & (positions < ramp_bounds[1])
        y = tc.values[mask]
        x = positions[mask]
        
        # checking & excluding nans
        non_nan_ratio = np.isfinite(y).sum() / len(y) if len(y) else 0.0
        results.update({
            f"{region}_valid_bins": False,
            f"{region}_slope": np.nan,
            f"{region}_intercept": np.nan,
            f"{region}_pval": np.nan,
            f"{region}_region": region,
        })
        results["tc"] = tc
        
        if non_nan_ratio < 0.8:
            continue # skip this fitting
        
        results[f"{region}_valid_bins"] = True
        model = sm.OLS(y, sm.add_constant(x), missing='drop').fit()
        results[f"{region}_slope"] = model.params[1]
        results[f"{region}_intercept"] = model.params[0]
        results[f"{region}_pval"] = model.pvalues[1]
        
    return results
