#-*- coding: utf-8 -*-
"""
Created on Mon July 22 13:05:00 2024

@author: author

"""
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Any

def generate_synthetic_gmm_data(components: List[Dict[str, Any]],
                                shuffle: bool = True,
                                random_state: int = 0) -> np.ndarray:
    """Generates synthetic data for a Gaussian Mixture Model (GMM).

    This function creates a dataset composed of multiple Gaussian distributions,
    each defined by a set of parameters.

    Parameters
    ----------
    components : List[Dict[str, Any]]
        A list of dictionaries, where each dictionary defines a Gaussian
        component and must contain the following keys:
        - 'n_samples': The number of samples to generate for this component.
        - 'mean': A NumPy array representing the mean of the distribution.
        - 'cov': A NumPy array representing the covariance matrix.
    shuffle : bool, optional
        If True (default), the generated data from all components will be
        shuffled to mix the populations.
    random_state : int, optional
        A seed for the random number generator to ensure reproducibility.
        Defaults to 0.

    Returns
    -------
    np.ndarray
        A NumPy array containing the combined and shuffled synthetic data.
        The shape of the array will be (total_samples, n_features).

    """
    rng = np.random.RandomState(random_state)
    all_data = []

    for comp in components:
        data = rng.multivariate_normal(
            comp['mean'], comp['cov'], comp['n_samples']
        )
        all_data.append(data)

    synthetic_data = np.vstack(all_data)

    if shuffle:
        np.random.shuffle(synthetic_data)

    return synthetic_data

def compare_gmm_results(original_params: List[Dict[str, Any]],
                        gmm_fit: GaussianMixture,
                        isotropic: bool) -> None:
    """Prints a comparison of original and GMM-fitted parameters.

    This utility function provides a clear, side-by-side view of the
    parameters used to generate synthetic data and those extracted by the GMM.

    Parameters
    ----------
    original_params : List[Dict[str, Any]]
        A list of dictionaries, where each dictionary contains the original
        parameters for a synthetic data component. For isotropic cases,
        it must contain 'center' and 'temp' keys. For anisotropic cases, it
        must contain 'center', 'temp_parallel', and 'temp_perp'.
    gmm_fit : GaussianMixture
        The fitted `sklearn.mixture.GaussianMixture` object.
    isotropic : bool
        A flag indicating whether the comparison should be for isotropic or
        anisotropic parameters.

    """
    from flekspy.amrex import AMReXParticleData

    print("--- Original Synthetic Data Parameters ---")
    for i, params in enumerate(original_params):
        if isotropic:
            print(f"  Population {i+1}: Center = {params['center'].tolist()}, "
                  f"Temperature = {params['temp']}")
        else:
            print(f"  Population {i+1}: Center = {params['center'].tolist()}, "
                  f"Temp Parallel = {params['temp_parallel']}, "
                  f"Temp Perp = {params['temp_perp']}")

    print("\n--- GMM Extracted Parameters ---")
    extracted_params = AMReXParticleData.get_gmm_parameters(
        gmm_fit, isotropic=isotropic
    )

    for i, params in enumerate(extracted_params):
        center = [round(c, 5) for c in params["center"]]
        if isotropic:
            v_th_sq = round(params["v_th_sq"], 5)
            print(f"  Component {i+1}: Center = {center}, v_th_sq = {v_th_sq}")
        else:
            v_th_sq_par = round(params["v_parallel_sq"], 5)
            v_th_sq_per = round(params["v_perp_sq"], 5)
            print(f"  Component {i+1}: Center = {center}, "
                  f"v_th_sq_parallel = {v_th_sq_par}, "
                  f"v_th_sq_perp = {v_th_sq_per}")
