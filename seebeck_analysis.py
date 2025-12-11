#!/usr/bin/env python3
"""
Seebeck Coefficient Analysis Script

Calculates Seebeck coefficient S(T) for thin film samples using indirect
thermometer calibration from cryogenic probe station measurements.

Author: Analysis script for CCR Dec 25 data
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path

# Configure matplotlib
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Paths
BASE_DIR = Path(__file__).parent.parent
CALIBRATION_DIR = BASE_DIR / "calibration"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed data"
VISUALS_DIR = BASE_DIR / "visuals"

# Ensure output directories exist
PROCESSED_DIR.mkdir(exist_ok=True)
VISUALS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_calibration_file(filepath):
    """
    Load a calibration TVCC CSV file.
    
    Returns DataFrame with columns:
    - heater_power: Heater Power [W]
    - strip_voltage: Strip Voltage [V]
    - strip_current: Strip Current [A]
    - strip_resistance: Calculated R = V/I [Ohm]
    """
    # Read the CSV, skipping the attributes line
    df = pd.read_csv(filepath, skiprows=1, skipinitialspace=True)
    
    # Clean column names: remove quotes, {Double} type annotations, and extra spaces
    cleaned_cols = []
    for col in df.columns:
        col = col.strip().strip('"').strip()
        # Remove {Double} or similar type annotations
        col = re.sub(r'\s*\{[^}]+\}', '', col).strip()
        cleaned_cols.append(col)
    df.columns = cleaned_cols
    
    # Find the actual column names (they might vary slightly)
    heater_power_col = [c for c in df.columns if 'Heater Power' in c][0]
    strip_voltage_col = [c for c in df.columns if 'Strip Voltage' in c and 'Error' not in c][0]
    strip_current_col = [c for c in df.columns if 'Strip Current' in c][0]
    
    # Extract relevant columns
    result = pd.DataFrame({
        'heater_power': pd.to_numeric(df[heater_power_col], errors='coerce'),
        'strip_voltage': pd.to_numeric(df[strip_voltage_col], errors='coerce'),
        'strip_current': pd.to_numeric(df[strip_current_col], errors='coerce'),
    })
    
    # Calculate resistance R = V/I (four-point probe)
    # Avoid division by zero
    mask = np.abs(result['strip_current']) > 1e-10
    result['strip_resistance'] = np.nan
    result.loc[mask, 'strip_resistance'] = (
        result.loc[mask, 'strip_voltage'] / result.loc[mask, 'strip_current']
    )
    
    return result


def extract_temperature_from_filename(filename):
    """Extract temperature value from calibration filename."""
    match = re.search(r'T=(\d+\.?\d*)\s*K', filename)
    if match:
        return float(match.group(1))
    return None


def load_all_calibration_data(side='hotside'):
    """
    Load all calibration files for one side (hotside or coldside).
    
    Returns dict: {temperature: DataFrame}
    """
    cal_dir = CALIBRATION_DIR / side
    files = glob.glob(str(cal_dir / "*.csv"))
    
    data = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        if filename.startswith('.'):
            continue
        temp = extract_temperature_from_filename(filename)
        if temp is not None:
            try:
                df = load_calibration_file(filepath)
                data[temp] = df
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    return data


def load_sample_thermovoltage(filepath):
    """
    Load sample thermovoltage data.
    
    Returns DataFrame with columns: temperature, dVdP, dVdP_error
    Filters out outliers and handles duplicate temperature measurements.
    """
    df = pd.read_csv(filepath, skiprows=1, skipinitialspace=True)
    # Clean column names: remove quotes, {Double} type annotations, and extra spaces
    cleaned_cols = []
    for col in df.columns:
        col = col.strip().strip('"').strip()
        col = re.sub(r'\s*\{[^}]+\}', '', col).strip()
        cleaned_cols.append(col)
    df.columns = cleaned_cols
    
    # Find the relevant columns
    temp_col = [c for c in df.columns if 'Temperature' in c][0]
    dVdP_col = [c for c in df.columns if 'Seebeck Power Coefficient' in c and 'Error' not in c][0]
    error_col = [c for c in df.columns if 'Error' in c][0]
    
    result = pd.DataFrame({
        'temperature': pd.to_numeric(df[temp_col], errors='coerce'),
        'dVdP': pd.to_numeric(df[dVdP_col], errors='coerce'),
        'dVdP_error': pd.to_numeric(df[error_col], errors='coerce')
    })
    
    # Filter outliers based on:
    # 1. Error/value ratio > 0.5 (error is more than half the value)
    # 2. Absolute dVdP value > 10x the median (likely measurement failure)
    # 3. Negative dVdP with large magnitude (probably sign error in failed measurement)
    median_dVdP = result['dVdP'].median()
    abs_median = np.abs(median_dVdP)
    
    # Create outlier mask
    error_ratio = np.abs(result['dVdP_error'] / result['dVdP'])
    outlier_mask = (
        (error_ratio > 0.5) |  # High relative error
        (np.abs(result['dVdP']) > 10 * abs_median) |  # Value far from median
        (np.abs(result['dVdP']) > 0.1)  # Absolute threshold (normal values are ~0.001-0.003)
    )
    
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        print(f"    Filtered {n_outliers} outlier(s) in thermovoltage data")
        result = result[~outlier_mask].copy()
    
    # Handle duplicate temperatures by keeping the measurement with lower relative error
    duplicated_temps = result['temperature'].duplicated(keep=False)
    if duplicated_temps.any():
        # For duplicates, calculate relative error and keep the best measurement
        result['rel_error'] = np.abs(result['dVdP_error'] / result['dVdP'])
        # Group by temperature and keep the row with minimum relative error
        idx_to_keep = result.groupby('temperature')['rel_error'].idxmin()
        n_dups = len(result) - len(idx_to_keep)
        result = result.loc[idx_to_keep].drop(columns=['rel_error'])
        if n_dups > 0:
            print(f"    Removed {n_dups} duplicate temperature measurements (kept best)")
    
    return result.sort_values('temperature').reset_index(drop=True)


def load_sample_conductivity(filepath):
    """
    Load sample conductivity data.
    
    Returns DataFrame with columns: temperature, conductivity, conductivity_error
    """
    df = pd.read_csv(filepath, skiprows=1, skipinitialspace=True)
    # Clean column names: remove quotes, {Double} type annotations, and extra spaces
    cleaned_cols = []
    for col in df.columns:
        col = col.strip().strip('"').strip()
        col = re.sub(r'\s*\{[^}]+\}', '', col).strip()
        cleaned_cols.append(col)
    df.columns = cleaned_cols
    
    # Find the relevant columns
    temp_col = [c for c in df.columns if 'Temperature' in c][0]
    cond_col = [c for c in df.columns if 'Conductivity' in c and 'Error' not in c][0]
    error_col = [c for c in df.columns if 'Error' in c][0]
    
    result = pd.DataFrame({
        'temperature': pd.to_numeric(df[temp_col], errors='coerce'),
        'conductivity': pd.to_numeric(df[cond_col], errors='coerce'),
        'conductivity_error': pd.to_numeric(df[error_col], errors='coerce')
    })
    
    return result.sort_values('temperature').reset_index(drop=True)


# =============================================================================
# Calibration Processing Functions
# =============================================================================

def extract_baseline_and_slope(df):
    """
    From R(P) data at a single temperature, extract:
    - R at P=0 (baseline resistance)
    - dR/dP (slope of linear fit)
    
    Uses only valid (non-NaN) resistance values.
    Returns: (R_baseline, dR_dP, R_baseline_err, dR_dP_err)
    """
    # Filter valid data
    mask = ~np.isnan(df['strip_resistance']) & ~np.isnan(df['heater_power'])
    P = df.loc[mask, 'heater_power'].values
    R = df.loc[mask, 'strip_resistance'].values
    
    if len(P) < 3:
        return np.nan, np.nan, np.nan, np.nan
    
    # Linear fit: R = R0 + (dR/dP) * P
    slope, intercept, r_value, p_value, std_err = stats.linregress(P, R)
    
    # Calculate intercept error
    n = len(P)
    residuals = R - (intercept + slope * P)
    mse = np.sum(residuals**2) / (n - 2)
    P_mean = np.mean(P)
    ss_P = np.sum((P - P_mean)**2)
    intercept_err = np.sqrt(mse * (1/n + P_mean**2/ss_P))
    
    return intercept, slope, intercept_err, std_err


def detect_outliers_zscore(values, threshold=3.0):
    """
    Detect outliers using z-score method.
    
    Returns: boolean mask where True indicates an outlier
    """
    if len(values) < 3:
        return np.zeros(len(values), dtype=bool)
    
    z_scores = np.abs(stats.zscore(values))
    return z_scores > threshold


def fit_polynomial_with_selection(T, values, orders=[3, 4, 5, 6], min_order=3):
    """
    Fit polynomials of different orders and select the best one.
    
    Selection criteria:
    - Adjusted R² (penalizes overfitting)
    - Residual analysis for systematic deviations
    
    Returns: (best_coefficients, best_order, metrics_dict)
    """
    best_order = min_order
    best_adj_r2 = -np.inf
    best_coeffs = None
    metrics = {}
    
    n = len(T)
    
    for order in orders:
        if order >= n - 1:
            continue
            
        coeffs = np.polyfit(T, values, order)
        fitted = np.polyval(coeffs, T)
        
        # Calculate R²
        ss_res = np.sum((values - fitted)**2)
        ss_tot = np.sum((values - np.mean(values))**2)
        r2 = 1 - ss_res / ss_tot
        
        # Adjusted R²
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - order - 1)
        
        metrics[order] = {
            'r2': r2,
            'adj_r2': adj_r2,
            'coeffs': coeffs,
            'rmse': np.sqrt(ss_res / n)
        }
        
        if adj_r2 > best_adj_r2:
            best_adj_r2 = adj_r2
            best_order = order
            best_coeffs = coeffs
    
    return best_coeffs, best_order, metrics


def polynomial_derivative(coeffs):
    """
    Compute derivative coefficients of a polynomial.
    
    If p(x) = c[0]*x^n + c[1]*x^(n-1) + ... + c[n]
    Then p'(x) = n*c[0]*x^(n-1) + (n-1)*c[1]*x^(n-2) + ...
    """
    n = len(coeffs) - 1
    deriv_coeffs = np.array([coeffs[i] * (n - i) for i in range(n)])
    return deriv_coeffs


def process_calibration_side(side='hotside'):
    """
    Process all calibration data for one side.
    
    Returns DataFrame with columns:
    - temperature
    - R_baseline (resistance at P=0)
    - dR_dP (resistance change per unit power)
    - R_baseline_err
    - dR_dP_err
    """
    print(f"\n{'='*60}")
    print(f"Processing {side} calibration data")
    print('='*60)
    
    # Load all calibration files
    cal_data = load_all_calibration_data(side)
    
    if not cal_data:
        print(f"No calibration data found for {side}")
        return None
    
    print(f"Found {len(cal_data)} temperature points: {sorted(cal_data.keys())}")
    
    # Extract baseline and slope at each temperature
    results = []
    for temp in sorted(cal_data.keys()):
        df = cal_data[temp]
        R0, dRdP, R0_err, dRdP_err = extract_baseline_and_slope(df)
        results.append({
            'temperature': temp,
            'R_baseline': R0,
            'dR_dP': dRdP,
            'R_baseline_err': R0_err,
            'dR_dP_err': dRdP_err
        })
    
    result_df = pd.DataFrame(results)
    
    # Check for outliers in R_baseline
    outlier_mask = detect_outliers_zscore(result_df['R_baseline'].values)
    if np.any(outlier_mask):
        outlier_temps = result_df.loc[outlier_mask, 'temperature'].values
        print(f"Warning: Potential outliers in R(T) at temperatures: {outlier_temps}")
    
    result_df['is_outlier'] = outlier_mask
    
    return result_df


def compute_calibration_fits(cal_df, side='hotside'):
    """
    Compute polynomial fits for R(T) and dR/dP(T).
    
    Returns dict with:
    - R_coeffs: polynomial coefficients for R(T)
    - R_order: polynomial order
    - dRdP_coeffs: polynomial coefficients for dR/dP(T)
    - dRdP_order: polynomial order
    - dRdT: derivative of R(T) evaluated at each temperature
    - dT_dP: temperature rise per unit power at each temperature
    """
    print(f"\nFitting polynomials for {side}...")
    
    # Use non-outlier data for fitting
    mask = ~cal_df['is_outlier']
    T = cal_df.loc[mask, 'temperature'].values
    R = cal_df.loc[mask, 'R_baseline'].values
    dRdP = cal_df.loc[mask, 'dR_dP'].values
    
    # Fit R(T) - this is usually smooth and well-behaved
    R_coeffs, R_order, R_metrics = fit_polynomial_with_selection(T, R)
    print(f"  R(T) best fit: order {R_order}, adj R² = {R_metrics[R_order]['adj_r2']:.6f}")
    
    # Fit dR/dP(T) - for plotting purposes, but we'll use raw values for dT/dP
    dRdP_coeffs, dRdP_order, dRdP_metrics = fit_polynomial_with_selection(T, dRdP)
    print(f"  dR/dP(T) best fit: order {dRdP_order}, adj R² = {dRdP_metrics[dRdP_order]['adj_r2']:.6f}")
    
    # Compute dR/dT from derivative of R(T) fit
    dRdT_coeffs = polynomial_derivative(R_coeffs)
    
    # Evaluate at all temperatures (including outliers for completeness)
    T_all = cal_df['temperature'].values
    R_fit = np.polyval(R_coeffs, T_all)
    dRdP_fit = np.polyval(dRdP_coeffs, T_all)
    dRdT = np.polyval(dRdT_coeffs, T_all)
    
    # Calculate dT/dP = (dR/dP) / (dR/dT)
    # Use RAW measured dR/dP values (not fitted) for more accurate results
    # The polynomial fit for dR/dP can introduce errors, especially at temperature extremes
    dRdP_raw = cal_df['dR_dP'].values
    dT_dP = dRdP_raw / dRdT
    
    return {
        'R_coeffs': R_coeffs,
        'R_order': R_order,
        'R_metrics': R_metrics,
        'dRdP_coeffs': dRdP_coeffs,
        'dRdP_order': dRdP_order,
        'dRdP_metrics': dRdP_metrics,
        'dRdT_coeffs': dRdT_coeffs,
        'R_fit': R_fit,
        'dRdP_fit': dRdP_fit,
        'dRdT': dRdT,
        'dT_dP': dT_dP
    }


def format_polynomial_equation(coeffs, var='T', precision=6):
    """Format polynomial coefficients as a readable equation string."""
    order = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = order - i
        if power == 0:
            terms.append(f"{c:.{precision}e}")
        elif power == 1:
            terms.append(f"{c:.{precision}e}·{var}")
        else:
            terms.append(f"{c:.{precision}e}·{var}^{power}")
    return " + ".join(terms)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_calibration_R_vs_T(cal_df, fits, side='hotside'):
    """Plot R(T) with polynomial fit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    T = cal_df['temperature'].values
    R = cal_df['R_baseline'].values
    outliers = cal_df['is_outlier'].values
    
    # Plot data points
    ax.errorbar(T[~outliers], R[~outliers], yerr=cal_df.loc[~outliers, 'R_baseline_err'],
                fmt='o', color='blue', markersize=6, capsize=3, label='Data')
    if np.any(outliers):
        ax.scatter(T[outliers], R[outliers], c='red', s=60, marker='x', 
                   linewidths=2, label='Outliers', zorder=5)
    
    # Plot fit
    T_smooth = np.linspace(T.min(), T.max(), 200)
    R_smooth = np.polyval(fits['R_coeffs'], T_smooth)
    ax.plot(T_smooth, R_smooth, 'r-', linewidth=2, label=f'Polynomial fit (order {fits["R_order"]})')
    
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Resistance [Ω]')
    ax.set_title(f'{side.capitalize()} Thermometer: R(T)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add fit equation
    eq = format_polynomial_equation(fits['R_coeffs'], precision=4)
    ax.text(0.02, 0.98, f"R(T) = {eq}", transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add R² value
    r2 = fits['R_metrics'][fits['R_order']]['adj_r2']
    ax.text(0.98, 0.02, f"Adj R² = {r2:.6f}", transform=ax.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / f'{side}_R_vs_T.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {side}_R_vs_T.png")


def plot_calibration_dRdP_vs_T(cal_df, fits, side='hotside'):
    """Plot dR/dP(T) with polynomial fit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    T = cal_df['temperature'].values
    dRdP = cal_df['dR_dP'].values
    outliers = cal_df['is_outlier'].values
    
    # Plot data points
    ax.errorbar(T[~outliers], dRdP[~outliers], yerr=cal_df.loc[~outliers, 'dR_dP_err'],
                fmt='o', color='blue', markersize=6, capsize=3, label='Data')
    if np.any(outliers):
        ax.scatter(T[outliers], dRdP[outliers], c='red', s=60, marker='x',
                   linewidths=2, label='Outliers', zorder=5)
    
    # Plot fit
    T_smooth = np.linspace(T.min(), T.max(), 200)
    dRdP_smooth = np.polyval(fits['dRdP_coeffs'], T_smooth)
    ax.plot(T_smooth, dRdP_smooth, 'r-', linewidth=2, label=f'Polynomial fit (order {fits["dRdP_order"]})')
    
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('dR/dP [Ω/W]')
    ax.set_title(f'{side.capitalize()} Thermometer: dR/dP(T)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add fit equation
    eq = format_polynomial_equation(fits['dRdP_coeffs'], precision=4)
    ax.text(0.02, 0.98, f"dR/dP(T) = {eq}", transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    r2 = fits['dRdP_metrics'][fits['dRdP_order']]['adj_r2']
    ax.text(0.98, 0.02, f"Adj R² = {r2:.6f}", transform=ax.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / f'{side}_dRdP_vs_T.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {side}_dRdP_vs_T.png")


def plot_calibration_dTdP_vs_T(cal_df, fits, side='hotside'):
    """Plot dT/dP(T)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    T = cal_df['temperature'].values
    dTdP = fits['dT_dP']
    
    ax.plot(T, dTdP, 'o-', color='green', markersize=6, linewidth=1.5)
    
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('dT/dP [K/W]')
    ax.set_title(f'{side.capitalize()} Thermometer: dT/dP(T) = (dR/dP)/(dR/dT)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / f'{side}_dTdP_vs_T.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {side}_dTdP_vs_T.png")


def plot_delta_T_per_P(T_common, delta_T_per_P):
    """Plot the net temperature difference per unit power."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(T_common, delta_T_per_P, 'o-', color='purple', markersize=6, linewidth=1.5)
    
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('ΔT/P [K/W]')
    ax.set_title('Net Temperature Difference per Unit Power: ΔT/P = (dT/dP)_hot − (dT/dP)_cold')
    ax.grid(True, alpha=0.3)
    
    # Check for negative values
    if np.any(delta_T_per_P < 0):
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.text(0.02, 0.02, "⚠ Warning: Negative ΔT/P values detected", 
                transform=ax.transAxes, color='red', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'delta_T_per_P.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved delta_T_per_P.png")


def plot_seebeck_individual(sample_name, T, S, S_err=None):
    """Plot Seebeck coefficient for a single sample."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if S_err is not None:
        ax.errorbar(T, S * 1e6, yerr=S_err * 1e6, fmt='o-', markersize=6, capsize=3)
    else:
        ax.plot(T, S * 1e6, 'o-', markersize=6)
    
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Seebeck Coefficient [μV/K]')
    ax.set_title(f'Seebeck Coefficient: {sample_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / f'{sample_name}_seebeck.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_conductivity_individual(sample_name, T, sigma, sigma_err=None):
    """Plot conductivity for a single sample."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if sigma_err is not None:
        ax.errorbar(T, sigma, yerr=sigma_err, fmt='o-', markersize=6, capsize=3)
    else:
        ax.plot(T, sigma, 'o-', markersize=6)
    
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Conductivity [S/cm]')
    ax.set_title(f'Electrical Conductivity: {sample_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / f'{sample_name}_conductivity.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_power_factor_individual(sample_name, T, PF, PF_err=None):
    """Plot power factor for a single sample."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if PF_err is not None:
        ax.errorbar(T, PF, yerr=PF_err, fmt='o-', markersize=6, capsize=3)
    else:
        ax.plot(T, PF, 'o-', markersize=6)
    
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Power Factor S²σ [μW/cm·K²]')
    ax.set_title(f'Power Factor: {sample_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / f'{sample_name}_power_factor.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(all_results, quantity, ylabel, title, filename):
    """Plot comparison of a quantity across all samples."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, (name, data) in enumerate(all_results.items()):
        T = data['temperature']
        y = data[quantity]
        y_err = data.get(f'{quantity}_err')
        
        if y_err is not None:
            ax.errorbar(T, y, yerr=y_err, fmt=markers[i]+'-', color=colors[i],
                       markersize=6, capsize=3, label=name, linewidth=1.5)
        else:
            ax.plot(T, y, markers[i]+'-', color=colors[i], markersize=6, 
                   label=name, linewidth=1.5)
    
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


# =============================================================================
# Main Analysis Functions
# =============================================================================

def run_calibration_analysis():
    """
    Run complete calibration analysis for both hot and cold sides.
    
    Returns: (hotside_df, coldside_df, hot_fits, cold_fits, common_T, delta_T_per_P)
    """
    print("\n" + "="*70)
    print("CALIBRATION ANALYSIS")
    print("="*70)
    
    # Process hotside
    hot_df = process_calibration_side('hotside')
    hot_fits = compute_calibration_fits(hot_df, 'hotside')
    hot_df['dRdT'] = hot_fits['dRdT']
    hot_df['dT_dP'] = hot_fits['dT_dP']
    
    # Process coldside
    cold_df = process_calibration_side('coldside')
    cold_fits = compute_calibration_fits(cold_df, 'coldside')
    cold_df['dRdT'] = cold_fits['dRdT']
    cold_df['dT_dP'] = cold_fits['dT_dP']
    
    # Generate calibration plots
    print("\nGenerating calibration plots...")
    plot_calibration_R_vs_T(hot_df, hot_fits, 'hotside')
    plot_calibration_dRdP_vs_T(hot_df, hot_fits, 'hotside')
    plot_calibration_dTdP_vs_T(hot_df, hot_fits, 'hotside')
    
    plot_calibration_R_vs_T(cold_df, cold_fits, 'coldside')
    plot_calibration_dRdP_vs_T(cold_df, cold_fits, 'coldside')
    plot_calibration_dTdP_vs_T(cold_df, cold_fits, 'coldside')
    
    # Calculate ΔT/P at common temperatures
    hot_temps = set(hot_df['temperature'].values)
    cold_temps = set(cold_df['temperature'].values)
    common_temps = sorted(hot_temps & cold_temps)
    
    print(f"\nCommon temperatures for ΔT/P calculation: {len(common_temps)} points")
    
    # For common temperatures, compute ΔT/P directly
    delta_T_per_P_direct = []
    for T in common_temps:
        hot_dTdP = hot_df.loc[hot_df['temperature'] == T, 'dT_dP'].values[0]
        cold_dTdP = cold_df.loc[cold_df['temperature'] == T, 'dT_dP'].values[0]
        delta_T_per_P_direct.append(hot_dTdP - cold_dTdP)
    
    common_T = np.array(common_temps)
    delta_T_per_P = np.array(delta_T_per_P_direct)
    
    # Plot ΔT/P
    plot_delta_T_per_P(common_T, delta_T_per_P)
    
    # Physical checks
    print("\nPhysical consistency checks:")
    print(f"  dT/dP (hot) > 0: {np.all(hot_df['dT_dP'] > 0)}")
    print(f"  dT/dP (cold) > 0: {np.all(cold_df['dT_dP'] > 0)}")
    print(f"  ΔT/P > 0: {np.all(delta_T_per_P > 0)}")
    
    if np.any(delta_T_per_P <= 0):
        print("  ⚠ Warning: Some ΔT/P values are non-positive!")
        problematic = common_T[delta_T_per_P <= 0]
        print(f"    Problematic temperatures: {problematic}")
        
        # Interpolate over bad values using only valid (positive) points
        valid_mask = delta_T_per_P > 0
        valid_T = common_T[valid_mask]
        valid_delta = delta_T_per_P[valid_mask]
        
        # Interpolate to get values at all temperatures
        delta_T_per_P_fixed = np.interp(common_T, valid_T, valid_delta)
        
        n_fixed = np.sum(~valid_mask)
        print(f"    Interpolated {n_fixed} bad ΔT/P values from {len(valid_T)} valid points")
        delta_T_per_P = delta_T_per_P_fixed
    
    # Save calibration data
    print("\nSaving calibration data...")
    hot_df.to_csv(PROCESSED_DIR / 'hotside_calibration.csv', index=False)
    cold_df.to_csv(PROCESSED_DIR / 'coldside_calibration.csv', index=False)
    
    delta_df = pd.DataFrame({
        'temperature': common_T,
        'delta_T_per_P': delta_T_per_P
    })
    delta_df.to_csv(PROCESSED_DIR / 'delta_T_per_P.csv', index=False)
    
    print("  Saved hotside_calibration.csv")
    print("  Saved coldside_calibration.csv")
    print("  Saved delta_T_per_P.csv")
    
    return hot_df, cold_df, hot_fits, cold_fits, common_T, delta_T_per_P


def run_sample_analysis(cal_T, delta_T_per_P):
    """
    Run analysis for all sample conditions.
    
    Returns: dict of results for each sample
    """
    print("\n" + "="*70)
    print("SAMPLE ANALYSIS")
    print("="*70)
    
    # Sample file mapping
    samples = {
        'pristine': ('prestine tv.csv', 'prestine cond.csv'),
        '0min': ('0min tv.csv', '0min cond.csv'),
        '1.5min': ('1.5min tv.csv', '1.5min cond.csv'),
        '5min': ('5min tv.csv', '5min cond.csv')
    }
    
    all_results = {}
    
    for sample_name, (tv_file, cond_file) in samples.items():
        print(f"\nProcessing {sample_name}...")
        
        # Load thermovoltage data
        tv_path = DATA_DIR / tv_file
        if not tv_path.exists():
            print(f"  Warning: {tv_file} not found, skipping")
            continue
        tv_data = load_sample_thermovoltage(tv_path)
        
        # Load conductivity data
        cond_path = DATA_DIR / cond_file
        if cond_path.exists():
            cond_data = load_sample_conductivity(cond_path)
        else:
            print(f"  Warning: {cond_file} not found")
            cond_data = None
        
        # Interpolate ΔT/P to sample temperatures
        sample_T = tv_data['temperature'].values
        
        # Use linear interpolation for ΔT/P
        delta_T_per_P_interp = np.interp(sample_T, cal_T, delta_T_per_P)
        
        # Calculate Seebeck coefficient: S = (dV/dP) / (ΔT/P)
        dVdP = tv_data['dVdP'].values
        dVdP_err = tv_data['dVdP_error'].values
        
        S = dVdP / delta_T_per_P_interp  # [V/K]
        
        # Propagate error (assuming ΔT/P error is negligible compared to dV/dP error)
        S_err = dVdP_err / delta_T_per_P_interp
        
        # Filter out physically unreasonable Seebeck values (> 30 μV/K)
        S_uV = S * 1e6
        seebeck_outlier_mask = np.abs(S_uV) > 30
        if np.any(seebeck_outlier_mask):
            n_outliers = np.sum(seebeck_outlier_mask)
            print(f"  Filtered {n_outliers} points with |S| > 30 μV/K")
            sample_T = sample_T[~seebeck_outlier_mask]
            dVdP = dVdP[~seebeck_outlier_mask]
            dVdP_err = dVdP_err[~seebeck_outlier_mask]
            delta_T_per_P_interp = delta_T_per_P_interp[~seebeck_outlier_mask]
            S = S[~seebeck_outlier_mask]
            S_err = S_err[~seebeck_outlier_mask]
        
        print(f"  Seebeck coefficient range: {S.min()*1e6:.1f} to {S.max()*1e6:.1f} μV/K")
        
        # Process conductivity and power factor
        if cond_data is not None:
            # Conductivity may have different temperature points, interpolate
            cond_T = cond_data['temperature'].values
            sigma = cond_data['conductivity'].values
            sigma_err = cond_data['conductivity_error'].values
            
            # For power factor, we need conductivity at the same temperatures as Seebeck
            sigma_interp = np.interp(sample_T, cond_T, sigma)
            
            # Power factor: S²σ (convert to μW/cm·K²)
            # S is in V/K, σ is in S/cm
            # S² in V²/K² = 10^12 μV²/K²
            # S²σ in V²·S/(cm·K²) = W/(cm·K²) = 10^6 μW/(cm·K²)
            PF = S**2 * sigma_interp * 1e6  # μW/(cm·K²)
            
            print(f"  Power factor range: {PF.min():.2f} to {PF.max():.2f} μW/cm·K²")
        else:
            sigma_interp = None
            PF = None
        
        # Store results
        result = {
            'temperature': sample_T,
            'seebeck': S * 1e6,  # Convert to μV/K
            'seebeck_err': S_err * 1e6,
            'dVdP': dVdP,
            'dVdP_err': dVdP_err,
            'delta_T_per_P': delta_T_per_P_interp
        }
        
        if sigma_interp is not None:
            result['conductivity'] = sigma_interp
            result['power_factor'] = PF
        
        all_results[sample_name] = result
        
        # Generate individual plots
        print(f"  Generating plots for {sample_name}...")
        plot_seebeck_individual(sample_name, sample_T, S, S_err)
        print(f"    Saved {sample_name}_seebeck.png")
        
        if cond_data is not None:
            plot_conductivity_individual(sample_name, cond_T, sigma, sigma_err)
            print(f"    Saved {sample_name}_conductivity.png")
            
            plot_power_factor_individual(sample_name, sample_T, PF)
            print(f"    Saved {sample_name}_power_factor.png")
        
        # Save processed data
        save_df = pd.DataFrame(result)
        safe_name = sample_name.replace('.', '_')
        save_df.to_csv(PROCESSED_DIR / f'seebeck_{safe_name}.csv', index=False)
        print(f"    Saved seebeck_{safe_name}.csv")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    
    # Seebeck comparison
    plot_comparison(all_results, 'seebeck', 'Seebeck Coefficient [μV/K]',
                   'Seebeck Coefficient Comparison', 'seebeck_comparison.png')
    
    # Conductivity comparison (only samples with conductivity data)
    cond_results = {k: v for k, v in all_results.items() if 'conductivity' in v}
    if cond_results:
        plot_comparison(cond_results, 'conductivity', 'Conductivity [S/cm]',
                       'Electrical Conductivity Comparison', 'conductivity_comparison.png')
        
        plot_comparison(cond_results, 'power_factor', 'Power Factor [μW/cm·K²]',
                       'Power Factor Comparison', 'power_factor_comparison.png')
    
    return all_results


def main():
    """Main entry point for the analysis."""
    print("="*70)
    print("SEEBECK COEFFICIENT ANALYSIS")
    print("="*70)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directories:")
    print(f"  Processed data: {PROCESSED_DIR}")
    print(f"  Visuals: {VISUALS_DIR}")
    
    # Run calibration analysis
    hot_df, cold_df, hot_fits, cold_fits, cal_T, delta_T_per_P = run_calibration_analysis()
    
    # Run sample analysis
    all_results = run_sample_analysis(cal_T, delta_T_per_P)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nProcessed data saved to: {PROCESSED_DIR}")
    print(f"Figures saved to: {VISUALS_DIR}")
    
    # Summary
    print("\n--- SUMMARY ---")
    for sample_name, data in all_results.items():
        T = data['temperature']
        S = data['seebeck']
        print(f"\n{sample_name}:")
        print(f"  Temperature range: {T.min():.0f} - {T.max():.0f} K")
        print(f"  Seebeck range: {S.min():.1f} to {S.max():.1f} μV/K")
        if 'power_factor' in data:
            PF = data['power_factor']
            print(f"  Max power factor: {PF.max():.2f} μW/cm·K² at {T[np.argmax(PF)]:.0f} K")


if __name__ == '__main__':
    main()
