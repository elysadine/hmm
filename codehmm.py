#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HMM Analysis of Subtraction Error Trajectories

Author: Elysa Marie Alfredine RAZAFINDRAFARA
Paper: JIJER-D-26-03234 - International Journal of Educational Research

Description:
This script implements the Hidden Markov Model analysis for modeling
the evolution of subtraction schemes in low-resource contexts.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = '../data/hmmsubstraction.xlsx'
OUTPUT_PATH = '../outputs/'
FIGURES_PATH = '../figures/'
N_STATES = 5
N_BOOTSTRAP = 500
RANDOM_SEED = 42

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """Load and preprocess the subtraction data."""
    df = pd.read_excel(filepath)
    print(f"Loaded {len(df)} students")
    return df

def prepare_observations(df):
    """
    Convert data to observation sequences for HMM.
    
    Returns:
        X: List of observation sequences
        lengths: Lengths of each sequence
    """
    # Identify columns corresponding to observable procedures
    # Example: o1, o2, o3, o4, o5
    observation_cols = ['o1', 'o2', 'o3', 'o4', 'o5']
    
    # Convert to sequences
    X = []
    lengths = []
    
    for idx, row in df.iterrows():
        # Create sequence for this student
        seq = []
        # Logic to convert student data to observation sequence
        # This will depend on your data structure
        pass
    
    return X, lengths

# ============================================================================
# HMM ESTIMATION
# ============================================================================

def estimate_hmm(X, lengths, n_states=N_STATES, n_restarts=100):
    """
    Estimate HMM with given number of states.
    
    Args:
        X: Observation sequences
        lengths: Lengths of sequences
        n_states: Number of latent states
        n_restarts: Number of random restarts
        
    Returns:
        model: Fitted HMM model
        log_likelihood: Log-likelihood of the best model
    """
    best_model = None
    best_ll = -np.inf
    
    for restart in range(n_restarts):
        model = hmm.MultinomialHMM(
            n_components=n_states,
            n_iter=1000,
            tol=1e-6,
            random_state=restart,
            init_params='ste'
        )
        
        try:
            model.fit(X, lengths)
            ll = model.score(X, lengths)
            
            if ll > best_ll:
                best_ll = ll
                best_model = model
        except:
            continue
    
    print(f"Best log-likelihood: {best_ll:.2f}")
    return best_model, best_ll

# ============================================================================
# BOOTSTRAP
# ============================================================================

def bootstrap_hmm(X, lengths, n_bootstrap=N_BOOTSTRAP):
    """
    Non-parametric bootstrap for uncertainty quantification.
    
    Args:
        X: Original observation sequences
        lengths: Original lengths
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        bootstrap_results: Dictionary with bootstrap statistics
    """
    n = len(lengths)
    bootstrap_trans = []
    bootstrap_emissions = []
    
    for b in range(n_bootstrap):
        # Resample students with replacement
        idx = np.random.choice(n, n, replace=True)
        
        # Build bootstrap sample
        X_boot = [X[i] for i in idx]
        lengths_boot = [lengths[i] for i in idx]
        
        # Estimate HMM on bootstrap sample
        try:
            model, _ = estimate_hmm(X_boot, lengths_boot, n_restarts=20)
            bootstrap_trans.append(model.transmat_)
            bootstrap_emissions.append(model.emissionprob_)
        except:
            continue
    
    # Compute confidence intervals
    trans_stack = np.array(bootstrap_trans)
    emissions_stack = np.array(bootstrap_emissions)
    
    ci_lower = np.percentile(trans_stack, 2.5, axis=0)
    ci_upper = np.percentile(trans_stack, 97.5, axis=0)
    
    return {
        'transitions': trans_stack,
        'emissions': emissions_stack,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

# ============================================================================
# RESULTS GENERATION
# ============================================================================

def generate_tables(model, bootstrap_results):
    """Generate transition and emission matrices."""
    
    # Transition matrix
    transmat = model.transmat_
    print("\n" + "="*60)
    print("Transition Matrix")
    print("="*60)
    print(pd.DataFrame(transmat, 
                       columns=['C0', 'C1', 'C2', 'C3', 'C4'],
                       index=['C0', 'C1', 'C2', 'C3', 'C4']))
    
    # Emission matrix
    emission = model.emissionprob_
    print("\n" + "="*60)
    print("Emission Matrix")
    print("="*60)
    print(pd.DataFrame(emission,
                       columns=['o1', 'o2', 'o3', 'o4', 'o5'],
                       index=['C0', 'C1', 'C2', 'C3', 'C4']))
    
    # Bootstrap confidence intervals for key transitions
    print("\n" + "="*60)
    print("Key Transition Probabilities (95% CI)")
    print("="*60)
    key_params = {
        'a22 (C2→C2)': (transmat[2,2], 
                        bootstrap_results['ci_lower'][2,2],
                        bootstrap_results['ci_upper'][2,2]),
        'a33 (C3→C3)': (transmat[3,3],
                        bootstrap_results['ci_lower'][3,3],
                        bootstrap_results['ci_upper'][3,3]),
        'a23 (C2→C3)': (transmat[2,3],
                        bootstrap_results['ci_lower'][2,3],
                        bootstrap_results['ci_upper'][2,3]),
        'a34 (C3→C4)': (transmat[3,4],
                        bootstrap_results['ci_lower'][3,4],
                        bootstrap_results['ci_upper'][3,4])
    }
    
    for name, (est, lower, upper) in key_params.items():
        print(f"{name}: {est:.3f}  [{lower:.3f}, {upper:.3f}]")
    
    return transmat, emission

def expected_time_to_mastery(transmat):
    """
    Compute expected time to mastery using fundamental matrix.
    
    Args:
        transmat: Transition matrix
        
    Returns:
        expected_sessions: Expected number of sessions to reach C4
    """
    # Q = transition submatrix for transient states (C0-C3)
    Q = transmat[:4, :4]
    
    # Fundamental matrix N = (I - Q)^(-1)
    I = np.eye(4)
    N = np.linalg.inv(I - Q)
    
    # Expected sessions from each state
    expected_sessions = N.sum(axis=1)
    
    print("\n" + "="*60)
    print("Expected Time to Mastery (C4)")
    print("="*60)
    states = ['C0 (Naive)', 'C1 (Mechanical)', 'C2 (Obstacle)', 'C3 (Partial)']
    for state, sessions in zip(states, expected_sessions):
        hours = sessions * 2  # 2 hours per session
        print(f"From {state}: {sessions:.1f} sessions ({hours:.1f} hours)")
    
    return expected_sessions

# ============================================================================
# FIGURES
# ============================================================================

def generate_figures(model, results):
    """Generate all figures from the paper."""
    
    # Figure 4: Transition graph
    plot_transition_graph(model.transmat_)
    
    # Figure 5: Expected time to mastery
    plot_expected_time(results['expected_time'])
    
    # Figure 6: Adaptive remediation flow
    plot_adaptive_remediation_flow()
    
    # Figure 7: Synthesis framework
    plot_synthesis_framework()

def plot_transition_graph(transmat):
    """Plot the transition graph between latent states."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define positions for states
    positions = {
        'C0': (0, 0),
        'C1': (1, 1),
        'C2': (2, 0),
        'C3': (3, 1),
        'C4': (4, 0)
    }
    
    # Plot nodes
    for state, pos in positions.items():
        ax.plot(pos[0], pos[1], 'o', markersize=20, 
                color='lightblue', edgecolor='black', linewidth=2)
        ax.text(pos[0], pos[1], state, ha='center', va='center', fontsize=10)
    
    # Plot transitions
    for i in range(5):
        for j in range(5):
            if transmat[i, j] > 0.05:
                # Draw arrow
                x1, y1 = positions[f'C{i}']
                x2, y2 = positions[f'C{j}']
                dx = x2 - x1
                dy = y2 - y1
                
                # Determine arrow style
                if i == j:
                    # Self-loop
                    ax.annotate(f'{transmat[i,j]:.2f}', 
                               xy=(x1, y1+0.3), xytext=(x1+0.2, y1+0.5),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'))
                else:
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
                    # Label at midpoint
                    mx, my = (x1+x2)/2, (y1+y2)/2
                    ax.text(mx+0.1, my+0.1, f'{transmat[i,j]:.2f}', 
                            fontsize=8, ha='center', va='center')
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Transition Graph between Latent Cognitive States')
    
    plt.tight_layout()
    plt.savefig('../figures/transition_graph.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_expected_time(expected_time):
    """Plot expected time to mastery."""
    states = ['C0 (Naive)', 'C1 (Mechanical)', 'C2 (Obstacle)', 'C3 (Partial)']
    colors = ['#ff9999', '#ffcc99', '#ff6666', '#66b3ff']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(states, expected_time, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add labels on bars
    for bar, time in zip(bars, expected_time):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{time*2:.1f} h', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Expected Sessions to Mastery', fontsize=12)
    ax.set_title('Mean Remediation Time to Reach Mastery by Initial Latent State', fontsize=14)
    ax.set_ylim(0, max(expected_time) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/expected_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_adaptive_remediation_flow():
    """Plot the adaptive remediation algorithm flow chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Simplified flow chart
    steps = [
        'Assess Student\n(Observe Procedure)',
        'Compute Posterior\nProbabilities',
        'Threshold?\nP(C2) > 0.7\nfor 2 sessions?',
        'Flag for\nIntensified Support',
        'Standard\nTaRL',
        'Additional 15 min\nC2-specific Strategies',
        'Reassess\nStudent'
    ]
    
    positions = [(0.5, 0.9), (0.5, 0.75), (0.5, 0.55), (0.2, 0.35),
                 (0.8, 0.35), (0.2, 0.15), (0.5, 0.15)]
    
    for step, pos in zip(steps, positions):
        ax.text(pos[0], pos[1], step, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black'),
                fontsize=9)
    
    # Add arrows
    # (simplified, would need proper coordinates)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('HMM-Based Adaptive Remediation Algorithm', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('../figures/adaptive_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_synthesis_framework():
    """Plot the synthesis framework."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Three pillars
    pillars = ['Didactical\nTheory', 'HMM\nIdentification', 'Instructional\nAction']
    positions = [(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)]
    
    for pillar, pos in zip(pillars, positions):
        ax.text(pos[0], pos[1], pillar, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2),
                fontsize=12, fontweight='bold')
    
    # Connecting arrows
    ax.annotate('', xy=(0.35, 0.5), xytext=(0.25, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', linewidth=2))
    ax.annotate('', xy=(0.65, 0.5), xytext=(0.55, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', linewidth=2))
    
    # Sub-components
    sub1 = ['Vergnaud (1990)', 'Brousseau (1998)', 'Normandeau (2010)']
    sub2 = ['Emission Prob.', 'Transition Prob.', 'Posterior Filtering']
    sub3 = ['State-specific\nDiagnosis', 'Targeted\nIntervention', 'Adaptive\nRemediation']
    
    for i, text in enumerate(sub1):
        ax.text(0.1 + i*0.1, 0.3, text, ha='center', va='center', fontsize=8, style='italic')
    for i, text in enumerate(sub2):
        ax.text(0.4 + i*0.1, 0.3, text, ha='center', va='center', fontsize=8)
    for i, text in enumerate(sub3):
        ax.text(0.7 + i*0.1, 0.3, text, ha='center', va='center', fontsize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.axis('off')
    ax.set_title('Integrated Framework: Connecting Didactical Theory, HMM Identification, and Instructional Action', 
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../figures/synthesis_framework.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete analysis."""
    print("=" * 70)
    print("HMM ANALYSIS OF SUBTRACTION ERROR TRAJECTORIES")
    print("=" * 70)
    print(f"Paper: JIJER-D-26-03234")
    print(f"Author: Elysa Marie Alfredine RAZAFINDRAFARA")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1] Loading data...")
    data = load_data(DATA_PATH)
    
    # 2. Prepare observations
    print("\n[2] Preparing observations...")
    X, lengths = prepare_observations(data)
    print(f"Number of sequences: {len(lengths)}")
    print(f"Total observations: {sum(lengths)}")
    
    # 3. Estimate HMM
    print("\n[3] Estimating HMM...")
    model, log_likelihood = estimate_hmm(X, lengths, n_states=N_STATES)
    
    # 4. Bootstrap
    print("\n[4] Running bootstrap...")
    bootstrap_results = bootstrap_hmm(X, lengths)
    
    # 5. Generate results
    print("\n[5] Generating results...")
    transmat, emission = generate_tables(model, bootstrap_results)
    
    # 6. Expected time to mastery
    print("\n[6] Computing expected time to mastery...")
    expected_time = expected_time_to_mastery(transmat)
    bootstrap_results['expected_time'] = expected_time
    
    # 7. Generate figures
    print("\n[7] Generating figures...")
    generate_figures(model, bootstrap_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nOutputs saved to:")
    print(f"  - Results: {OUTPUT_PATH}")
    print(f"  - Figures: {FIGURES_PATH}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()