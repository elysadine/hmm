import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------
# The Excel file must contain at least two columns:
# 'pretest'  : initial state (0 = failure, 1 = success)
# 'posttest' : final state (0 = failure, 1 = success)

file_path = "hmm.xlsx"
df = pd.read_excel(file_path)

# Define pretest and posttest component columns
pretest_component_cols = ['PRETESTADDITION', 'PRETESTSOUSTRACTION', 'PRETESTMULTIPLICATION', 'PRETESTDIVISION']
posttest_component_cols = ['TESTFINALADDITION', 'TESTFINALSOUSTRACTION', 'TESTFINALMULTIPLICATION', 'TESTFINALDIVISION']

# Convert component columns to numeric, coercing errors to NaN
df[pretest_component_cols] = df[pretest_component_cols].apply(pd.to_numeric, errors='coerce')
df[posttest_component_cols] = df[posttest_component_cols].apply(pd.to_numeric, errors='coerce')

# Fill any NaN values in the component columns with 0 (assuming missing means failure)
df[pretest_component_cols] = df[pretest_component_cols].fillna(0)
df[posttest_component_cols] = df[posttest_component_cols].fillna(0)

# Create the 'pretest' and 'posttest' columns by taking the maximum success across components
# A student is considered 'successful' (1) in the overall pretest/posttest if they succeed in at least one component.
# Otherwise, they are considered 'failed' (0) if they failed all components.
df['pretest'] = df[pretest_component_cols].max(axis=1)
df['posttest'] = df[posttest_component_cols].max(axis=1)

# Preview dataset
print("Preview of the dataset:")
print(df.head())

# Basic statistics
print("\nDescriptive statistics:")
print(df.describe())
# -------------------------------------------------------
# 2. TRANSITION MATRIX ESTIMATION
# -------------------------------------------------------
# Cross-tabulation counts
transition_counts = pd.crosstab(df['pretest'], df['posttest'])

# Convert counts to probabilities
transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)

print("\nTransition counts:")
print(transition_counts)

print("\nEmpirical transition matrix (probabilities):")
print(transition_matrix)

# Convert to numpy array for further computations
A_hat = transition_matrix.values
# -------------------------------------------------------
# 3. STOCHASTIC MONOTONICITY CHECK
# -------------------------------------------------------
# Probability of success before and after intervention
p_success_pre = df['pretest'].mean()
p_success_post = df['posttest'].mean()

print(f"\nProbability of success at pretest: {p_success_pre:.3f}")
print(f"Probability of success at posttest: {p_success_post:.3f}")

# Check monotonic progression
if p_success_post >= p_success_pre:
    print("Stochastic monotonicity validated: learning progression is globally increasing.")
else:
    print("Warning: monotonic learning progression is not verified.")
    # -------------------------------------------------------
# 4. COGNITIVE ATTRACTOR INERTIA
# -------------------------------------------------------
# Probability of remaining in failure state (state 0 -> 0)
a_00 = A_hat[0, 0]

# Expected time to exit the attractor
if a_00 < 1:
    expected_time_exit = 1 / (1 - a_00)
else:
    expected_time_exit = np.inf

print(f"\nProbability of staying in failure state (a_00): {a_00:.3f}")
print(f"Expected time to escape the cognitive attractor: {expected_time_exit:.2f} sessions")
# -------------------------------------------------------
# 5. FUNDAMENTAL MATRIX COMPUTATION
# -------------------------------------------------------
# Assume state 0 = non-expert (transient), state 1 = expert (absorbing)

Q = np.array([[A_hat[0, 0]]])  # Submatrix of non-expert states
I = np.eye(Q.shape[0])

# Fundamental matrix
N = np.linalg.inv(I - Q)

print("\nFundamental matrix N:")
print(N)

print("Expected number of sessions to reach mastery from failure:")
print(N.sum())
# -------------------------------------------------------
# 6. VISUALIZATION OF TRANSITIONS
# -------------------------------------------------------
plt.figure()
plt.imshow(A_hat)
plt.title("Empirical Transition Matrix")
plt.xlabel("Post-test State")
plt.ylabel("Pretest State")
plt.colorbar()
plt.show()
# -------------------------------------------------------
# 7. AUTOMATIC INTERPRETATION SUMMARY
# -------------------------------------------------------
print("\n--- INTERPRETATION SUMMARY ---")

if a_00 > 0.7:
    print("Strong cognitive attractor detected: persistent misconceptions.")
elif a_00 > 0.5:
    print("Moderate persistence of misconceptions observed.")
else:
    print("Low persistence: students progress relatively quickly.")

if p_success_post > p_success_pre:
    print("Intervention improves learning outcomes.")
else:
    print("No significant improvement detected.")