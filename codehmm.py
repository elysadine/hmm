import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ======================================
# HMM Parameters (5 states, 4 observations)
# ======================================
A = np.array([
    [0.45, 0.40, 0.15, 0.00, 0.00],  # C0: Naive
    [0.00, 0.60, 0.35, 0.05, 0.00],  # C1: Mechanical
    [0.00, 0.00, 0.82, 0.15, 0.03],  # C2: Obstacle ("Petit-moins-Grand")
    [0.00, 0.00, 0.05, 0.55, 0.40],  # C3: Partial Understanding
    [0.00, 0.00, 0.00, 0.00, 1.00]   # C4: Expert
])

B = np.array([
    [0.80, 0.10, 0.05, 0.05],  # C0
    [0.70, 0.20, 0.05, 0.05],  # C1
    [0.02, 0.88, 0.05, 0.05],  # C2
    [0.30, 0.05, 0.50, 0.15],  # C3
    [1.00, 0.00, 0.00, 0.00]   # C4
])

state_labels = ["C0 Naive", "C1 Mechanical", "C2 Obstacle", "C3 Partial", "C4 Expert"]
obs_labels = ["Success (o1)", "Inv. P-G (o2)", "Forget Carry (o3)", "Decomp. (o4)"]

# ======================================
# 1. State Transition Diagram
# ======================================
G = nx.DiGraph()
for i, s_from in enumerate(state_labels):
    for j, s_to in enumerate(state_labels):
        if A[i,j] > 0.01:  # ignore very low probabilities
            G.add_edge(s_from, s_to, weight=A[i,j])

plt.figure(figsize=(8,6))
pos = nx.circular_layout(G)
edges = G.edges()
weights = [G[u][v]['weight']*5 for u,v in edges]
nx.draw(G, pos, with_labels=True, node_size=2500, node_color='skyblue',
        arrowsize=20, width=weights)
nx.draw_networkx_edge_labels(G, pos,
                             edge_labels={(u,v): f"{G[u][v]['weight']:.2f}" for u,v in edges})
plt.title("Figure 1: Cognitive State Transition Graph")
plt.show()

# ======================================
# 2. Temporal Bayesian Network
# ======================================
plt.figure(figsize=(8,4))
for t in range(3):  # 3 time steps
    plt.scatter([t]*5, range(5), s=1000, color='lightgreen')
    for s in range(5):
        plt.text(t, s, state_labels[s], ha='center', va='center')
        if t > 0:
            plt.arrow(t-1, s, 0.8, 0, head_width=0.2, head_length=0.05, fc='k', ec='k')
plt.title("Figure 2: Temporal Structure of the HMM (Z=latent state, Y=observation)")
plt.axis('off')
plt.show()

# ======================================
# 3. Emission Probabilities (Error Profiles)
# ======================================
plt.figure(figsize=(8,5))
x = np.arange(len(obs_labels))
width = 0.15

for i in range(4):  # C0-C3
    plt.bar(x + i*width, B[i], width=width, label=state_labels[i])

plt.xticks(x + 1.5*width, obs_labels)
plt.ylabel("Emission Probability")
plt.title("Figure 3: Error Emission Profile by Latent State")
plt.legend()
plt.show()

# ======================================
# 4. Learning Curve & Remediation Time
# ======================================
def simulate_absorption_prob(A, start, steps=20):
    prob = np.zeros(steps)
    state = start
    for t in range(steps):
        state = np.dot([1 if i==state else 0 for i in range(A.shape[0])], A)
        prob[t] = state[4]  # probability to reach C4
    return prob

prob_C0 = simulate_absorption_prob(A, start=0)
prob_C2 = simulate_absorption_prob(A, start=2)

plt.figure(figsize=(8,5))
plt.plot(range(1,len(prob_C0)+1), prob_C0, marker='o', label='Naive Student (C0)')
plt.plot(range(1,len(prob_C2)+1), prob_C2, marker='s', label='Blocked Student (C2)')
plt.xlabel("TaRL Sessions")
plt.ylabel("Probability of Reaching C4")
plt.title("Figure 4: Simulated Remediation Trajectory")
plt.legend()
plt.grid(True)
plt.show()

# ======================================
# 5. Baum-Welch Algorithm Flow
# ======================================
plt.figure(figsize=(6,4))
plt.plot([0,1,2,3,4], [0,0,0,0,0], 'o', markersize=20, color='orange')
plt.text(0,0,'O: Observations', ha='center', va='bottom')
plt.text(2,0,'E-step', ha='center', va='bottom')
plt.text(4,0,'M-step', ha='center', va='bottom')
plt.arrow(0,0.05,1.8,0, head_width=0.05, head_length=0.05, color='k')
plt.arrow(2,0.05,1.8,0, head_width=0.05, head_length=0.05, color='k')
plt.title("Figure 5: Iterative Baum-Welch Process")
plt.axis('off')
plt.show()