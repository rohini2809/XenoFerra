import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import csv
import io

# Function to build the network based on user inputs
def build_network(glucose_conc, reaction_rates):
    G = nx.DiGraph()
    # Define nodes with initial concentrations
    G.add_node("Glucose", concentration=glucose_conc)
    G.add_node("Pyruvate", concentration=0)
    G.add_node("Acetyl-CoA", concentration=0)
    G.add_node("TCA Cycle", concentration=0)
    G.add_node("ETC", concentration=0)
    G.add_node("Fe3+ Reduction", concentration=0)
    
    # Define edges with reaction rates from the reaction_rates dictionary
    G.add_edge("Glucose", "Pyruvate", reaction_rate=reaction_rates['GP'])
    G.add_edge("Pyruvate", "Acetyl-CoA", reaction_rate=reaction_rates['PA'])
    G.add_edge("Acetyl-CoA", "TCA Cycle", reaction_rate=reaction_rates['AC'])
    G.add_edge("TCA Cycle", "ETC", reaction_rate=reaction_rates['TC'])
    G.add_edge("ETC", "Fe3+ Reduction", reaction_rate=reaction_rates['ET'])
    
    return G

# Function to run the simulation
def run_simulation(G, time_steps=100, dt=1):
    # Extract initial concentrations
    concentrations = {node: G.nodes[node]["concentration"] for node in G.nodes()}
    history = {node: [] for node in G.nodes()}
    
    for t in range(time_steps):
        # Record concentrations for plotting
        for node in G.nodes():
            history[node].append(concentrations[node])
        changes = {node: 0.0 for node in G.nodes()}
        # For each reaction edge, compute flux = reaction_rate * [substrate]
        for (source, target, data) in G.edges(data=True):
            rate = data['reaction_rate']
            flux = rate * concentrations[source]
            changes[source] -= flux * dt
            changes[target] += flux * dt
        for node in G.nodes():
            concentrations[node] += changes[node]
    return history

# Streamlit Dashboard Layout

st.title("Digital Twin: Metabolic Simulation Dashboard")

st.markdown("""
This dashboard simulates a simple metabolic pathway. Adjust the parameters below to see how changes affect metabolite concentrations over time.
""")

# Sidebar for parameter inputs
st.sidebar.header("Simulation Parameters")

# Initial concentration for Glucose
glucose_conc = st.sidebar.slider("Initial Glucose Concentration", 1.0, 10.0, 5.0, step=0.5)

# Reaction rates for each step
st.sidebar.subheader("Reaction Rates")
gp_rate = st.sidebar.slider("Glucose → Pyruvate", 0.01, 1.0, 0.1, step=0.01)
pa_rate = st.sidebar.slider("Pyruvate → Acetyl-CoA", 0.01, 1.0, 0.08, step=0.01)
ac_rate = st.sidebar.slider("Acetyl-CoA → TCA Cycle", 0.01, 1.0, 0.07, step=0.01)
tc_rate = st.sidebar.slider("TCA Cycle → ETC", 0.01, 1.0, 0.05, step=0.01)
et_rate = st.sidebar.slider("ETC → Fe3+ Reduction", 0.01, 1.0, 0.03, step=0.01)

reaction_rates = {
    'GP': gp_rate,
    'PA': pa_rate,
    'AC': ac_rate,
    'TC': tc_rate,
    'ET': et_rate
}

# Simulation time
time_steps = st.sidebar.slider("Number of Time Steps", 50, 300, 100, step=10)

# Build the network based on user inputs
G = build_network(glucose_conc, reaction_rates)

# Display the network diagram
st.subheader("Annotated Metabolic Network")
fig_network, ax_network = plt.subplots(figsize=(6, 4))
pos = nx.spring_layout(G, seed=42)
node_labels = {node: f"{node}\n({G.nodes[node]['concentration']})" for node in G.nodes()}
nx.draw_networkx(G, pos, labels=node_labels, node_color='lightblue', 
                 node_size=1200, arrowstyle='->', arrowsize=20, ax=ax_network)
ax_network.set_title("Metabolic Network")
ax_network.axis('off')
st.pyplot(fig_network)

# Run simulation
history = run_simulation(G, time_steps=time_steps, dt=1)

# Plot simulation results
st.subheader("Dynamic Simulation Results")
fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
for node in history:
    ax_sim.plot(history[node], label=node)
ax_sim.set_xlabel("Time Steps")
ax_sim.set_ylabel("Concentration")
ax_sim.set_title("Metabolite Concentration Over Time")
ax_sim.legend()
st.pyplot(fig_sim)

# Option to download simulation results as CSV
st.subheader("Download Simulation Data")
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
header = ["Time"] + list(history.keys())
writer.writerow(header)
for t in range(time_steps):
    row = [t] + [history[node][t] for node in history]
    writer.writerow(row)
csv_data = csv_buffer.getvalue()
st.download_button(label="Download CSV", data=csv_data, file_name="simulation_results.csv", mime="text/csv")
