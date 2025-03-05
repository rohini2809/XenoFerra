import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import io, csv

# Function to build the network, now with an adjusted Fe3+ reduction rate
def build_network(glucose_conc, reaction_rates, fe3_adjusted_rate):
    G = nx.DiGraph()
    # Add nodes with initial concentrations
    G.add_node("Glucose", concentration=glucose_conc)
    G.add_node("Pyruvate", concentration=0)
    G.add_node("Acetyl-CoA", concentration=0)
    G.add_node("TCA Cycle", concentration=0)
    G.add_node("ETC", concentration=0)
    G.add_node("Fe3+ Rate", concentration=0)
    
    # Add edges for the metabolic pathway; for the Fe3+ reduction step, use the adjusted rate.
    G.add_edge("Glucose", "Pyruvate", reaction_rate=reaction_rates['GP'])
    G.add_edge("Pyruvate", "Acetyl-CoA", reaction_rate=reaction_rates['PA'])
    G.add_edge("Acetyl-CoA", "TCA Cycle", reaction_rate=reaction_rates['AC'])
    G.add_edge("TCA Cycle", "ETC", reaction_rate=reaction_rates['TC'])
    G.add_edge("ETC", "Fe3+ Reduction", reaction_rate=fe3_adjusted_rate)
    
    return G

# Function to run the dynamic simulation using Euler integration
def run_simulation(G, time_steps=100, dt=1):
    concentrations = {node: G.nodes[node]["concentration"] for node in G.nodes()}
    history = {node: [] for node in G.nodes()}
    
    for t in range(time_steps):
        for node in G.nodes():
            history[node].append(concentrations[node])
        changes = {node: 0.0 for node in G.nodes()}
        for (source, target, data) in G.edges(data=True):
            rate = data['reaction_rate']
            flux = rate * concentrations[source]
            changes[source] -= flux * dt
            changes[target] += flux * dt
        for node in G.nodes():
            concentrations[node] += changes[node]
    return history

# -------------------- Streamlit App Begins --------------------
st.title("Martian Bioleaching Digital Twin")
st.markdown("""
### Overview  
This digital twin simulates the metabolic pathway of *Shewanella oneidensis* used for bioleaching iron from Martian regolith.  
**Bioleaching** is the process by which microbes dissolve metals from ores—in this case, extracting iron by reducing Fe³⁺.  
The simulation incorporates Martian environmental conditions (temperature, radiation, and regolith iron content) that influence microbial efficiency.
""")

# Display an image of the microbe (Shewanella oneidensis)
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Shewanella_oneidensis_MR-1.jpg/640px-Shewanella_oneidensis_MR-1.jpg", 
         caption="Shewanella oneidensis", use_column_width=True)

# Sidebar for simulation parameters
st.sidebar.header("Simulation Parameters")

# Basic metabolic parameters
glucose_conc = st.sidebar.slider("Initial Glucose Concentration", 1.0, 20.0, 5.0, step=0.5)
gp_rate = st.sidebar.slider("Glucose → Pyruvate Rate", 0.01, 1.0, 0.1, step=0.01)
pa_rate = st.sidebar.slider("Pyruvate → Acetyl-CoA Rate", 0.01, 1.0, 0.08, step=0.01)
ac_rate = st.sidebar.slider("Acetyl-CoA → TCA Cycle Rate", 0.01, 1.0, 0.07, step=0.01)
tc_rate = st.sidebar.slider("TCA Cycle → ETC Rate", 0.01, 1.0, 0.05, step=0.01)

# Martian environmental conditions
st.sidebar.header("Martian Conditions")
martian_temp = st.sidebar.slider("Martian Temperature (°C)", -80, 20, -40, step=1)
radiation = st.sidebar.slider("Martian Radiation (arbitrary units)", 0.0, 10.0, 5.0, step=0.1)
regolith_fe = st.sidebar.slider("Regolith Iron Content (factor)", 0.1, 5.0, 1.0, step=0.1)

# Adjust the Fe3+ reduction reaction based on Martian conditions:
# - If temperature is below 0°C, microbial activity is reduced (factor 0.5).
# - Higher radiation reduces activity further: for radiation above 5, reduce activity by 10% per unit (min factor 0.5).
temp_factor = 1.0 if martian_temp >= 0 else 0.5
radiation_factor = max(0.5, 1.0 - (radiation - 5) * 0.1)  # ensures a minimum factor of 0.5
base_et_rate = 0.03  # base rate for ETC → Fe3+ reduction
fe3_adjusted_rate = base_et_rate * regolith_fe * temp_factor * radiation_factor

# Simulation time
time_steps = st.sidebar.slider("Number of Time Steps", 50, 500, 100, step=10)

# Build the reaction rates dictionary for the first four steps
reaction_rates = {
    'GP': gp_rate,
    'PA': pa_rate,
    'AC': ac_rate,
    'TC': tc_rate
}

# Build the network with enhanced environmental factors
G = build_network(glucose_conc, reaction_rates, fe3_adjusted_rate)

# Display the annotated metabolic network
st.subheader("Enhanced Metabolic Network")
fig_network, ax_network = plt.subplots(figsize=(3, 3))
pos = nx.spring_layout(G, seed=42)
node_labels = {node: f"{node}\n({G.nodes[node]['concentration']})" for node, data in G.nodes(data=True)}
nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=800, ax=ax_network)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=8, ax=ax_network)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=4, ax=ax_network)
ax_network.set_title("Enhanced Metabolic Network with Martian Conditions")
ax_network.axis('off')
st.pyplot(fig_network)

# Run the dynamic simulation
history = run_simulation(G, time_steps=time_steps, dt=1)

# Plot simulation results
st.subheader("Dynamic Simulation Results")
fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
for node in history:
    ax_sim.plot(history[node], label=node)
ax_sim.set_xlabel("Time Steps")
ax_sim.set_ylabel("Concentration")
ax_sim.set_title("Metabolite Concentrations Over Time")
ax_sim.legend()
st.pyplot(fig_sim)

# Option to download simulation data as CSV
st.subheader("Download Simulation Data")
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
header = ["Time"] + list(history.keys())
writer.writerow(header)
for t in range(time_steps):
    row = [t] + [history[node][t] for node in history]
    writer.writerow(row)
csv_data = csv_buffer.getvalue()
st.download_button(label="Download CSV", data=csv_data, file_name="enhanced_simulation_results.csv", mime="text/csv")

st.markdown("""
### Insights from the Simulation

- **Microbial Bioleaching:**  
  The simulation shows how *Shewanella oneidensis* converts glucose through various metabolic stages, ultimately reducing Fe³⁺, which represents the bioleaching process extracting iron from Martian regolith.

- **Martian Conditions Impact:**  
  Lower temperatures and higher radiation levels reduce the efficiency of the Fe³⁺ reduction step. This adjustment reflects the challenges of operating a bioreactor under harsh Martian conditions.

- **Iron Extraction Potential:**  
  By modifying environmental parameters, we can explore how optimal conditions might be achieved to maximize iron extraction in a Martian bio-reactor setting.
""")
