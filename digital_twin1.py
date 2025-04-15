import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import io, csv, math
from pint import UnitRegistry

# Set up Pint unit registry
ureg = UnitRegistry()
Q_ = ureg.Quantity

# ----------------------------
# Functions for Enhanced Simulation
# ----------------------------

def arrhenius_adjustment(rate, temperature):
    """
    Adjust reaction rate using a simplified Arrhenius factor.
    - temperature: in °C
    For demonstration, we assume that rate increases with temperature.
    """
    # Define a pseudo activation energy factor (placeholder value)
    # We use an exponential relationship: adjusted_rate = rate * exp[alpha * (temperature - T_ref)]
    T_ref = 25  # reference temperature in °C
    alpha = 0.03  # sensitivity constant (1/°C)
    adjusted_rate = rate * math.exp(alpha * (temperature - T_ref))
    return adjusted_rate

def build_network(glucose_conc, reaction_rates, fe3_adjusted_rate):
    """
    Build the metabolic network graph.
    """
    G = nx.DiGraph()
    # Add nodes; concentrations are stored as float values but documented with units (g/L, for instance)
    G.add_node("Glucose", concentration=float(glucose_conc))
    G.add_node("Pyruvate", concentration=0.0)
    G.add_node("Acetyl-CoA", concentration=0.0)
    G.add_node("TCA Cycle", concentration=0.0)
    G.add_node("ETC", concentration=0.0)
    G.add_node("Fe3+ Reduction", concentration=0.0)
    
    # Add edges with reaction_rate attributes (units: 1/h)
    G.add_edge("Glucose", "Pyruvate", reaction_rate=reaction_rates['GP'])
    G.add_edge("Pyruvate", "Acetyl-CoA", reaction_rate=reaction_rates['PA'])
    G.add_edge("Acetyl-CoA", "TCA Cycle", reaction_rate=reaction_rates['AC'])
    G.add_edge("TCA Cycle", "ETC", reaction_rate=reaction_rates['TC'])
    G.add_edge("ETC", "Fe3+ Reduction", reaction_rate=fe3_adjusted_rate)
    
    return G

def run_simulation(G, time_steps=100, dt=1):
    """
    Run a dynamic simulation over the metabolic network using Euler integration.
    """
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

# ----------------------------
# Streamlit App - Enhanced Digital Twin for Martian Bioleaching
# ----------------------------

st.title("Enhanced Martian Bioleaching Digital Twin")
st.markdown("""
#### Overview  
This app simulates the metabolic pathway of *Shewanella oneidensis* for bioleaching iron (Fe³⁺ reduction) from Martian regolith.  
It incorporates credible units using the Pint library and adjusts reaction rates based on Martian environmental conditions (temperature, radiation, and regolith iron content).  
The process demonstrates how microbial activity may be harnessed for in situ resource utilization (ISRU) on Mars.
""")

# Display an image of the microbe (Shewanella oneidensis)
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Shewanella_oneidensis_MR-1.jpg/640px-Shewanella_oneidensis_MR-1.jpg", 
         caption="Shewanella oneidensis", use_column_width=True)

# Sidebar Inputs: Basic Metabolic Parameters (with credible units and values)
st.sidebar.header("Metabolic Parameters")
glucose_conc = st.sidebar.slider("Initial Glucose Concentration (g/L)", 1.0, 20.0, 5.0, step=0.5)
# For each reaction, units of 1/h are assumed
gp_rate = st.sidebar.slider("Glucose → Pyruvate (1/h)", 0.01, 0.2, 0.05, step=0.005)
pa_rate = st.sidebar.slider("Pyruvate → Acetyl-CoA (1/h)", 0.01, 0.2, 0.04, step=0.005)
ac_rate = st.sidebar.slider("Acetyl-CoA → TCA Cycle (1/h)", 0.01, 0.2, 0.03, step=0.005)
tc_rate = st.sidebar.slider("TCA Cycle → ETC (1/h)", 0.01, 0.2, 0.02, step=0.005)

# Environmental (Martian) Conditions
st.sidebar.header("Martian Environmental Conditions")
martian_temp = st.sidebar.slider("Martian Temperature (°C)", -100, 30, -60, step=1)
radiation = st.sidebar.slider("Radiation Level (arbitrary units)", 0.0, 10.0, 5.0, step=0.1)
regolith_fe = st.sidebar.slider("Regolith Iron Content Factor", 0.1, 5.0, 1.0, step=0.1)

# Adjust Fe3+ reduction rate based on environmental conditions:
# Apply Arrhenius adjustment to modify the base Fe3+ reduction rate
base_et_rate = 0.03  # base rate (1/h) for ETC → Fe3+ reduction under nominal conditions
# Adjust for temperature: lower temperatures reduce the rate
temp_adjustment = arrhenius_adjustment(1.0, martian_temp)  # factor relative to nominal at 25°C
# Adjust for radiation: higher radiation reduces microbial efficiency (simple linear model)
radiation_adjustment = max(0.5, 1.0 - (radiation - 5) * 0.1)
# Final adjusted rate factors in regolith iron content
fe3_adjusted_rate = base_et_rate * regolith_fe * temp_adjustment * radiation_adjustment

# Simulation time parameters
time_steps = st.sidebar.slider("Number of Simulation Time Steps", 50, 500, 200, step=10)

# Build reaction rates dictionary for the first steps (in 1/h)
reaction_rates = {
    'GP': gp_rate,
    'PA': pa_rate,
    'AC': ac_rate,
    'TC': tc_rate
}

# Build the metabolic network
G = build_network(glucose_conc, reaction_rates, fe3_adjusted_rate)

# Enhanced visual: custom color mapping for nodes
color_map = {
    "Glucose": 'skyblue',
    "Pyruvate": 'limegreen',
    "Acetyl-CoA": 'gold',
    "TCA Cycle": 'orange',
    "ETC": 'violet',
    "Fe3+ Reduction": 'tomato'
}
node_colors = [color_map[node] for node in G.nodes()]

# Display the annotated metabolic network
st.subheader("Enhanced Metabolic Network")
fig_network, ax_network = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
node_labels = {node: f"{node}\n({G.nodes[node]['concentration']} g/L)" for node in G.nodes()}
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, ax=ax_network)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, ax=ax_network)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax_network)
ax_network.set_title("Metabolic Network (with Units & Martian Adjustments)")
ax_network.axis('off')
plt.tight_layout()
st.pyplot(fig_network)

# Run the dynamic simulation
history = run_simulation(G, time_steps=time_steps, dt=1)

# Plot simulation results
st.subheader("Dynamic Simulation Results")
fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
for node in history:
    ax_sim.plot(history[node], label=node)
ax_sim.set_xlabel("Time Steps (h)")
ax_sim.set_ylabel("Concentration (g/L)")
ax_sim.set_title("Evolution of Metabolite Concentrations Over Time")
ax_sim.legend()
st.pyplot(fig_sim)

# Option to download simulation results as CSV
st.subheader("Download Simulation Data")
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
header = ["Time (h)"] + list(history.keys())
writer.writerow(header)
for t in range(time_steps):
    row = [t] + [history[node][t] for node in history]
    writer.writerow(row)
csv_data = csv_buffer.getvalue()
st.download_button(label="Download CSV", data=csv_data, file_name="enhanced_simulation_results.csv", mime="text/csv")

# Additional explanation markdown
st.markdown("""
### Insights and Interpretation

- **Process Overview:**  
  *Shewanella oneidensis* metabolizes glucose through a sequence of reactions, culminating in the reduction of Fe³⁺—a key step in bioleaching iron from Martian regolith.

- **Influence of Martian Conditions:**  
  Lower temperatures and increased radiation levels decrease microbial efficiency. Our simulation adjusts the rate of Fe³⁺ reduction accordingly, which is critical for ISRU applications on Mars.

- **Application for NASA/ESA:**  
  This model illustrates how environmental variables affect bioleaching efficiency. Optimizing these parameters could improve in situ resource utilization (ISRU) for future Mars missions.
""")
