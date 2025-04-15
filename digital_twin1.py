import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import io, csv, math
from pint import UnitRegistry
import numpy as np

# Set up the unit registry (Pint)
ureg = UnitRegistry()
Q_ = ureg.Quantity

# --- Kinetics Helper Functions ---

def arrhenius_adjustment(rate, temperature):
    """
    Adjust reaction rate using a simplified Arrhenius factor.
    For demonstration, adjusted_rate = rate * exp(alpha * (temperature - T_ref))
    where T_ref is 25°C and alpha is a sensitivity constant.
    """
    T_ref = 25  # reference temperature in °C
    alpha = 0.03  # sensitivity constant (1/°C)
    return rate * math.exp(alpha * (temperature - T_ref))

def michaelis_menten_flux(Vmax, Km, substrate_conc):
    """
    Calculate flux using Michaelis–Menten kinetics.
    Vmax: maximum rate (1/h) with units 1/h
    Km: Michaelis constant (g/L) - the substrate concentration at half Vmax
    substrate_conc: current concentration of the substrate (g/L)
    Returns flux (g/L/h) assuming linearity in concentration dimension.
    """
    return Vmax * substrate_conc / (Km + substrate_conc)

# --- Build the Metabolic Network (with Units) ---

def build_network(glucose_conc, reaction_rates, final_step_kinetics):
    """
    Builds a directed graph representing the metabolic pathway.
    - `glucose_conc` is the initial glucose concentration (g/L).
    - `reaction_rates` is a dictionary of linear rate constants (1/h) for early steps.
    - `final_step_kinetics` is a dictionary containing Vmax and Km for the final Fe3+ reduction step.
    """
    G = nx.DiGraph()
    # Nodes: concentrations in g/L
    G.add_node("Glucose", concentration=float(glucose_conc))
    G.add_node("Pyruvate", concentration=0.0)
    G.add_node("Acetyl-CoA", concentration=0.0)
    G.add_node("TCA Cycle", concentration=0.0)
    G.add_node("ETC", concentration=0.0)
    # "Fe3+ Reduction" is linked to iron extraction (g/L)
    G.add_node("Fe3+ Reduction", concentration=0.0)
    
    # Edges for the first four steps: linear kinetics (rate * [S])
    G.add_edge("Glucose", "Pyruvate", reaction_rate=reaction_rates['GP'])
    G.add_edge("Pyruvate", "Acetyl-CoA", reaction_rate=reaction_rates['PA'])
    G.add_edge("Acetyl-CoA", "TCA Cycle", reaction_rate=reaction_rates['AC'])
    G.add_edge("TCA Cycle", "ETC", reaction_rate=reaction_rates['TC'])
    # For ETC -> Fe3+ Reduction, we store Vmax and Km for M-M kinetics in edge attributes.
    G.add_edge("ETC", "Fe3+ Reduction", Vmax=final_step_kinetics['Vmax'], Km=final_step_kinetics['Km'])
    return G

# --- Dynamic Simulation Function ---

def run_simulation(G, time_steps=100, dt=1):
    """
    Run a dynamic simulation using Euler integration.
    - For the first four steps, use linear kinetics: flux = rate * [S]
    - For the final step (ETC -> Fe3+ Reduction), use Michaelis–Menten kinetics.
    Returns a dictionary `history` of concentration vs. time for each node.
    """
    concentrations = {node: G.nodes[node]["concentration"] for node in G.nodes()}
    history = {node: [] for node in G.nodes()}
    
    for t in range(time_steps):
        for node in G.nodes():
            history[node].append(concentrations[node])
        changes = {node: 0.0 for node in G.nodes()}
        # Process each edge
        for (source, target, data) in G.edges(data=True):
            if source == "ETC" and target == "Fe3+ Reduction":
                # Use Michaelis–Menten kinetics
                Vmax = data['Vmax']
                Km = data['Km']
                flux = michaelis_menten_flux(Vmax, Km, concentrations[source])
            else:
                rate = data['reaction_rate']
                flux = rate * concentrations[source]
            changes[source] -= flux * dt
            changes[target] += flux * dt
        for node in G.nodes():
            concentrations[node] += changes[node]
    return history

# --- Sensitivity Analysis Function ---

def sensitivity_analysis(G, parameter, param_range, time_steps=100, dt=1):
    """
    Vary a parameter (e.g., Vmax for the final step) over a range,
    run simulations for each value, and return final Fe3+ Reduction values.
    - parameter: string, 'Vmax' or 'Km'
    - param_range: list or array of parameter values to test.
    Returns a tuple (param_values, final_values) for plotting.
    """
    final_values = []
    original_data = G.get_edge_data("ETC", "Fe3+ Reduction")
    for val in param_range:
        # Modify a copy of G for each simulation
        G_mod = G.copy()
        if parameter == 'Vmax':
            G_mod["ETC"]["Fe3+ Reduction"]["Vmax"] = val
        elif parameter == 'Km':
            G_mod["ETC"]["Fe3+ Reduction"]["Km"] = val
        hist = run_simulation(G_mod, time_steps=time_steps, dt=dt)
        # Collect final concentration for Fe3+ Reduction
        final_values.append(hist["Fe3+ Reduction"][-1])
    return param_range, final_values

# --- STREAMLIT APP ---

st.title("Enhanced Martian Bioleaching Digital Twin with Kinetics & Sensitivity Analysis")
st.markdown("""
#### Overview  
This application simulates the metabolic pathway of *Shewanella oneidensis* for bioleaching iron from Martian regolith.  
It uses realistic units (via the Pint library) and incorporates detailed kinetics:
- **Linear kinetics** for early steps.
- **Michaelis–Menten kinetics** for the final Fe³⁺ reduction step.  

The app also enables interactive sensitivity analysis to examine how varying key parameters affects iron extraction.
""")

# Display schematic or diagram (replace URL with your own diagram if available)
st.image("https://via.placeholder.com/800x200.png?text=Bioleaching+Reactor+Diagram", caption="Schematic of the Martian Bioleaching Reactor", use_column_width=True)

# Sidebar Inputs for Metabolic Parameters
st.sidebar.header("Metabolic Parameters")
glucose_conc = st.sidebar.slider("Initial Glucose (g/L)", 1.0, 20.0, 5.0, step=0.5)
gp_rate = st.sidebar.slider("Glucose → Pyruvate Rate (1/h)", 0.01, 0.2, 0.05, step=0.005)
pa_rate = st.sidebar.slider("Pyruvate → Acetyl-CoA Rate (1/h)", 0.01, 0.2, 0.04, step=0.005)
ac_rate = st.sidebar.slider("Acetyl-CoA → TCA Cycle Rate (1/h)", 0.01, 0.2, 0.03, step=0.005)
tc_rate = st.sidebar.slider("TCA Cycle → ETC Rate (1/h)", 0.01, 0.2, 0.02, step=0.005)

# Sidebar Inputs for Martian Conditions
st.sidebar.header("Martian Environmental Conditions")
martian_temp = st.sidebar.slider("Martian Temperature (°C)", -100, 30, -60, step=1)
radiation = st.sidebar.slider("Radiation Level (a.u.)", 0.0, 10.0, 5.0, step=0.1)
regolith_fe_factor = st.sidebar.slider("Regolith Iron Content Factor", 0.1, 5.0, 1.0, step=0.1)

# Process inputs for regolith and expected iron extraction
st.sidebar.header("Regolith Processing")
regolith_mass = st.sidebar.slider("Regolith Mass Processed (g)", 100, 10000, 1000, step=100)
process_efficiency = st.sidebar.slider("Process Efficiency (%)", 10, 100, 50, step=1)
iron_fraction = 0.179  # 17.9 wt% iron in regolith
max_iron_g = regolith_mass * iron_fraction
expected_iron_yield = max_iron_g * process_efficiency / 100

# Sidebar Inputs for Final Step Kinetics (Michaelis–Menten for Fe3+ Reduction)
st.sidebar.header("Fe³⁺ Reduction Kinetics (ETC → Fe3+ Reduction)")
Vmax = st.sidebar.slider("Vₘₐₓ (1/h)", 0.01, 0.1, 0.03, step=0.001)
Km = st.sidebar.slider("Kₘ (g/L)", 0.1, 10.0, 1.0, step=0.1)

# Adjust final Fe3+ reduction rate for environmental conditions:
base_et_rate = Vmax  # use Vmax as base for final step kinetics
temp_adj = arrhenius_adjustment(1.0, martian_temp)
radiation_adj = max(0.5, 1.0 - (radiation - 5) * 0.1)
fe3_adjusted_kinetics = {'Vmax': base_et_rate * regolith_fe_factor * temp_adj * radiation_adj, 'Km': Km}

# Simulation Time Input
time_steps = st.sidebar.slider("Simulation Time (h)", 50, 500, 200, step=10)

# Build reaction rates dictionary for linear steps (1/h)
reaction_rates = {
    'GP': gp_rate,
    'PA': pa_rate,
    'AC': ac_rate,
    'TC': tc_rate
}

# Build the metabolic network
G = build_network(glucose_conc, reaction_rates, fe3_adjusted_kinetics)

# Custom color mapping for nodes
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
ax_network.set_title("Metabolic Network with Kinetic & Martian Adjustments")
ax_network.axis('off')
plt.tight_layout()
st.pyplot(fig_network)

# Run dynamic simulation
history = run_simulation(G, time_steps=time_steps, dt=1)

# Plot simulation results (concentration vs. time)
st.subheader("Dynamic Simulation Results")
fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
for node in history:
    ax_sim.plot(history[node], label=node)
ax_sim.set_xlabel("Time (h)")
ax_sim.set_ylabel("Concentration (g/L)")
ax_sim.set_title("Metabolite Concentration Profiles Over Time")
ax_sim.legend()
st.pyplot(fig_sim)

# Display calculated iron extraction data
st.subheader("Estimated Iron Extraction from Regolith")
st.markdown(f"**Regolith Processed:** {regolith_mass} g")
st.markdown(f"**Iron in Regolith (17.9 wt%):** {max_iron_g:.1f} g")
st.markdown(f"**Expected Iron Yield (at {process_efficiency}% efficiency):** {expected_iron_yield:.1f} g")

# Sensitivity Analysis Section
st.subheader("Interactive Sensitivity Analysis (Fe³⁺ Reduction Vₘₐₓ)")
if st.checkbox("Show Sensitivity Analysis for Vₘₐₓ"):
    # Define a range for Vmax values for the final step
    v_range = np.linspace(0.01, 0.1, 50)
    param_values, final_iron = sensitivity_analysis(G, 'Vmax', v_range, time_steps=time_steps, dt=1)
    fig_sens, ax_sens = plt.subplots(figsize=(8, 6))
    ax_sens.plot(param_values, final_iron, 'b-', marker='o')
    ax_sens.set_xlabel("Vₘₐₓ (1/h)")
    ax_sens.set_ylabel("Final Fe³⁺ Reduction Concentration (g/L)")
    ax_sens.set_title("Sensitivity Analysis: Effect of Vₘₐₓ on Iron Extraction")
    st.pyplot(fig_sens)

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

st.markdown("""
### Detailed Insights
- **Units and Values:**  
  Concentrations are expressed in grams per liter (g/L) and reaction rates in per hour (1/h).  
  These values have been adjusted based on literature (e.g., Martian regolith iron is ~17.9 wt%).
  
- **Kinetic Modeling:**  
  The Fe³⁺ reduction step uses Michaelis–Menten kinetics with user-specified Vₘₐₓ and Kₘ, allowing for a more realistic non-linear response.
  
- **Sensitivity Analysis:**  
  The sensitivity analysis section shows how variations in Vₘₐₓ affect final iron concentration, providing insight into which parameters are most critical.
  
- **Process Understanding:**  
  The model estimates the potential iron yield from a given amount of processed regolith, and highlights how environmental conditions (temperature, radiation) impact microbial efficiency.
  
- **Future Improvements:**  
  Further refinements could include more detailed Monod kinetics for microbial growth, incorporation of additional inhibition factors, and validation with experimental data.
""")
