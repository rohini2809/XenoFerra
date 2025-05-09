import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import io, csv, math
from pint import UnitRegistry
import numpy as np

# Set up the unit registry using Pint
ureg = UnitRegistry()
Q_ = ureg.Quantity

# --- Kinetics Helper Functions ---

def arrhenius_adjustment(rate, temperature):
    """
    Adjust reaction rate using a simplified Arrhenius factor.
    Adjusted rate = rate * exp(alpha * (temperature - T_ref))
    where T_ref is 25°C and alpha is a sensitivity constant.
    """
    T_ref = 25  # reference temperature in °C
    alpha = 0.03  # sensitivity constant (1/°C)
    return rate * math.exp(alpha * (temperature - T_ref))

def michaelis_menten_flux(Vmax, Km, substrate_conc):
    """
    Calculate flux using Michaelis–Menten kinetics.
    Vmax: Maximum rate (1/h)
    Km: Michaelis constant (g/L)
    substrate_conc: Substrate concentration (g/L)
    Returns flux in g/L/h.
    """
    return Vmax * substrate_conc / (Km + substrate_conc)

# --- Build the Metabolic Network with Kinetics ---

def build_network(glucose_conc, reaction_rates, final_step_kinetics):
    """
    Builds a metabolic network representing a microbial pathway.
    The first four steps use linear kinetics.
    The final step (ETC → Fe³⁺ Reduction) uses Michaelis–Menten kinetics.
    """
    G = nx.DiGraph()
    # Nodes: Store concentrations in g/L
    G.add_node("Glucose", concentration=float(glucose_conc))
    G.add_node("Pyruvate", concentration=0.0)
    G.add_node("Acetyl-CoA", concentration=0.0)
    G.add_node("TCA Cycle", concentration=0.0)
    G.add_node("ETC", concentration=0.0)
    # "Fe³⁺ Reduction" represents the cumulative microbial conversion (proxy for iron extraction)
    G.add_node("Fe3+ Reduction", concentration=0.0)
    
    # Linear steps: assign reaction rates (1/h)
    G.add_edge("Glucose", "Pyruvate", reaction_rate=reaction_rates['GP'])
    G.add_edge("Pyruvate", "Acetyl-CoA", reaction_rate=reaction_rates['PA'])
    G.add_edge("Acetyl-CoA", "TCA Cycle", reaction_rate=reaction_rates['AC'])
    G.add_edge("TCA Cycle", "ETC", reaction_rate=reaction_rates['TC'])
    # Final step: use Michaelis–Menten kinetics with parameters Vmax and Km stored on the edge
    G.add_edge("ETC", "Fe3+ Reduction", Vmax=final_step_kinetics['Vmax'], Km=final_step_kinetics['Km'])
    return G

# --- Dynamic Simulation Function ---

def run_simulation(G, biomass, time_steps=100, dt=1):
    """
    Run the dynamic simulation using Euler integration.
    The Michaelis–Menten flux for the final step is multiplied by the bacterial biomass.
    Returns a dictionary "history" with metabolite concentration profiles (g/L) over time (h).
    """
    concentrations = {node: G.nodes[node]["concentration"] for node in G.nodes()}
    history = {node: [] for node in G.nodes()}
    
    for t in range(time_steps):
        for node in G.nodes():
            history[node].append(concentrations[node])
        changes = {node: 0.0 for node in G.nodes()}
        for (source, target, data) in G.edges(data=True):
            if source == "ETC" and target == "Fe3+ Reduction":
                Vmax = data['Vmax']
                Km = data['Km']
                flux = biomass * michaelis_menten_flux(Vmax, Km, concentrations[source])
            else:
                rate = data['reaction_rate']
                flux = rate * concentrations[source]
            changes[source] -= flux * dt
            changes[target] += flux * dt
        for node in G.nodes():
            concentrations[node] += changes[node]
    return history

# --- Sensitivity Analysis Function ---

def sensitivity_analysis(G, parameter, param_range, biomass, time_steps=100, dt=1):
    """
    Varies the specified parameter (e.g., Vmax or Km) over a range,
    runs the simulation for each value, and returns the final Fe3+ Reduction concentration.
    """
    final_values = []
    for val in param_range:
        G_mod = G.copy()
        if parameter == 'Vmax':
            G_mod["ETC"]["Fe3+ Reduction"]["Vmax"] = val
        elif parameter == 'Km':
            G_mod["ETC"]["Fe3+ Reduction"]["Km"] = val
        hist = run_simulation(G_mod, biomass, time_steps=time_steps, dt=dt)
        final_values.append(hist["Fe3+ Reduction"][-1])
    return param_range, final_values

# --- STREAMLIT APP ---

try:
    st.title("Enhanced Martian Bioleaching Digital Twin")
    st.markdown("""
#### Overview  
This app simulates the microbial bioleaching of iron from Martian regolith via the metabolic pathway of *Shewanella oneidensis*.
- **Units:** Concentrations are in g/L and reaction rates in 1/h.
- **Kinetics:** Early steps use linear kinetics, while the final Fe³⁺ reduction step uses Michaelis–Menten kinetics scaled by bacterial biomass.
- **Reactor Scaling:** Final Fe³⁺ reduction concentration (g/L) is multiplied by reactor volume to estimate total iron recovery.
- **Nutrient Consumption:** An estimate is provided (0.41 g nutrient per g iron reduced).

Use the sidebar to adjust parameters and observe how process conditions affect iron recovery.
    """)

    # Display a schematic diagram (replace the placeholder URL with an actual schematic if available)
    st.image("https://via.placeholder.com/800x200.png?text=Bioreactor+Schematic", 
             caption="Schematic of the Martian Bioleaching Reactor", use_column_width=True)

    # Sidebar: Metabolic Parameters
    st.sidebar.header("Metabolic Parameters")
    glucose_conc = st.sidebar.slider("Initial Glucose (g/L)", 1.0, 20.0, 5.0, step=0.5)
    gp_rate = st.sidebar.slider("Glucose → Pyruvate Rate (1/h)", 0.01, 0.2, 0.05, step=0.005)
    pa_rate = st.sidebar.slider("Pyruvate → Acetyl-CoA Rate (1/h)", 0.01, 0.2, 0.04, step=0.005)
    ac_rate = st.sidebar.slider("Acetyl-CoA → TCA Cycle Rate (1/h)", 0.01, 0.2, 0.03, step=0.005)
    tc_rate = st.sidebar.slider("TCA Cycle → ETC Rate (1/h)", 0.01, 0.2, 0.02, step=0.005)
    
    # Sidebar: Martian Environmental Conditions
    st.sidebar.header("Martian Conditions")
    martian_temp = st.sidebar.slider("Martian Temperature (°C)", -100, 30, -60, step=1)
    radiation = st.sidebar.slider("Radiation Level (a.u.)", 0.0, 10.0, 5.0, step=0.1)
    regolith_fe_factor = st.sidebar.slider("Regolith Iron Content Factor", 0.1, 5.0, 1.0, step=0.1)
    
    # Sidebar: Process & Reactor Parameters
    st.sidebar.header("Process & Reactor Parameters")
    regolith_mass = st.sidebar.slider("Regolith Mass Processed (g)", 100, 10000, 1000, step=100)
    process_efficiency = st.sidebar.slider("Process Efficiency (%)", 10, 100, 50, step=1)
    reactor_volume = st.sidebar.slider("Reactor Volume (L)", 10, 1000, 100, step=10)
    
    # Sidebar: Bacterial & Kinetic Parameters
    st.sidebar.header("Bacterial & Kinetic Parameters")
    biomass = st.sidebar.slider("Initial Bacterial Biomass (g/L)", 0.1, 5.0, 1.0, step=0.1)
    Vmax = st.sidebar.slider("Vₘₐₓ for Fe³⁺ Reduction (1/h)", 0.01, 0.1, 0.03, step=0.001)
    Km = st.sidebar.slider("Kₘ for Fe³⁺ Reduction (g/L)", 0.1, 10.0, 1.0, step=0.1)
    
    # Adjust final step kinetics for environmental effects
    base_et_rate = Vmax
    temp_adj = arrhenius_adjustment(1.0, martian_temp)
    radiation_adj = max(0.5, 1.0 - (radiation - 5) * 0.1)
    fe3_adjusted_kinetics = {'Vmax': base_et_rate * regolith_fe_factor * temp_adj * radiation_adj, 'Km': Km}
    
    time_steps = st.sidebar.slider("Simulation Time (h)", 50, 500, 200, step=10)
    
    # Build reaction rates dictionary for the linear steps
    reaction_rates = {
        'GP': gp_rate,
        'PA': pa_rate,
        'AC': ac_rate,
        'TC': tc_rate
    }
    
    # Build the metabolic network
    G = build_network(glucose_conc, reaction_rates, fe3_adjusted_kinetics)
    
    # Custom colors for nodes
    color_map = {
        "Glucose": 'skyblue',
        "Pyruvate": 'limegreen',
        "Acetyl-CoA": 'gold',
        "TCA Cycle": 'orange',
        "ETC": 'violet',
        "Fe3+ Reduction": 'tomato'
    }
    node_colors = [color_map[node] for node in G.nodes()]
    
    # Display the network diagram
    st.subheader("Enhanced Metabolic Network")
    fig_network, ax_network = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    node_labels = {node: f"{node}\n({G.nodes[node]['concentration']} g/L)" for node in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, ax=ax_network)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, ax=ax_network)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax_network)
    ax_network.set_title("Metabolic Network with Kinetics & Martian Adjustments")
    ax_network.axis('off')
    plt.tight_layout()
    st.pyplot(fig_network)
    
    # Run dynamic simulation (final step flux scaled by biomass)
    history = run_simulation(G, biomass, time_steps=time_steps, dt=1)
    
    # Plot simulation results: metabolite concentration profiles over time
    st.subheader("Dynamic Simulation Results")
    fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
    for node in history:
        ax_sim.plot(history[node], label=node)
    ax_sim.set_xlabel("Time (h)")
    ax_sim.set_ylabel("Concentration (g/L)")
    ax_sim.set_title("Metabolite Concentration Profiles Over Time")
    ax_sim.legend()
    st.pyplot(fig_sim)
    
    # Calculate simulated iron yield: final Fe³⁺ Reduction concentration scaled by reactor volume
    simulated_yield = history["Fe3+ Reduction"][-1] * reactor_volume  # yield in grams
    st.subheader("Simulated Iron Recovery")
    st.markdown(f"**Final Fe³⁺ Reduction Concentration:** {history['Fe3+ Reduction'][-1]:.3f} g/L")
    st.markdown(f"**Reactor Volume:** {reactor_volume} L")
    st.markdown(f"**Estimated Iron Recovery:** {simulated_yield:.1f} g (from microbial activity)")
    
    # Theoretical yield from regolith processing using Martian regolith iron content (~17.9 wt%)
    iron_fraction = 0.179
    max_iron_g = regolith_mass * iron_fraction
    expected_iron_yield = max_iron_g * process_efficiency / 100
    st.subheader("Theoretical Iron Extraction from Regolith")
    st.markdown(f"**Regolith Processed:** {regolith_mass} g")
    st.markdown(f"**Iron in Regolith (17.9 wt%):** {max_iron_g:.1f} g")
    st.markdown(f"**Expected Iron Yield (at {process_efficiency}% efficiency):** {expected_iron_yield:.1f} g")
    
    # Estimate Nutrient Consumption (assume 0.41 g nutrient per g iron reduced)
    nutrient_ratio = 0.41
    nutrient_consumption = simulated_yield * nutrient_ratio
    st.subheader("Estimated Nutrient Consumption")
    st.markdown(f"At 0.41 g nutrient per g iron, the process consumes about **{nutrient_consumption:.1f} g** nutrient.")
    
    # Sensitivity Analysis: Explore the effect of varying Vₘₐₓ on the final Fe³⁺ Reduction concentration
    st.subheader("Interactive Sensitivity Analysis for Fe³⁺ Reduction Vₘₐₓ")
    if st.checkbox("Show Sensitivity Analysis for Vₘₐₓ"):
        v_range = np.linspace(0.01, 0.1, 50)
        param_values, final_fe3 = sensitivity_analysis(G, 'Vmax', v_range, biomass, time_steps=time_steps, dt=1)
        fig_sens, ax_sens = plt.subplots(figsize=(8, 6))
        ax_sens.plot(param_values, final_fe3, 'b-', marker='o')
        ax_sens.set_xlabel("Vₘₐₓ (1/h)")
        ax_sens.set_ylabel("Final Fe³⁺ Reduction Concentration (g/L)")
        ax_sens.set_title("Sensitivity of Iron Extraction to Vₘₐₓ")
        st.pyplot(fig_sens)
    
    # Option to download simulation data as CSV
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
    
    # --- Detailed Explanation Section ---
    with st.expander("Detailed Explanation of the Model and Process"):
        st.markdown(r"""
**1. How the Simulation Works**

- **Metabolic Network Representation:**  
  The model simulates a microbial metabolic pathway using nodes (e.g., Glucose, Pyruvate, etc.). Each node’s value represents the concentration of that metabolite in grams per liter (g/L).

- **Reaction Kinetics:**  
  - *Linear Kinetics:* For early steps (Glucose → Pyruvate, etc.), the flux (amount converted per hour) is calculated by multiplying the reaction rate (1/h) by the substrate concentration (g/L).  
  - *Michaelis–Menten Kinetics:* For the final step (ETC → Fe³⁺ Reduction), a more realistic enzyme-driven kinetic is used. The flux is computed as:  

    \[
    \text{Flux} = \text{Biomass} \times \frac{V_{\text{max}} \times [\text{ETC}]}{K_m + [\text{ETC}]}
    \]

    The biomass factor modulates how effective the bacteria are at converting substrates into "reduced iron" (our proxy for iron extraction).

- **Euler Integration:**  
  The simulation uses Euler integration to update metabolite concentrations at each time step (in hours) based on the calculated fluxes.

**2. Estimating Iron Recovery**

- **Final Node – "Fe³⁺ Reduction":**  
  This node accumulates the product of the iron reduction process. Its value (in g/L) after the simulation represents the concentration of reduced iron.

- **Reactor Volume Scaling:**  
  The total mass of iron recovered is estimated by scaling the final concentration (g/L) by the reactor’s volume (L):

  \[
  \text{Total Iron Recovery (g)} = \text{Final Concentration (g/L)} \times \text{Reactor Volume (L)}
  \]

  Even if the reactor volume remains constant, it is a critical scaling factor when discussing process scalability.

- **Theoretical vs. Simulated Yield:**  
  - *Theoretical Yield* is calculated based on the regolith properties (e.g., ~17.9 wt% iron in Martian regolith) and assumed processing efficiency.  
  - *Simulated Yield* is derived from the dynamic simulation and reflects the microbial process under the chosen conditions.

**3. Reactor Volume as a Scaling Factor**

The reactor volume is set via a slider and acts solely as a multiplier for scaling the final concentration to a total mass. Changing the reactor volume will adjust the total iron recovered, demonstrating the scalability of the process.

**4. Bringing It All Together for Your Presentation**

- **Explain the Metabolic Network:**  
  Describe that each node represents a stage in microbial metabolism, with concentrations calculated over time.

- **Detail the Kinetic Modeling:**  
  Early steps use linear kinetics; the key iron reduction step uses Michaelis–Menten kinetics influenced by bacterial biomass.

- **Scaling Explanation:**  
  Clarify that the simulated concentration is converted to total iron recovery by multiplying with the reactor volume, highlighting potential outputs if scaled up.

- **Data Sources and Parameter Choices:**  
  Reference literature values for regolith iron content, reaction rates, and nutrient ratios. Indicate that while Mars experiments are limited, the model is based on the best available data.

- **Sensitivity and Optimization:**  
  Use sensitivity analysis to show which parameters (e.g., Vₘₐₓ) most affect the iron yield and discuss potential optimization strategies.
        """)
    
    # --- Footer: Copyright & Data Sources ---
    st.markdown("""
---
© 2025 Your Name. All rights reserved.

**Data Details & Sources:**
- **Martian Regolith Iron Content:** Approximately 17.9 wt% (based on current literature).
- **Microbial Kinetics & Bioleaching Parameters:** Derived from published research (e.g., Nature Communications) and experimental datasets.
- **Nutrient Consumption:** Assumed at 0.41 g nutrient per g iron reduced.
- **Additional Data:** Reactor scaling and process efficiency based on current ISRU studies.

This work is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
