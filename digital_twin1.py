import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import io, csv, math
from pint import UnitRegistry
import numpy as np

# Set up the unit registry (Pint)
ureg = UnitRegistry()
Q_ = ureg.Quantity

# --- Debug Message to show app is running ---
st.write("Hello, world! The app is starting...")

# --- Kinetics Helper Functions ---

def arrhenius_adjustment(rate, temperature):
    """
    Adjust reaction rate using a simplified Arrhenius factor.
    Adjusted rate = rate * exp(alpha * (temperature - T_ref))
    where T_ref = 25°C and alpha is a sensitivity constant.
    """
    T_ref = 25  # reference temperature in °C
    alpha = 0.03  # sensitivity constant (1/°C)
    return rate * math.exp(alpha * (temperature - T_ref))

def michaelis_menten_flux(Vmax, Km, substrate_conc):
    """
    Calculate flux using Michaelis–Menten kinetics.
    Vmax: maximum rate (1/h)
    Km: Michaelis constant (g/L)
    substrate_conc: current concentration (g/L)
    Returns flux in g/L/h.
    """
    return Vmax * substrate_conc / (Km + substrate_conc)

# --- Build the Metabolic Network with Kinetics ---

def build_network(glucose_conc, reaction_rates, final_step_kinetics):
    """
    Builds the metabolic network.
    - The first four steps use linear kinetics.
    - The final step (ETC → Fe3+ Reduction) uses Michaelis–Menten kinetics.
    """
    G = nx.DiGraph()
    # Nodes: concentrations in g/L
    G.add_node("Glucose", concentration=float(glucose_conc))
    G.add_node("Pyruvate", concentration=0.0)
    G.add_node("Acetyl-CoA", concentration=0.0)
    G.add_node("TCA Cycle", concentration=0.0)
    G.add_node("ETC", concentration=0.0)
    # "Fe3+ Reduction" represents cumulative conversion corresponding to iron extraction (g/L)
    G.add_node("Fe3+ Reduction", concentration=0.0)
    
    # Linear steps: assign reaction rates in 1/h
    G.add_edge("Glucose", "Pyruvate", reaction_rate=reaction_rates['GP'])
    G.add_edge("Pyruvate", "Acetyl-CoA", reaction_rate=reaction_rates['PA'])
    G.add_edge("Acetyl-CoA", "TCA Cycle", reaction_rate=reaction_rates['AC'])
    G.add_edge("TCA Cycle", "ETC", reaction_rate=reaction_rates['TC'])
    # Final step: store Vmax and Km for Michaelis–Menten kinetics in the edge attributes.
    G.add_edge("ETC", "Fe3+ Reduction", Vmax=final_step_kinetics['Vmax'], Km=final_step_kinetics['Km'])
    return G

# --- Dynamic Simulation Function ---

def run_simulation(G, biomass, time_steps=100, dt=1):
    """
    Run the dynamic simulation using Euler integration.
    For the final step, Michaelis–Menten flux is multiplied by the bacterial biomass.
    Returns a history dictionary of concentrations (g/L) vs. time (h).
    """
    concentrations = {node: G.nodes[node]["concentration"] for node in G.nodes()}
    history = {node: [] for node in G.nodes()}
    
    for t in range(time_steps):
        for node in G.nodes():
            history[node].append(concentrations[node])
        changes = {node: 0.0 for node in G.nodes()}
        for (source, target, data) in G.edges(data=True):
            if source == "ETC" and target == "Fe3+ Reduction":
                # Use Michaelis–Menten kinetics scaled by biomass
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
    Perform sensitivity analysis by varying a parameter (e.g., Vmax or Km) over a range.
    Returns a tuple (parameter values, final Fe3+ Reduction values) for plotting.
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
    st.title("Enhanced Martian Bioleaching Digital Twin with Kinetics & Sensitivity Analysis")
    st.markdown("""
    #### Overview  
    This application simulates the metabolic pathway of *Shewanella oneidensis* for bioleaching iron from Martian regolith.
    It incorporates:
    - Credible units (g/L, 1/h) using the Pint library.
    - Michaelis–Menten kinetics for the Fe³⁺ reduction step.
    - A bacterial biomass factor that modulates the final iron reduction.
    - Interactive sensitivity analysis.
    
    Use the sidebar to adjust parameters and observe how changes affect the simulated iron recovery.
    """)

    # Display schematic image (replace placeholder URL with your own schematic image URL)
    st.image("https://via.placeholder.com/800x200.png?text=Martian+Bioleaching+Reactor+Diagram",
             caption="Schematic of Martian Bioleaching Reactor", use_column_width=True)

    # Sidebar Inputs: Metabolic Parameters
    st.sidebar.header("Metabolic Parameters")
    glucose_conc = st.sidebar.slider("Initial Glucose (g/L)", 1.0, 20.0, 5.0, step=0.5)
    gp_rate = st.sidebar.slider("Glucose → Pyruvate Rate (1/h)", 0.01, 0.2, 0.05, step=0.005)
    pa_rate = st.sidebar.slider("Pyruvate → Acetyl-CoA Rate (1/h)", 0.01, 0.2, 0.04, step=0.005)
    ac_rate = st.sidebar.slider("Acetyl-CoA → TCA Cycle Rate (1/h)", 0.01, 0.2, 0.03, step=0.005)
    tc_rate = st.sidebar.slider("TCA Cycle → ETC Rate (1/h)", 0.01, 0.2, 0.02, step=0.005)

    st.sidebar.header("Martian Conditions")
    martian_temp = st.sidebar.slider("Martian Temperature (°C)", -100, 30, -60, step=1)
    radiation = st.sidebar.slider("Radiation Level (a.u.)", 0.0, 10.0, 5.0, step=0.1)
    regolith_fe_factor = st.sidebar.slider("Regolith Iron Content Factor", 0.1, 5.0, 1.0, step=0.1)

    st.sidebar.header("Process & Reactor Parameters")
    regolith_mass = st.sidebar.slider("Regolith Mass Processed (g)", 100, 10000, 1000, step=100)
    process_efficiency = st.sidebar.slider("Process Efficiency (%)", 10, 100, 50, step=1)
    reactor_volume = st.sidebar.slider("Reactor Volume (L)", 10, 1000, 100, step=10)

    st.sidebar.header("Bacterial & Kinetic Parameters")
    biomass = st.sidebar.slider("Initial Bacterial Biomass (g/L)", 0.1, 5.0, 1.0, step=0.1)
    Vmax = st.sidebar.slider("Vₘₐₓ for Fe³⁺ Reduction (1/h)", 0.01, 0.1, 0.03, step=0.001)
    Km = st.sidebar.slider("Kₘ for Fe³⁺ Reduction (g/L)", 0.1, 10.0, 1.0, step=0.1)

    # Adjust final step kinetics based on Martian conditions
    base_et_rate = Vmax
    temp_adj = arrhenius_adjustment(1.0, martian_temp)
    radiation_adj = max(0.5, 1.0 - (radiation - 5) * 0.1)
    fe3_adjusted_kinetics = {'Vmax': base_et_rate * regolith_fe_factor * temp_adj * radiation_adj, 'Km': Km}

    time_steps = st.sidebar.slider("Simulation Time (h)", 50, 500, 200, step=10)

    # Build linear reaction rates dictionary for the first steps
    reaction_rates = {
        'GP': gp_rate,
        'PA': pa_rate,
        'AC': ac_rate,
        'TC': tc_rate
    }

    # Build the network
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

    # Run the dynamic simulation (includes biomass in final step)
    history = run_simulation(G, biomass, time_steps=time_steps, dt=1)

    # Plot simulation results
    st.subheader("Dynamic Simulation Results")
    fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
    for node in history:
        ax_sim.plot(history[node], label=node)
    ax_sim.set_xlabel("Time (h)")
    ax_sim.set_ylabel("Concentration (g/L)")
    ax_sim.set_title("Metabolite Concentration Profiles Over Time")
    ax_sim.legend()
    st.pyplot(fig_sim)

    # Calculate simulated iron yield: final Fe3+ Reduction concentration scaled by reactor volume
    simulated_yield = history["Fe3+ Reduction"][-1] * reactor_volume  # in grams
    st.subheader("Simulated Iron Recovery")
    st.markdown(f"**Final Fe³⁺ Reduction Concentration:** {history['Fe3+ Reduction'][-1]:.3f} g/L")
    st.markdown(f"**Reactor Volume:** {reactor_volume} L")
    st.markdown(f"**Estimated Iron Recovery:** {simulated_yield:.1f} g (from microbial activity)")

    # Theoretical yield from regolith processing
    iron_fraction = 0.179  # 17.9 wt% iron in regolith
    max_iron_g = regolith_mass * iron_fraction
    expected_iron_yield = max_iron_g * process_efficiency / 100
    st.subheader("Theoretical Iron Extraction from Regolith")
    st.markdown(f"**Regolith Processed:** {regolith_mass} g")
    st.markdown(f"**Iron in Regolith (17.9 wt%):** {max_iron_g:.1f} g")
    st.markdown(f"**Expected Iron Yield (at {process_efficiency}% efficiency):** {expected_iron_yield:.1f} g")

    # Sensitivity Analysis Section (for Fe³⁺ Reduction Vmax)
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

    st.markdown("""
    ### Detailed Insights
    - **Units & Kinetics:**  
      Concentrations are in g/L, reaction rates in 1/h. The final step uses Michaelis–Menten kinetics modulated by bacterial biomass.
    - **Environmental Effects:**  
      Martian temperature and radiation adjust reaction rates through Arrhenius and linear penalties.
    - **Iron Yield Estimates:**  
      The simulation scales the final Fe³⁺ reduction concentration by reactor volume for an estimated recovery, which is compared against theoretical yield from regolith.
    - **Sensitivity Analysis:**  
      Explore how changing Vₘₐₓ for Fe³⁺ reduction affects the overall iron extraction.
    - **Further Improvements:**  
      Integrate experimental data to refine kinetics, model dynamic biomass growth, and use interactive plots for deeper analysis.
    """)
except Exception as e:
    st.error(f"An error occurred: {e}")
