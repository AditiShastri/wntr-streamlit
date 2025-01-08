import streamlit as st

import wntr
import random
import matplotlib.pyplot as plt
from wntr.network.io import write_inpfile
import tempfile
from wntr.network.io import read_inpfile 
import networkx as nx
import numpy as np
from scipy.stats import expon
import pandas as pd

# Function to create water network (as before)
#st.set_option('deprecation.showPyplotGlobalUse', False)


def create_water_network(num_junctions, num_tanks, num_reservoirs):
    wn = wntr.network.WaterNetworkModel()

    # Generate junctions with random demands and elevations
    junctions = []
    for i in range(num_junctions):
        demand = random.randint(10, 50)  # Random demand between 10 and 50
        elevation = random.randint(10, 100)  # Random elevation between 10 and 100
        junction_id = f'J{i+1}'
        wn.add_junction(junction_id, base_demand=demand, elevation=elevation)
        junctions.append(junction_id)

    # Generate tanks with random attributes
    tanks = []
    for i in range(num_tanks):
        elevation = random.randint(100, 200)  # Random elevation between 100 and 200
        min_level = random.randint(5, 15)  # Min level between 5 and 15
        max_level = random.randint(min_level+5, 30)  # Max level should be greater than min_level
        init_level = random.randint(min_level, max_level)  # Init level between min_level and max_level
        tank_id = f'T{i+1}'
        wn.add_tank(tank_id, elevation=elevation, init_level=init_level, min_level=min_level, max_level=max_level, diameter=10)
        tanks.append(tank_id)

    # Generate reservoirs with random attributes
    reservoirs = []
    for i in range(num_reservoirs):
        elevation = random.randint(50, 150)  # Random elevation between 50 and 150
        base_head = random.randint(60, 100)  # Random base head between 60 and 100
        reservoir_id = f'R{i+1}'
        wn.add_reservoir(reservoir_id, base_head=base_head)
        reservoirs.append(reservoir_id)

    # Pipe counter to keep track of pipes
    pipe_counter = 1

    # Connect junctions to tanks and reservoirs with pipes (ensuring no isolated nodes)
    for i in range(min(num_junctions, num_tanks)):
        wn.add_pipe(f'P{pipe_counter}', junctions[i], tanks[i], length=random.randint(800, 1200), diameter=random.randint(250, 500), roughness=random.randint(100, 150))
        pipe_counter += 1
    
    for i in range(min(num_junctions, num_reservoirs)):
        wn.add_pipe(f'P{pipe_counter}', reservoirs[i], junctions[i], length=random.randint(500, 800), diameter=random.randint(200, 400), roughness=random.randint(100, 150))
        pipe_counter += 1
    
    # Ensure all junctions are connected
    if num_junctions > 1:
        for i in range(1, num_junctions):
            wn.add_pipe(f'P{pipe_counter}', junctions[i-1], junctions[i], length=random.randint(800, 1200), diameter=random.randint(250, 500), roughness=random.randint(100, 150))
            pipe_counter += 1
    
    # Ensure tanks are connected to reservoirs (if present)
    if num_tanks > 0 and num_reservoirs > 0:
        for i in range(min(num_tanks, num_reservoirs)):
            wn.add_pipe(f'P{pipe_counter}', tanks[i], reservoirs[i], length=random.randint(1000, 1500), diameter=random.randint(300, 500), roughness=random.randint(100, 150))
            pipe_counter += 1

    return wn

def save_inp_file(wn, filename='network.inp'):
    # Save the network to an INP file
    write_inpfile(wn, filename)

# Function to run earthquake simulation
def run_earthquake_simulation(epicenter, magnitude, depth, inp_file):
    wn = wntr.network.WaterNetworkModel(inp_file)
    wn = wntr.morph.scale_node_coordinates(wn, 1000)  # Optional, adjust coordinates

    # Initialize earthquake object
    earthquake = wntr.scenario.Earthquake(epicenter, magnitude, depth)
    
    # Calculate distance to epicenter for each pipe in the network
    R = earthquake.distance_to_epicenter(wn, element_type=wntr.network.Pipe)
    
    # Calculate PGA, PGV, and Repair Rate (RR)
    pga = earthquake.pga_attenuation_model(R)
    pgv = earthquake.pgv_attenuation_model(R)
    RR = earthquake.repair_rate_model(pgv)

    # Generate fragility curve for pipes
    pipe_FC = wntr.scenario.FragilityCurve()
    pipe_FC.add_state('Minor Leak', 1, {'Default': expon(scale=0.2)})
    pipe_FC.add_state('Major Leak', 2, {'Default': expon()})

    # Calculate leak probabilities based on repair rate
    L = pd.Series(wn.query_link_attribute('length', link_type=wntr.network.Pipe))
    pipe_Pr = pipe_FC.cdf_probability(RR * L)
    pipe_damage_state = pipe_FC.sample_damage_state(pipe_Pr)

    return wn, R, pga, pgv, RR, pipe_FC  # Returning 6 values
st.title("Water Network Designer App")


# Define the horizontal menu as a set of radio buttons
menu_options = ["Water Network Designer", "Network Analysis", "Pipe Criticality Analysis", "Earthquake Simulation"]
page = st.radio("Select a Page", menu_options, horizontal=True)

# Page 1: Water Network Designer
if page == "Water Network Designer":
    st.header("Design Your Water Network")
    # User inputs for network creation
    num_junctions = st.number_input("Enter number of Junctions", min_value=1, max_value=100, value=3)
    num_tanks = st.number_input("Enter number of Tanks", min_value=1, max_value=10, value=2)
    num_reservoirs = st.number_input("Enter number of Reservoirs", min_value=1, max_value=10, value=1)

    layout_option = st.selectbox("Choose network layout", ("Spring", "Kamada-Kawai", "Spectral"))

    if st.button("Generate Network"):
        if num_junctions > 0 and num_tanks > 0 and num_reservoirs > 0:
            # Create the water network
            wn = create_water_network(num_junctions, num_tanks, num_reservoirs)
            st.success("Network created successfully!")

            # Display the network summary using describe() method
            

            # Plot the network graphic with the selected layout
            G = wn.get_graph()

            # Choose layout based on user input
            if layout_option == "Spring":
                pos = nx.spring_layout(G, k=0.15, iterations=20)
            elif layout_option == "Kamada-Kawai":
                pos = nx.kamada_kawai_layout(G)
            else:  # Spectral Layout
                pos = nx.spectral_layout(G)

            plt.figure(figsize=(12, 10))  # Increase figure size for better readability
            nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
            plt.title(f"Water Network - {layout_option} Layout")
            st.pyplot(plt)

            st.header("Network description:")
            st.write(wn.describe(level=1))

            # Save and provide option to download the INP file
            inp_filename = 'network.inp'
            save_inp_file(wn, inp_filename)

            with open(inp_filename, "rb") as file:
                st.download_button(
                    label="Download INP File",
                    data=file,
                    file_name=inp_filename,
                    mime="application/octet-stream"
                )
            
            st.header("Graph Creation Algorithms")

            st.subheader('1. Spring Layout (Force-directed layout):')
            st.write(
                """
                - **Force-based system:** The Spring layout uses a physical model of forces between nodes (like springs), where edges act as springs that pull connected nodes closer, and nodes repel each other like charged particles, helping to find a balance.
                - **Good for general-purpose graphs:** It works well for graphs of arbitrary structure, producing visually appealing layouts where edges are approximately straight, and nodes are spaced out, making the graph easy to interpret.
                """
            )

            # Spectral Layout
            st.subheader('2. Spectral Layout:')
            st.write(
                """
                - **Eigenvalue decomposition:** Spectral layout uses the eigenvalues and eigenvectors of the graph's Laplacian matrix (a matrix representation of the graph's structure) to compute positions of nodes in space.
                - **Effective for graph clustering:** It is particularly useful for identifying clusters or communities within a graph since similar nodes are likely to end up close to each other in the layout.
                """
            )

            # Kamada-Kawai Layout
            st.subheader('3. Kamada-Kawai Layout:')
            st.write(
                """
                - **Energy-based minimization:** The Kamada-Kawai layout is an energy minimization algorithm that minimizes an energy function based on the distances between nodes, trying to make the Euclidean distance between nodes reflect the graph's topological distances.
                - **Non-force-based method:** Unlike Spring layout, which uses forces, Kamada-Kawai directly adjusts node positions using a spring-like energy model, often resulting in more evenly spaced layouts for medium to large graphs.
                """
            )
        else:
            st.warning("Please enter valid values for junctions, tanks, and reservoirs.")

# Page 2: INP File Upload
elif page == "Network Analysis":
    st.header("Upload INP File and View Network")
    # Your code for this page...
    uploaded_file = st.file_uploader("Choose an INP file", type=["inp"])

    if uploaded_file is not None:
        st.success("INP file uploaded successfully!")

        # Create a temporary file to save the uploaded INP file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.inp') as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmpfile.close()

            # Load the network model from the temporary INP file using the correct method
            wn = read_inpfile(tmpfile.name)  # Correct method to read INP file

            # Plot the network from the uploaded INP file
            fig, ax = plt.subplots(figsize=(12, 10))
            wntr.graphics.plot_network(wn, ax=ax)
            st.pyplot(fig)
            st.write("Basic Network Graph")

            # Calculate Average Expected Demand (AED)
            AED = wntr.metrics.average_expected_demand(wn)
            st.write("Average Expected Demand (AED):")
            st.write(AED.head())  # Display the first few rows of AED

            # Plot the AED on the network
            fig, ax = plt.subplots(figsize=(12, 10))
            wntr.graphics.plot_network(wn, node_attribute=AED, node_range=(0, 0.025), title='Average Expected Demand (m³/s)', ax=ax)
            st.pyplot(fig)

            # Identify junctions with zero demand
            zero_demand = AED[AED == 0].index
            st.write("Junctions with zero demand:")
            st.write(zero_demand)

            # Plot the network highlighting junctions with zero demand
            fig, ax = plt.subplots(figsize=(12, 10))
            wntr.graphics.plot_network(wn, node_attribute=list(zero_demand), title='Zero Demand Junctions', ax=ax)
            st.pyplot(fig)

            # Display the elevation data of junctions
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = wntr.graphics.plot_network(wn, node_attribute='elevation', node_colorbar_label='Elevation (m)', ax=ax)
            st.pyplot(fig)
            st.write("Elevation Graph")

# Page 3: Pipe Criticality Analysis
elif page == "Pipe Criticality Analysis":
    st.header("Pipe Criticality Analysis")
    # Your code for this page...
    uploaded_file = st.file_uploader("Choose an INP file", type=["inp"])

    if uploaded_file is not None:
        st.success("INP file uploaded successfully!")

        # Save the uploaded INP file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.inp') as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmpfile.close()

            # Load the water network model directly from the INP file path
            wn = wntr.network.WaterNetworkModel(tmpfile.name)  # Using the constructor with the file path

            # Adjust simulation options for criticality analyses
            analysis_end_time = 72 * 3600  # 72 hours in seconds
            wn.options.time.duration = analysis_end_time
            wn.options.hydraulic.demand_model = 'PDD'
            wn.options.hydraulic.required_pressure = 17.57  # Adjust based on the network's specifications
            wn.options.hydraulic.minimum_pressure = 0

            # List of pipes with large diameter for analysis (greater than or equal to 24 inches)
            pipes = wn.query_link_attribute('diameter', np.greater_equal, 24 * 0.0254, 
                                           link_type=wntr.network.model.Pipe)
            pipes = list(pipes.index)

            # Plot the pipes included in the criticality analysis
            fig, ax = plt.subplots(figsize=(12, 8))
            wntr.graphics.plot_network(wn, link_attribute=pipes, title='Pipes Included in Criticality Analysis', ax=ax)
            st.pyplot(fig)

            # Define the pressure threshold
            pressure_threshold = 14.06  # Minimum pressure for normal conditions

            # Preliminary simulation to determine junctions below pressure threshold under normal conditions
            sim = wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()
            min_pressure = results.node['pressure'].loc[:, wn.junction_name_list].min()
            below_threshold_normal_conditions = set(min_pressure[min_pressure < pressure_threshold].index)

            # Perform the criticality analysis, closing one pipe at a time
            junctions_impacted = {}  # Dictionary to store impacted junctions for each pipe
            for pipe_name in pipes:
                st.write(f'Running criticality analysis for Pipe: {pipe_name}...')
                
                # Reset network state for each pipe closure
                wn.reset_initial_values()

                # Close the pipe by adding a control
                pipe = wn.get_link(pipe_name)
                act = wntr.network.controls.ControlAction(pipe, 'status', wntr.network.LinkStatus.Closed)
                cond = wntr.network.controls.SimTimeCondition(wn, '=', '24:00:00')
                ctrl = wntr.network.controls.Control(cond, act)
                wn.add_control(f'close pipe {pipe_name}', ctrl)

                # Run the simulation
                sim = wntr.sim.WNTRSimulator(wn)
                results = sim.run_sim()

                # Extract junctions that fall below the pressure threshold after closing the pipe
                min_pressure = results.node['pressure'].loc[:, wn.junction_name_list].min()
                below_threshold = set(min_pressure[min_pressure < pressure_threshold].index)

                # Calculate the difference in impacted junctions
                junctions_impacted[pipe_name] = below_threshold - below_threshold_normal_conditions

                # Remove the control for the next iteration
                wn.remove_control(f'close pipe {pipe_name}')

            # Plot number of junctions impacted for each pipe closure
            number_of_junctions_impacted = {k: len(v) for k, v in junctions_impacted.items()}
            fig, ax = plt.subplots(figsize=(12, 8))
            wntr.graphics.plot_network(wn, link_attribute=number_of_junctions_impacted, 
                                       node_size=0, link_width=2, title='Number of Junctions Impacted by Pipe Closure', ax=ax)
            st.pyplot(fig)

            # Add a slider to let the user choose a specific pipe to visualize its impact
            pipe_names = list(junctions_impacted.keys())
            pipe_choice = st.slider("Select a Pipe to View Impacted Junctions", 
                                    min_value=0, max_value=len(pipe_names)-1, 
                                    value=0, step=1)
            
         

            selected_pipe = pipe_names[pipe_choice]
            st.write(f"Showing impacted junctions for Pipe {selected_pipe}...")

            # Plot impacted junctions for the selected pipe
            fig, ax = plt.subplots(figsize=(12, 8))

# Create a dictionary where each impacted junction is mapped to a constant value (1)
            impacted_junctions_dict = {junction: 1 for junction in junctions_impacted.get(selected_pipe, [])}

            # Plot the network with the impacted junctions highlighted
            wntr.graphics.plot_network(wn, node_attribute=impacted_junctions_dict,
                                    link_attribute=[selected_pipe], node_size=20, 
                                    title=f'Pipe {selected_pipe} Critical for Pressure Conditions at {len(junctions_impacted.get(selected_pipe, []))} Nodes', ax=ax)

            st.pyplot(fig)


# Page 4: Earthquake Simulation
elif page == "Earthquake Simulation":
    st.title("Earthquake Simulation for Water Network")
    # Your code for this page...
    uploaded_file = st.file_uploader("Choose a .inp file", type=["inp"])

    if uploaded_file is not None:
        # Save the uploaded file
        inp_file = uploaded_file.name
        with open(inp_file, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"INP file uploaded successfully: {inp_file}")

        # User inputs for earthquake characteristics
        st.header("Earthquake Characteristics")
        epicenter_x = st.number_input("Epicenter X (m)", value=32000)
        epicenter_y = st.number_input("Epicenter Y (m)", value=15000)
        magnitude = st.slider("Magnitude", min_value=4.0, max_value=9.0, step=0.1, value=6.5)
        depth = st.number_input("Depth (m)", value=10000)

        epicenter = (epicenter_x, epicenter_y)

        # Run the simulation when the user clicks the button
        if st.button("Run Earthquake Simulation"):
            # Run the earthquake simulation
            wn, R, pga, pgv, RR, pipe_FC  = run_earthquake_simulation(epicenter, magnitude, depth, inp_file)

            # Display earthquake characteristics
            st.header("Earthquake Characteristics")
            st.write(f"Epicenter: {epicenter}")
            st.write(f"Magnitude: {magnitude} on the Richter scale")
            st.write(f"Depth: {depth} meters")

            # Plot fragility curve
            st.header("Fragility Curve")
            plt.figure()
            wntr.graphics.plot_fragility_curve(pipe_FC, title="Fragility Curve", xlabel="Repair Rate * Pipe Length")
            st.pyplot()

            # Plot distance to epicenter, PGA, and PGV
            st.header("Impact Maps")

            # Plot Distance to Epicenter
            st.subheader("Distance to Epicenter")
            cmap = plt.cm.get_cmap('viridis')

            wntr.graphics.plot_network(wn, link_attribute=R, node_size=0, link_cmap= cmap, title='Distance to Epicenter')
            st.pyplot()

            # Plot PGA
            st.subheader("Peak Ground Acceleration (PGA)")
            wntr.graphics.plot_network(wn, link_attribute=pga, node_size=0, link_cmap= cmap, link_width=1.5, title='Peak Ground Acceleration (PGA)')
            st.pyplot()

            # Plot PGV
            st.subheader("Peak Ground Velocity (PGV)")
            wntr.graphics.plot_network(wn, link_attribute=pgv, node_size=0, link_cmap=cmap, link_width=1.5, title='Peak Ground Velocity (PGV)')
            st.pyplot()

            # Plot Probability of Leaks
            st.subheader("Leak Probabilities")

            # Plot Minor Leak Probability
            L = pd.Series(wn.query_link_attribute('length', link_type=wntr.network.Pipe))
            pipe_Pr = pipe_FC.cdf_probability(RR * L)
            pipe_damage_state = pipe_FC.sample_damage_state(pipe_Pr)
            wntr.graphics.plot_network(wn, link_attribute=pipe_Pr['Minor Leak'], node_size=0, link_cmap=cmap, link_range=[0, 1], link_width=1.5, title='Probability of Minor Leak')
            st.pyplot()

            # Plot Major Leak Probability
            wntr.graphics.plot_network(wn, link_attribute=pipe_Pr['Major Leak'], node_size=0, link_cmap=cmap, link_range=[0, 1], link_width=1.5, title='Probability of Major Leak')
            st.pyplot()

            # Display the calculated stats
            st.header("Calculated Stats")
            st.write(f"Min, Max, Average PGA: {np.round(pga.min(), 2)}, {np.round(pga.max(), 2)}, {np.round(pga.mean(), 2)} g")
            st.write(f"Min, Max, Average PGV: {np.round(pgv.min(), 2)}, {np.round(pgv.max(), 2)}, {np.round(pgv.mean(), 2)} m/s")
            st.write(f"Min, Max, Average Repair Rate: {np.round(RR.min(), 5)}, {np.round(RR.max(), 5)}, {np.round(RR.mean(), 5)} per m")

    else:
        st.info("Please upload an INP file to begin the simulation.")
