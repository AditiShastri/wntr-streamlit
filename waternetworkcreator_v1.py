import streamlit as st
import wntr
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
from wntr.network.io import write_inpfile

def create_water_network(num_junctions, num_tanks, num_reservoirs):
    # Create a blank water network model
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
    # Use the correct function to save the network to an INP file
    write_inpfile(wn, filename)

# Streamlit app layout
st.title("Water Network Designer")

# Input for number of junctions, tanks, and reservoirs
num_junctions = st.number_input("Enter number of Junctions", min_value=1, max_value=100, value=3)
num_tanks = st.number_input("Enter number of Tanks", min_value=1, max_value=10, value=2)
num_reservoirs = st.number_input("Enter number of Reservoirs", min_value=1, max_value=10, value=1)

# Option for selecting the layout
layout_option = st.selectbox("Choose network layout", ("Spring", "Kamada-Kawai", "Spectral"))

# Generate network when button is clicked
if st.button("Generate Network"):
    if num_junctions > 0 and num_tanks > 0 and num_reservoirs > 0:
        # Create the water network
        wn = create_water_network(num_junctions, num_tanks, num_reservoirs)
        st.success("Network created successfully!")

        # Display the network summary using describe() method
        st.write(wn.describe(level=1))

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
    else:
        st.warning("Please enter valid values for junctions, tanks, and reservoirs.")
