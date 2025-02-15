import streamlit as st
import openai
import numpy as np
import plotly.graph_objs as go
from umap import UMAP
import tiktoken
import os
import fitz
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline
import scipy.stats as stats

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")  # Assuming you have a 'style.css' file with your custom styles


def create_spline_through_nodes(embedding_3d, line_colors):
    traces = []
    num_points = 50  # Number of points to interpolate between each pair of nodes

    for i in range(len(embedding_3d) - 1):
        # Extract the start and end points for the current segment
        start_point = embedding_3d[i]
        end_point = embedding_3d[i + 1]

        # Generate a sequence of x, y, z coordinates between the start and end points
        x = np.linspace(start_point[0], end_point[0], num_points)
        y = np.linspace(start_point[1], end_point[1], num_points)
        z = np.linspace(start_point[2], end_point[2], num_points)

        # Create a spline curve through the sequence of points
        cs_x = CubicSpline([0, num_points - 1], [start_point[0], end_point[0]])(
            range(num_points)
        )
        cs_y = CubicSpline([0, num_points - 1], [start_point[1], end_point[1]])(
            range(num_points)
        )
        cs_z = CubicSpline([0, num_points - 1], [start_point[2], end_point[2]])(
            range(num_points)
        )

        # Create a trace for the spline curve
        spline_trace = go.Scatter3d(
            x=cs_x,
            y=cs_y,
            z=cs_z,
            mode="lines",
            line=dict(
                width=2, color=line_colors[i]
            ),  # Use the corresponding color for this segment
            name=f"Spline Segment {i + 1}",
        )
        traces.append(spline_trace)

    return traces


# Function definitions from the notebook
def chunk_text(
    text: str, encoding_name: str = "cl100k_base", chunk_size: int = 512
) -> list:
    # Load the encoding
    encoding = tiktoken.get_encoding(encoding_name)

    # Tokenize the text
    tokens = encoding.encode(text)

    # Split the tokens into chunks
    chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    # Convert token chunks back into strings
    chunked_text = [encoding.decode(chunk) for chunk in chunks]

    return chunked_text


def chunk_text_rolling_window(
    text: str,
    encoding_name: str = "cl100k_base",
    chunk_size: int = 128,
    stride: int = 1,
) -> list:
    # Load the encoding
    print(f"Chunk size: {chunk_size}, stride: {stride}")
    encoding = tiktoken.get_encoding(encoding_name)

    # Tokenize the text
    tokens = encoding.encode(text)

    # Adjust the chunk size to not split words

    def find_chunk_boundary(start_index, tokens, chunk_size):
        end_index = start_index + chunk_size
        if end_index >= len(tokens):
            return len(tokens)  # Return the end of the token list if it's reached
        # Look for the nearest space or newline before the end_index to avoid splitting words
        while end_index > start_index and tokens[end_index - 1] not in {" ", "\n"}:
            end_index -= 1
        return (
            end_index if end_index > start_index else start_index + chunk_size
        )  # Ensure at least one token is included

    # Split the tokens into chunks with a rolling window
    chunks = []
    i = 0
    while i < len(tokens) - chunk_size + 1:
        end_index = find_chunk_boundary(i, tokens, chunk_size)
        chunks.append(tokens[i:end_index])
        i += stride  # Move the window by the stride

    # Convert token chunks back into strings
    chunked_text = [encoding.decode(chunk) for chunk in chunks]

    return chunked_text


@st.cache_data
def embed_text_chunks(text: str, openai_api_key: str) -> list:
    openai.api_key = openai_api_key
    chunks = chunk_text(text)
    embeddings = []
    for chunk in chunks:
        response = openai.embeddings.create(input=chunk, model="text-embedding-3-large")
        # Assuming the response object has the structure as mentioned
        # and that 'Embedding' is the correct class name for the items in the 'data' list
        embedding_vector = response.data[0].embedding
        embeddings.append(embedding_vector)
    return embeddings


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def embed_text_chunks_rolling_window(
    text: str, openai_api_key: str, chunk_size: int = 128, stride: int = 1
) -> tuple:
    openai.api_key = openai_api_key
    chunks = chunk_text_rolling_window(text, chunk_size=chunk_size, stride=stride)
    embeddings = []

    # Initialize the progress bar
    progress_bar = st.progress(0)
    progress_placeholder = st.empty()

    for i, chunk in enumerate(chunks):
        response = openai.embeddings.create(input=chunk, model="text-embedding-3-large")
        embedding_vector = response.data[0].embedding
        embeddings.append(embedding_vector)

        # Update the progress bar
        progress_bar.progress((i + 1) / len(chunks))

        # This will help Streamlit recognize that it needs to rerender the page
        progress_placeholder.text("")

    # Ensure the progress bar is filled at the end
    progress_bar.progress(1)

    return embeddings, chunks  # Return both embeddings and chunks


def extract_text_from_pdf(pdf_file):
    # Read the PDF file into bytes
    pdf_bytes = pdf_file.read()
    # Open the PDF file
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = ""
        # Iterate over each page
        for page in doc:
            # Extract text from the page
            text += page.get_text()
        return text


def wrap_text(text, line_length):
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= line_length:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)  # Append the last line

    return "<br>".join(lines)


def create_lines_to_nearest_nodes(
    question_embedding, nearest_indices, original_embeddings, umap_coords
):
    lines = []
    question_x, question_y, question_z = umap_coords["question"]

    # Calculate all distances to determine the range for color scaling
    distances = [
        np.linalg.norm(question_embedding - original_embeddings[index])
        for index in nearest_indices
    ]
    max_distance = max(distances)
    min_distance = min(distances)
    distance_range = max_distance - min_distance

    for index in nearest_indices:
        node_embedding = original_embeddings[index]
        distance = np.linalg.norm(question_embedding - node_embedding)

        # Normalize the distance to a 0-1 scale for color interpolation
        normalized_distance = (
            (distance - min_distance) / distance_range if distance_range else 0
        )

        # Interpolate color based on normalized distance
        # Assuming bright red for closest (normalized_distance = 0) and dark red for furthest (normalized_distance = 1)
        color = f"rgb({255 * (1 - normalized_distance)}, {0}, {0})"

        node_x, node_y, node_z = umap_coords["nodes"][index]

        lines.append(
            go.Scatter3d(
                x=[question_x, node_x],
                y=[question_y, node_y],
                z=[question_z, node_z],
                mode="lines",
                line=dict(width=2, color=color, dash="dot"),
                name=f"Line to Node {index + 1} (dist: {distance:.2f})",
            )
        )
    return lines


# Streamlit UI components
st.title("Text Embeddings Visualization")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    extracted_text = extract_text_from_pdf(uploaded_file)
    # Set the extracted text as the default value for the paragraph text_area
    paragraph = extracted_text
    # Text input for the paragraph
    paragraph = st.text_area(
        "Enter text to embed and visualize:", value=paragraph, height=300
    )
else:
    # Text input for the paragraph with an empty default value
    paragraph = st.text_area("Enter text to embed and visualize:", height=300)


question = st.text_area("Enter a question to embed and highlight:", height=75)


n_components = 3  # Number of dimensions for UMAP reduction
with st.sidebar:
    st.title("Settings")
    with st.expander("Chunk Settings"):
        chunk_size = st.slider(
            "Chunk Size", min_value=16, max_value=2048, value=512, step=16
        )
        stride = st.slider("Stride", min_value=0, max_value=1024, value=8, step=1)

    with st.expander("UMAP Settings"):
        n_neighbors = st.slider(
            "UMAP n_neighbors", min_value=5, max_value=50, value=15, step=1
        )
        min_dist = st.slider(
            "UMAP min_dist", min_value=0.0, max_value=0.99, value=0.1, step=0.01
        )

    with st.expander("Visualization Settings"):
        top_k = st.slider(
            "Top-k nearest nodes", min_value=1, max_value=100, value=5, step=1
        )
        threshold = st.slider(
            "Threshold", min_value=0.0, max_value=5.0, value=0.5, step=0.1
        )


# Button to trigger embeddings and visualization
if st.button("Generate Embeddings and Visualize"):
    if paragraph:
        openai_api_key = os.getenv("OPENAI_API_KEY")

        embeddings, rolling_chunks = embed_text_chunks_rolling_window(
            paragraph, openai_api_key, chunk_size=chunk_size, stride=stride
        )

        # Check if embeddings are empty
        if not embeddings:
            st.error(
                "No embeddings were generated. Please check the input text and try again."
            )
            st.stop()

        st.write(f"Number of embeddings: {len(embeddings)}")
        embedding_array = np.array(embeddings)

        # Calculate the distance from each embedding to the question embedding before UMAP
        question_embedding = embed_text_chunks(question, openai_api_key)[0]
        distances = np.linalg.norm(embedding_array - question_embedding, axis=1)

        pre_umap_distances = np.linalg.norm(
            embedding_array - question_embedding, axis=1
        )
        # Get the indices of the nearest chunks before UMAP
        pre_umap_nearest_indices = np.argsort(pre_umap_distances)[:top_k]

        # UMAP reduction
        reducer = UMAP(
            n_neighbors=n_neighbors, n_components=n_components, metric="euclidean"
        )
        embedding_3d = reducer.fit_transform(embedding_array)
        question_embedding_3d = reducer.transform(
            [question_embedding]
        )  # Transform the question embedding
        question_x, question_y, question_z = question_embedding_3d[0]

        # Create a colormap for the rainbow cycle
        colormap = plt.cm.rainbow
        norm = plt.Normalize(vmin=0, vmax=len(embedding_3d) - 1)

        # Calculate distances between consecutive nodes after UMAP reduction

        # Initialize the first color
        current_color = colormap(norm(0))
        consecutive_distances = [
            (
                np.linalg.norm(embedding_3d[i] - embedding_3d[i + 1])
                if i < len(embedding_3d) - 1
                else 0
            )
            for i in range(len(embedding_3d) - 1)
        ]
        # Define line colors based on consecutive distances
        line_colors = [current_color]
        for i, distance in enumerate(consecutive_distances):
            if distance > threshold:
                current_color = colormap(norm(i + 1))
            line_colors.append(current_color)

        # Convert RGBA to RGB color format for Plotly
        line_colors = [
            f"rgb({int(255*col[0])}, {int(255*col[1])}, {int(255*col[2])})"
            for col in line_colors
        ]

        # Calculate the distance from each embedding to the question embedding after UMAP
        x_coords, y_coords, z_coords = (
            embedding_3d[:, 0],
            embedding_3d[:, 1],
            embedding_3d[:, 2],
        )
        umap_coords = {
            "question": [question_x, question_y, question_z],
            "nodes": list(zip(x_coords, y_coords, z_coords)),
        }

        # Create a trace for the scatter plot with markers and lines
        trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers+lines",
            marker=dict(size=5, opacity=0.8),
            line=dict(
                width=4,
                color=line_colors,  # Apply the line colors
            ),
            text=[
                f"Node {i}:<br>{wrap_text(chunk, 120)}"  # Adjust 120 to your preferred line length
                for i, chunk in enumerate(rolling_chunks, start=1)
            ],
            hoverinfo="text",
        )
        question_trace = go.Scatter3d(
            x=[question_x],
            y=[question_y],
            z=[question_z],
            mode="markers",
            marker=dict(size=10, color="green", opacity=1),
            name="Question",
        )

        distances = np.linalg.norm(embedding_array - question_embedding, axis=1)
        # Get the indices of the 5 nearest chunks
        nearest_indices = np.argsort(distances)[:5]

        # Define the layout of the plot
        layout = go.Layout(
            title="Text Embeddings Visualization",
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(
                    title="UMAP X",
                    backgroundcolor="rgb(200, 200, 230)",
                    gridcolor="grey",
                ),
                yaxis=dict(
                    title="UMAP Y",
                    backgroundcolor="rgb(230, 200,230)",
                    gridcolor="grey",
                ),
                zaxis=dict(
                    title="UMAP Z",
                    backgroundcolor="rgb(230, 230,200)",
                    gridcolor="grey",
                ),
                aspectmode="cube",
            ),
            legend=dict(x=1, y=0.9),
            hovermode="closest",
        )

        # Create the figure with the trace and layout
        fig = go.Figure(data=[trace, question_trace], layout=layout)

        lines_to_nearest_nodes = create_lines_to_nearest_nodes(
            question_embedding,  # Use the original question embedding
            pre_umap_nearest_indices,
            embedding_array,  # Original embeddings, not reduced
            umap_coords,  # UMAP-reduced coordinates
        )

        fig.add_traces(lines_to_nearest_nodes)
        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Please enter text to visualize.")

    high_dim_distances = [
        np.linalg.norm(embedding_array[i] - embedding_array[i + 1])
        for i in range(len(embedding_array) - 1)
    ]

    # Generate the x-axis values which represent each node
    x_values = list(range(1, len(embedding_array)))

    # Create a plot for the distances
    fig_distances = go.Figure(
        data=go.Scatter(
            x=x_values,
            y=high_dim_distances,
            mode="lines+markers",
            name="High-Dimensional Distances",
        ),
        layout=go.Layout(
            title="High-Dimensional Distances Between Consecutive Nodes",
            xaxis=dict(title="Node"),
            yaxis=dict(title="Distance"),
            margin=dict(l=40, r=40, t=40, b=40),
        ),
    )

    # Calculate the Z-scores of the distances
    z_scores = np.abs(stats.zscore(high_dim_distances))

    # Set a threshold for what you consider an anomaly
    threshold = 2

    # Find indices where the distance changes are considered anomalies
    anomalies = np.where(z_scores > threshold)[0]

    # Generate the x-axis values which represent each node
    x_values = list(range(1, len(embedding_array) + 1))  # +1 to include the last node

    # Create a plot for the distances
    fig_distances = go.Figure()

    # Add the original distance plot
    fig_distances.add_trace(
        go.Scatter(
            x=x_values,
            y=high_dim_distances,
            mode="lines+markers",
            name="High-Dimensional Distances",
        )
    )

    # Initialize a color cycle for the bars
    color_cycle = plt.cm.rainbow(np.linspace(0, 1, len(anomalies) + 1))

    # Overlay semi-transparent colored bars for anomalies
    for i, anomaly_index in enumerate(anomalies):
        # Determine the end of the colored bar (next anomaly or end of the range)
        next_anomaly_index = (
            anomalies[i + 1] if i < len(anomalies) - 1 else len(high_dim_distances)
        )

        color = f"rgba({int(255*color_cycle[i][0])}, {int(255*color_cycle[i][1])}, {int(255*color_cycle[i][2])}, 0.5)"  # Semi-transparent color
        fig_distances.add_vrect(
            x0=x_values[anomaly_index]
            - 0.5,  # Subtract 0.5 to center the bar on the node
            x1=(
                x_values[next_anomaly_index] - 0.5
                if next_anomaly_index < len(x_values)
                else x_values[-1] + 0.5
            ),  # Extend to the next anomaly or to the end
            fillcolor=color,
            opacity=0.5,
            layer="below",
            line_width=0,
        )

    # Update the layout to include the title and axis labels
    fig_distances.update_layout(
        title="High-Dimensional Distances Between Consecutive Nodes with Anomalies",
        xaxis=dict(title="Node"),
        yaxis=dict(title="Distance"),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # Display the distance plot with anomalies in Streamlit
    st.plotly_chart(fig_distances, use_container_width=True)
    st.markdown("---")
    for i, chunk in enumerate(rolling_chunks, start=1):
        # Use the color from the line_colors array
        color_rgb = line_colors[
            i - 1
        ]  # This should already be a string in 'rgb(r, g, b)' format

        st.markdown(
            f"<div style='color:{color_rgb};'>---- Node {i} ----<br>{chunk}</div>",
            unsafe_allow_html=True,
        )
