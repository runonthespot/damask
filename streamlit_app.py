import streamlit as st
import openai
import numpy as np
import plotly.graph_objs as go
from umap import UMAP
import tiktoken
import os
import fitz

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


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

    # Split the tokens into chunks with a rolling window
    chunks = [
        tokens[i : i + chunk_size]
        for i in range(0, len(tokens) - chunk_size + 1, stride)
    ]

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


@st.cache_data
def embed_text_chunks_rolling_window(
    text: str, openai_api_key: str, chunk_size: int = 128, stride: int = 1
) -> tuple:
    openai.api_key = openai_api_key
    chunks = chunk_text_rolling_window(text, chunk_size=chunk_size, stride=stride)
    embeddings = []

    # Initialize the progress bar
    progress_bar = st.progress(0)

    for i, chunk in enumerate(chunks):
        response = openai.embeddings.create(input=chunk, model="text-embedding-3-large")
        embedding_vector = response.data[0].embedding
        embeddings.append(embedding_vector)

        # Update the progress bar
        progress_bar.progress((i + 1) / len(chunks))

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


def create_colored_line_segments(x, y, z, threshold):
    # Calculate the distances between consecutive points
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)

    # Initialize lists to hold the segmented coordinates and colors
    segmented_x = []
    segmented_y = []
    segmented_z = []
    colors = []

    # Initialize the current segment
    current_segment_x = [x[0]]
    current_segment_y = [y[0]]
    current_segment_z = [z[0]]

    # Use a default color for the first segment
    current_color = "blue"

    # Iterate over the points
    for i, distance in enumerate(distances):
        # Add the next point to the current segment
        current_segment_x.append(x[i + 1])
        current_segment_y.append(y[i + 1])
        current_segment_z.append(z[i + 1])

        # If the distance exceeds the threshold, finish the current segment
        if distance > threshold:
            # Append the current segment and its color to the lists
            segmented_x.append(current_segment_x)
            segmented_y.append(current_segment_y)
            segmented_z.append(current_segment_z)
            colors.append(current_color)

            # Start a new segment with the last point of the previous segment
            current_segment_x = [x[i + 1]]
            current_segment_y = [y[i + 1]]
            current_segment_z = [z[i + 1]]

            # Append a grey segment to represent the gap
            segmented_x.append([x[i], x[i + 1]])
            segmented_y.append([y[i], y[i + 1]])
            segmented_z.append([z[i], z[i + 1]])
            colors.append("grey")  # Grey color for the gap

            # Reset the current color for the next segment
            current_color = "blue" if current_color == "red" else "red"

    # Append the last segment
    segmented_x.append(current_segment_x)
    segmented_y.append(current_segment_y)
    segmented_z.append(current_segment_z)
    colors.append(current_color)

    # Create a trace for each segment
    traces = []
    for seg_x, seg_y, seg_z, color in zip(
        segmented_x, segmented_y, segmented_z, colors
    ):
        traces.append(
            go.Scatter3d(
                x=seg_x, y=seg_y, z=seg_z, mode="lines", line=dict(width=2, color=color)
            )
        )

    return traces


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

st.sidebar.title("Settings")

# Sliders for chunk size and stride

# Sliders for chunk size and stride
chunk_size = st.sidebar.slider(
    "Chunk Size", min_value=128, max_value=2048, value=512, step=64
)
stride = st.sidebar.slider("Stride", min_value=0, max_value=1024, value=8, step=1)

threshold = st.sidebar.slider(
    "Threshold", min_value=0.0, max_value=5.0, value=0.5, step=0.1
)

n_neighbors = st.sidebar.slider(
    "UMAP n_neighbors", min_value=5, max_value=50, value=15, step=1
)
n_components = st.sidebar.slider(
    "UMAP n_components", min_value=2, max_value=3, value=3, step=1
)
min_dist = st.sidebar.slider(
    "UMAP min_dist", min_value=0.0, max_value=0.99, value=0.1, step=0.01
)
top_k = st.sidebar.slider(
    "Top-k nearest nodes", min_value=1, max_value=100, value=5, step=1
)


# Button to trigger embeddings and visualization
if st.button("Generate Embeddings and Visualize"):
    # Check if a file has been uploaded

    if paragraph:
        # Assuming you have set your OpenAI API key in the environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        # Get the rolling window chunks and embeddings
        embeddings, rolling_chunks = embed_text_chunks_rolling_window(
            paragraph, openai_api_key, chunk_size=chunk_size, stride=stride
        )

        st.write(f"Number of embeddings: {len(embeddings)}")
        # Convert embeddings to numpy array for UMAP
        embedding_array = np.array(embeddings)

        # Calculate the distance from each embedding to the question embedding before UMAP
        question_embedding = embed_text_chunks(question, openai_api_key)[0]

        pre_umap_distances = np.linalg.norm(
            embedding_array - question_embedding, axis=1
        )
        # Get the indices of the 5 nearest chunks before UMAP
        pre_umap_nearest_indices = np.argsort(pre_umap_distances)[:top_k]

        # UMAP reduction
        reducer = UMAP(
            n_neighbors=n_neighbors, n_components=n_components, metric="euclidean"
        )
        embedding_2d = reducer.fit_transform(embedding_array)
        question_embedding_2d = reducer.transform(
            [question_embedding]
        )  # Transform the question embedding
        question_x, question_y, question_z = question_embedding_2d[0]

        # Calculate the distance from each embedding to the question embedding after UMAP
        x_coords, y_coords, z_coords = (
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            embedding_2d[:, 2],
        )

        # Create a trace for the scatter plot with markers and lines
        trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers+lines",
            marker=dict(size=5, opacity=0.8),
            line=dict(width=2, color="blue"),
            text=[
                f"Node {i}:<br>{wrap_text(chunk, 120)}"  # Adjust 30 to your preferred line length
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
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(title="UMAP X"),
                yaxis=dict(title="UMAP Y"),
                zaxis=dict(title="UMAP Z"),
            ),
        )

        # Create the figure with the trace and layout
        fig = go.Figure(data=[trace, question_trace], layout=layout)

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Please enter text to visualize.")

        # Print the text chunks with separators, node numbers, and colour
        # Print the text chunks with separators, node numbers, and colour
    st.markdown("---")
    for i, chunk in enumerate(rolling_chunks, start=1):
        # Check if the current chunk index is in the top-k nearest indices
        color = "#00FFFF" if i - 1 in pre_umap_nearest_indices else "#FFFFFF"
        st.markdown(
            f"<div style='color:{color};'>---- Node {i} ----<br>{chunk}</div>",
            unsafe_allow_html=True,
        )
