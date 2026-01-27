# Setup Guide

## Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher (for Flash Attention and compilation support)
- CUDA-capable GPU (recommended for training)

## Installation

1. Clone the repository:
   git clone https://github.com/udaykiriti/NanoTextLM.git
   cd NanoTextLM

2. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

## Docker Setup

For a consistent environment, you can use Docker.

1. Build the image:
   docker build -t nanotextlm .

2. Run the container (exposing port 5000 for the web interface):
   docker run -p 5000:5000 --gpus all nanotextlm

Note: The `--gpus all` flag requires the NVIDIA Container Toolkit. If running on CPU, omit this flag.
