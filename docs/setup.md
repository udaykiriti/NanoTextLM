# Setup Guide

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with CUDA support recommended)
- Docker (optional)

## Local Installation

1. Clone the repository:
   git clone https://github.com/udaykiriti/NanoTextLM.git
   cd NanoTextLM

2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

## Docker

1. Build the image:
   docker build -t nanotextlm .

2. Run the container:
   docker run -p 5000:5000 --gpus all nanotextlm