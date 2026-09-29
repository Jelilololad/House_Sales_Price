FROM kserve/sklearnserver:v0.16.0

USER root

# Install OpenMP runtime required by LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Use global pip to install lightgbm directly into /prod_venv's site-packages
RUN /usr/local/bin/pip install --no-cache-dir --target=/prod_venv/lib/python3.11/site-packages lightgbm

# Copy custom wrangler module
RUN mkdir -p /opt/model-code
COPY src/wrangler.py /opt/model-code/wrangler.py

ENV PYTHONPATH="/opt/model-code:${PYTHONPATH}"

USER 1000