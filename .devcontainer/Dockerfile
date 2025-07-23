ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && cp /etc/skel/.bashrc /home/$USERNAME/ \
    && cp /etc/skel/.profile /home/$USERNAME/ \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

# Set environment variables for ROCm
ENV PATH="/opt/rocm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"

# Install any additional system dependencies if needed (e.g., git, curl)
RUN apt-get update && apt-get install -y --no-install-recommends jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Set the working directory
WORKDIR /workspace

CMD ["bash"]