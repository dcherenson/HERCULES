# syntax=docker/dockerfile:1
FROM ros:humble-ros-base-jammy
SHELL ["/bin/bash", "-c"]
ARG DEV_UID=1000
ARG DEV_GID=1000
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config git curl unzip sudo \
    python3-colcon-common-extensions python3-rosdep python3-pytest python3-pip \
    python3-numpy python3-matplotlib python3-pil ffmpeg \
    libeigen3-dev gdb gdbserver clangd procps sysstat iproute2 netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir \
    osqp==1.0.4 jinja2==3.1.6 joblib==1.5.1 MarkupSafe==3.0.2 \
    rpc-msgpack==0.6
# Dockerfile.dockerignore makes this a manifest-only tree, not a source copy.
COPY ros2/src/ /tmp/ros-dependencies/src/
RUN source /opt/ros/humble/setup.bash && \
    (test -f /etc/ros/rosdep/sources.list.d/20-default.list || rosdep init) && \
    rosdep update --rosdistro humble && apt-get update && \
    rosdep install --from-paths /tmp/ros-dependencies/src --ignore-src --rosdistro humble -y && \
    rm -rf /var/lib/apt/lists/* /tmp/ros-dependencies
RUN groupadd --non-unique --gid ${DEV_GID} developer && \
    useradd --non-unique --uid ${DEV_UID} --gid ${DEV_GID} --create-home --shell /bin/bash developer && \
    echo 'developer ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/developer && \
    cp -a /root/.ros /home/developer/.ros && chown -R ${DEV_UID}:${DEV_GID} /home/developer/.ros
COPY docker/ros2/entrypoint.sh /usr/local/bin/hercules-entrypoint
RUN chmod +x /usr/local/bin/hercules-entrypoint && \
    echo 'source /workspaces/hercules/docker/ros2/env.sh' >> /home/developer/.bashrc
USER developer
WORKDIR /workspaces/hercules
ENTRYPOINT ["/usr/local/bin/hercules-entrypoint"]
CMD ["sleep", "infinity"]
