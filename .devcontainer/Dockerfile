FROM mcr.microsoft.com/devcontainers/anaconda:0-3

# Check if the files exist before trying to copy them
COPY ./ ./
RUN mkdir -p /tmp/conda-tmp/

RUN if [ -f "environment.yml" ]; then cp environment.yml /tmp/conda-tmp/; fi
RUN if [ -f ".devcontainer/noop.txt" ]; then cp .devcontainer/noop.txt /tmp/conda-tmp/; fi

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
