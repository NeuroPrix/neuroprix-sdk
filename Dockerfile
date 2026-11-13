# NeuroPrix Runner in a container: for a machine you rent (or any computer with Docker).
#   docker build -t neuroprix-runner .
#   docker run -it -v neuroprix:/home/runner/.neuroprix neuroprix-runner --server https://neuroprix.com/api
# The volume keeps the runner's link to your account, so it doesn't ask for a new code each time it starts.
# Your own trainer: mount it and pass --trainer, e.g.
#   docker run -it -v neuroprix:/home/runner/.neuroprix -v $PWD/my_trainer.py:/work/my_trainer.py:ro neuroprix-runner \
#       --server https://neuroprix.com/api --trainer "python /work/my_trainer.py"
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# CPU-only PyTorch: the brains are small, and the time goes into simulating the game, not into the network
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install "numpy>=1.26"
WORKDIR /src
COPY . /src
RUN pip install /src && rm -rf /src
# Not root: the runner and its trainer only need their own folder
RUN useradd --create-home --uid 10001 runner && mkdir -p /home/runner/.neuroprix /work \
    && chown runner:runner /home/runner/.neuroprix /work && chmod 700 /home/runner/.neuroprix
USER runner
WORKDIR /work
ENTRYPOINT ["neuroprix-runner"]
