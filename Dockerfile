FROM continuumio/anaconda3:latest

WORKDIR /workspace

ENV CONDA_ENV_NAME=llmtrain
ENV PATH=/opt/conda/envs/llmtrain/bin:$PATH

RUN conda create -n llmtrain python=3.11 -y && \
    /opt/conda/envs/llmtrain/bin/pip install --upgrade pip setuptools wheel && \
    /opt/conda/envs/llmtrain/bin/pip install \
        torch \
        numpy \
        tqdm \
        matplotlib \
        tokenizers \
        fastapi \
        uvicorn \
        pydantic \
        jupyterlab \
        notebook \
        ipykernel && \
    /opt/conda/envs/llmtrain/bin/python -m ipykernel install \
        --name llmtrain \
        --display-name "Python (llmtrain)" \
        --sys-prefix && \
    conda clean -afy

EXPOSE 6888

CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=6888 --no-browser --allow-root --notebook-dir=/workspace --ServerApp.token=${JUPYTER_TOKEN} --ServerApp.password=''"]
