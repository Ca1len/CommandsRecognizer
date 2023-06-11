FROM python:3.11

RUN mkdir /work
RUN mkdir /work/App

WORKDIR /work/App

COPY include include
COPY weights weights
COPY main.py main.py

# COPY requirements.txt requirements.txt
RUN python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install numpy 

CMD ["python", "-u", "main.py"]
