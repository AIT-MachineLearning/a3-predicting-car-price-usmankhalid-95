FROM python:3.12.4-bookworm

WORKDIR /root/source_code

RUN pip3 install utils
RUN pip3 install scikit-learn
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install mlflow
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip install streamlit
RUN pip install notebook
RUN pip3 install mlflow

RUN pip install --upgrade pip

COPY ./.devcontainer/source_code /root/source_code

CMD tail -f /dev/null
EXPOSE 8501