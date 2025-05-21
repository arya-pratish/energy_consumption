FROM python:3.9
WORKDIR /opt/flask_predictor
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY data.csv .
COPY model.py .
RUN python3.9 model.py
RUN mkdir templates
COPY index.html templates
COPY result.html templates
COPY flask_app.py .
ENTRYPOINT python3.9 flask_app.py

