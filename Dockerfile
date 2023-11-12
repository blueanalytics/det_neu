FROM python:3.11.4

RUN apt-get update -y && \
    apt-get install python3-opencv -y 

WORKDIR /home/src

COPY . ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando para ejecutar tu aplicación
CMD ["python", "main.py"]