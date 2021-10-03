FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
ADD requirements.txt /requirements.txt
RUN pip install DALL-E
RUN pip install git+https://github.com/openai/CLIP.git 
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" TZ="Europe/London" apt-get install imagemagick -y
RUN pip install -r /requirements.txt
ADD edit_imagemagik.py /edit_imagemagik.py
RUN python /edit_imagemagik.py
CMD ["python3"]