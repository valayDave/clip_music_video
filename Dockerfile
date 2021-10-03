FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
ADD requirements.txt /requirements.txt
RUN pip install DALL-E
RUN pip install git+https://github.com/openai/CLIP.git 
USER root
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get install -y \
    imagemagick libmagickwand-dev --no-install-recommends \
    && pecl install imagick \
    && docker-php-ext-enable imagick
RUN pip install -r /requirements.txt
CMD ["python3"]