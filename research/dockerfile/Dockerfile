FROM gcr.io/tensorflow/tensorflow:1.5.0-gpu-py3
MAINTAINER notname "notname@gmail.com"
ENV data /data
WORKDIR data
# Update aptitude with new repo
RUN apt-get update

# Install software 
RUN apt-get install -y git
# Make ssh dir
RUN mkdir /root/.ssh/

# Copy over private key, and set permissions
RUN apt-get -y install openssh-client
RUN ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa

# Create known_hosts
RUN touch /root/.ssh/known_hosts
# Generate ssh key
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# clone latest code 
RUN git clone https://github.com/notnamegg/notnamegg-tensorflow-Distributed-training.git
RUN cd notnamegg-tensorflow-Distributed-training && git submodule init
RUN cd notnamegg-tensorflow-Distributed-training && git submodule update
RUN cd notnamegg-tensorflow-Distributed-training/models && git remote add github https://github.com/notnamegg/tensorflow-model.git
RUN cd notnamegg-tensorflow-Distributed-training/models && git fetch github
RUN cd notnamegg-tensorflow-Distributed-training/models && git pull github master
