NS ?= jihong
REPO = keras-gpu
VERSION ?= latest

DOCKER=nvidia-docker

PORTS = -p 8888:8888
VOLUMES = -v `pwd`:/root/workspace \
	-v ${HOME}/Datasets:/root/workspace/data
ENV = -e KERAS_BACKEND=tensorflow

.PHONY: setup bash notebook test

setup:
	$(DOCKER) run --rm -it $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(REPO):$(VERSION) /root/workspace/setup_env.sh

bash:
	$(DOCKER) run --rm -it $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(REPO):$(VERSION) bash

notebook: 
	$(DOCKER) run --rm -it $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(REPO):$(VERSION) /root/workspace/run_notebook.sh

test: 
	$(DOCKER) run --rm -it $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(REPO):$(VERSION) /root/workspace/run_tests.sh

