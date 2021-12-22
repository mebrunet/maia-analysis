#!make

SHELL = /bin/bash


# *****************************************************************************
# Configurable Parameters
# You should edit this for your project
# *****************************************************************************

# Set a jupyter token for easy access
JUPYTER_TOKEN = my_secret_token
JUPYTER_PORT = 8888

TUNNEL_DEST = ada
TUNNEL_SERVER = cs.toronto.edu
TUNNEL_TMP_FILE = .port-forward.ssh.tmp

# Conda enviroment name
CONDA_ENV_NAME = maia-analysis


# *****************************************************************************
# Helper Functions
# *****************************************************************************

define _ip
hostname -I | tr -d [:space:]
endef

define _url
Jupyter Lab: http://$(1):$(JUPYTER_PORT)/lab/?token=$(JUPYTER_TOKEN)
endef

# Help
.PHONY: usage
usage:
	@ echo "A collection of helpers."
	@ echo "TODO: write a useful usage message."


# *****************************************************************************
# Dependencies Related
# *****************************************************************************
.PHONY: dep.install
dep.install:
	conda env update -n $(CONDA_ENV_NAME) -f environment.yml


# *****************************************************************************
# Jupyter
# *****************************************************************************
.PHONY: jupyter.start
jupyter.start:
	@ echo Starting $(call _url,$(call _ip))
	jupyter notebook --NotebookApp.token=$(JUPYTER_TOKEN) --no-browser --notebook-dir '/' \
	--ip 0.0.0.0 --port $(JUPYTER_PORT)

.PHONY: jupyter.url
jupyter.url:
	@ echo $(call _url,'localhost')


# *****************************************************************************
# Port-forwarding
# *****************************************************************************
.PHONY: tunnel.open
tunnel.open:
	@ echo directing localhost:8888 to $(TUNNEL_DEST):8888
	ssh -M -S $(TUNNEL_TMP_FILE) -fN -L 8888:$(TUNNEL_DEST):8888 $(TUNNEL_SERVER) \
		&& echo tunnel established

.PHONY: tunnel.close
tunnel.close:
	@ ssh -S $(TUNNEL_TMP_FILE) -O exit $(TUNNEL_SERVER) && echo tunnel closed


# *****************************************************************************
# Run tests
# *****************************************************************************
.PHONY: test
test:
	pytest test
