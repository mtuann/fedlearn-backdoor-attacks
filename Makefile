.ONESHELL:

CONDAPATH = $$(conda info --base)
ENV_NAME = aba

install:
	conda env create -f environment.yml
	${CONDAPATH}/envs/$(ENV_NAME)/bin/pip install -r requirements.txt

install-mac:
	conda env create -f environment.yml
	conda install nomkl
	${CONDAPATH}/envs/$(ENV_NAME)/bin/pip install -r requirements.txt

update:
	# conda env update --prune -f environment.yml
	${CONDAPATH}/envs/$(ENV_NAME)/bin/pip install -r requirements.txt --upgrade

clean:
	conda env remove --name $(ENV_NAME)