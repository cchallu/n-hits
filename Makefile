IMAGE := nhits
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PARENT_ROOT := $(shell dirname ${ROOT})
PORT := 8888

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/app \
	-w /app \
	-e HOME=/tmp

ifdef gpu
	DOCKER_PARAMETERS += --gpus all
endif

init:
	docker build -t ${IMAGE} .

get_dataset:
	$(MAKE) run_module module="mkdir -p data/"
	$(MAKE) run_module module="wget  -O data/datasets.zip https://nhits-experiments.s3.amazonaws.com/datasets.zip"
	$(MAKE) run_module module="unzip data/datasets.zip -d data/"

jupyter:
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p ${PORT}:8888 ${IMAGE} \
		bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

run_module: .require-module
	docker run -i --rm ${DOCKER_PARAMETERS} \
		${IMAGE} ${module}

bash_docker:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE}

.require-module:
ifndef module
	$(error module is required)
endif
