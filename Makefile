NVCC_RESULT := $(shell which nvcc 2> NULL; rm NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus '"device=7"'
else
GPUS=
endif


# Set flag for docker run command
MYUSER=myuser
SERVER_NAME = $(shell hostname)
# If using a remote server, set data directory to /path/to/data, otherwise assume data is on same level as the repo
ifeq ($(SERVER_NAME),gpu-node-12)
DATADIR=/path/to/data
else
DATADIR=~/data
endif
SCRATCH_DIR=~/scratch_LOB
GYMNAX_DIR=/path/to/AlphaTrade/gymnax_exchange
LOBBENCH_DIR=~/lob_bench
BASE_FLAGS=-it --rm \
	-v ${PWD}:/home/$(MYUSER) \
	-v $(DATADIR):/home/$(MYUSER)/data \
	-v $(SCRATCH_DIR):/home/$(MYUSER)/scratch \
	-v $(GYMNAX_DIR):/home/$(MYUSER)/gymnax_exchange \
	-v $(LOBBENCH_DIR):/home/$(MYUSER)/lob_bench \
	--shm-size 20G
PORT_FLAGS= -p 8060:80 -p 8064:6006
RUN_FLAGS=$(BASE_FLAGS) $(PORT_FLAGS) $(GPUS) 
BASIC_FLAGS=$(GPUS) $(BASE_FLAGS)


DOCKER_IMAGE_NAME = lobs5
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
DOCKER_RUN_BASIC=docker run --gpus '"device=$(gpu)"' $(BASE_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)
ID = $(shell id -u)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile_LOBS5 --build-arg USE_CUDA=$(USE_CUDA) --build-arg MYUSER=$(MYUSER) --build-arg UID=$(ID) --tag $(IMAGE) --progress=plain ${PWD}/. 
	

run:
	$(DOCKER_RUN) /bin/bash

test:
	$(DOCKER_RUN) /bin/bash -c "pytest ./tests/"

train_small:
	$(DOCKER_RUN_BASIC) /bin/bash -c "sh bin/run_experiments/run_lobster_padded_small.sh"

train_large:
	$(DOCKER_RUN_BASIC) /bin/bash -c "sh bin/run_experiments/run_lobster_padded_large.sh"

inference:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./run_inference.py --stock AMZN --checkpoint_step 37 --test_split 1"

eval:
	$(DOCKER_RUN_BASIC) /bin/bash -c "sh bin/eval_bash/eval_local.sh"

benchmark:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 lob_bench/run_bench.py --stock AMZN --model_version ruby-aardvark --data_dir ./eval_local --save_dir ./benchmark_local/"

workflow-test:
	# without -it flag
	docker run --rm -v ${PWD}:/home/workdir --shm-size 20G $(IMAGE) /bin/bash -c "pytest ./tests/"