# Load local env file.
ifneq (,$(wildcard ./.env))
	include .env
	export
endif


INSTANCE_NAMES := $(INSTANCE_NAMES)
ADDRESS := $(ADDRESS)
SSH_KEY_PATH := $(SSH_KEY_PATH)
STARTUP_SCRIPT := $(abspath $(STARTUP_SCRIPT))
ZONE := asia-east1-c
MACHINE_TYPE := n1-highmem-8
GPU_TYPE := nvidia-tesla-v100


create-instance:
	gcloud compute instances create $(INSTANCE_NAMES) \
		--machine-type $(MACHINE_TYPE) \
		--zone $(ZONE) \
		--network-interface address=$(ADDRESS) \
		--metadata-from-file ssh-keys=$(SSH_KEY_PATH) \
		--accelerator type=$(GPU_TYPE),count=1 \
		--boot-disk-size 200GB \
		--image-family ubuntu-1804-lts \
		--image-project ubuntu-os-cloud \
		--maintenance-policy TERMINATE --restart-on-failure \

create-instance-preemptible:
	gcloud compute instances create --preemptible $(INSTANCE_NAMES) \
		--zone $(ZONE) \
		--network-interface address=$(ADDRESS) \
		--metadata-from-file ssh-keys=$(SSH_KEY_PATH) \
		--accelerator type=$(GPU_TYPE),count=1 \
		--boot-disk-size 200GB \
		--machine-type $(MACHINE_TYPE) \
		--image-family ubuntu-1804-lts \
		--image-project ubuntu-os-cloud

start-instance:
	gcloud compute instances start $(INSTANCE_NAMES) --zone $(ZONE)

stop-instance:
	gcloud compute instances stop $(INSTANCE_NAMES) --zone $(ZONE)

delete-instance:
	gcloud compute instances delete $(INSTANCE_NAMES) --zone $(ZONE)

connect-instance:
	gcloud compute ssh $(INSTANCE_NAMES) --zone $(ZONE)

tb-server:
	tensorboard --logdir ./tb_logs/

jn:
	poetry run jupyter lab

test:
	poetry run pytest -s --pdb tests/

