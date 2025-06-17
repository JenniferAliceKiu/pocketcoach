run_preprocess:
	python -c 'from pocketcoach.main import preprocess; preprocess()'




# call like that: make test_predict TEXT="I love my wife soooooo much"
.PHONY: test_predict
test_predict:
	@if [ -z "$(TEXT)" ]; then \
	  echo "Usage: make test_predict TEXT=\"your text here\""; \
	  exit 1; \
	fi
	python -c 'import sys; from pocketcoach.main import classify; classify(sys.argv[1])' "$(TEXT)"

run_server_locally:
	uvicorn api.fast:app --reload

docker_build_local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local .

DOCKER_IMAGE_PATH := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME)

docker_show_image_path:
	@echo $(DOCKER_IMAGE_PATH)

#		--platform linux/amd64
docker_build:
	docker build \
		-t $(DOCKER_IMAGE_PATH):prod .

docker_run:
	docker run \
		-p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		-v $(GOOGLE_APPLICATION_CREDENTIALS):/app/credentials.json \
		-e GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json" \
		-e PORT=8080 \
		$(DOCKER_IMAGE_NAME)

docker_allow:
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev

docker_create_repo:
	gcloud artifacts repositories create $(DOCKER_REPO_NAME) \
		--repository-format=docker \
		--location=$(GCP_REGION) \
		--description="Repository for storing docker images"

docker_push:
	docker push $(DOCKER_IMAGE_PATH):prod

docker_deploy:
	gcloud run deploy \
		--image $(DOCKER_IMAGE_PATH):prod \
		--memory $(GAR_MEMORY) \
		--region $(GCP_REGION) \
		--env-vars-file .env.yaml
