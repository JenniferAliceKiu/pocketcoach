run_preprocess:
	python -c 'from pocketcoach.main import preprocess; preprocess()'




# call like that: make test_predict TEXT="I love my wife soooooo much"
.PHONY: test_predict
test_predict
	@if [ -z "$(TEXT)" ]; then \
	  echo "Usage: make test_predict TEXT=\"your text here\""; \
	  exit 1; \
	fi
	python -c 'import sys; from pocketcoach.main import predict; predict(sys.argv[1])' "$(TEXT)"

run_server_locally:
	uvicorn api.fast:app --reload
