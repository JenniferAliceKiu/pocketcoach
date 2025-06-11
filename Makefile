run_preprocess:
	python -c 'from pocketcoach.main import preprocess; preprocess()'


test_predict:
	python -c 'from pocketcoach.main import predict; predict("I hate my dog")'
