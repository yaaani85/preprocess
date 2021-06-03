requirements:
	pipenv lock -r > requirements.txt

build:              
	brane unpublish -f preprocessing 1.0.0
	brane remove -f preprocessing
	brane build container.yml
	brane push preprocessing 1.0.0
