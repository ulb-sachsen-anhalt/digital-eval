.ONESHELL:


all: languagetool-start

languagetool-start: FORCE
	docker run -d -p 8010:8010 --name languagetool silviof/docker-languagetool

languagetool-stop: FORCE
	docker stop languagetool
	docker remove languagetool

languagetool-restart: FORCE languagetool-stop languagetool-start

languagetool-start-at-49151: FORCE
	docker run -d -p 49151:8010 --name languagetool silviof/docker-languagetool

remove-and-stop-all-running-containers:
	docker ps -aq | xargs docker rm -f

.PHONY: FORCE
FORCE:
