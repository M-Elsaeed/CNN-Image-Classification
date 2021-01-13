docker build -f ./dockerfiles/cpu.Dockerfile -t 16p8160 .
docker run -t -i --privileged 16p8160 bash /home/16p8160_runscript.sh
pause