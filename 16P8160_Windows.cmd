@echo off
echo "||||||||||||||||||||||||||||||||||||||||||||||||||||"
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "|||||||||||||- WILL TEST TRAINING NOW -|||||||||||||"
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "||||||||||||||||||||||||||||||||||||||||||||||||||||"
TIMEOUT 5
@echo on
python ./code/Train.py
@echo off
echo "||||||||||||||||||||||||||||||||||||||||||||||||||||"
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "|||||||||||||- WILL TEST INFERENCE NOW USING MY SAMPLE -|||||||||||||"
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "||||||||||||||||||||||||||||||||||||||||||||||||||||"
TIMEOUT 5
@echo on
python ./code/Inference.py D:\Learning\DL\16p8160_project\code\trainingSet\7\img_10015.jpg
@echo off
echo "||||||||||||||||||||||||||||||||||||||||||||||||||||"
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "|||||||||||||- TEST INFERENCE USING YOUR SAMPLE -|||||||||||||"
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "||||||||||||||||||||||||||||||||||||||||||||||||||||"
TIMEOUT 5
@echo on
python ./code/Inference.py
@echo off
echo "||||||||||||||||||||||||||||||||||||||||||||||||||||"
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "|||||||||||||- TEST TENSORBOARD -|||||||||||||"
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "||||||||||||||||||||||||||||||||||||||||||||||||||||"
TIMEOUT 5
@echo on
explorer "http://127.0.0.1:6060"
python -m tensorboard.main --logdir="./code/" --host localhost --port 6060
PAUSE