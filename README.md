# age-gender-predictor-demo
![](demo_movie.gif)
## installation
### if you use conda
* execute below command
```terminal
conda env create -f environment.yml
conda activate demo_env
```
### else you use pure-python>=3.x
```terminal
pip install -r requirements.txt
```

## how to use
```terminal
python3 video2rec.py
```
### if you want change video source
please change VIDEO_SOURCE constant of video2rec.py 
if you use usb-video-cam: 0
else if html video source of network cam or droid-cam: rtsp://~~ or http://~
