[image1]: ./results_yolov3/test-image1.jpg "Yolo v3 test 1"
[image2]: ./results_yolov3/test-image2.jpg "Yolo v3 test 2"
[image3]: ./results_yolov3/test-image3.jpg "Yolo v3 test 3"
[image4]: ./results_yolov3/test-image4.jpg "Yolo v3 test 4"
[image5]: ./results_yolov4/test-image1.jpg "Yolo v4 test 1"
[image6]: ./results_yolov4/test-image2.jpg "Yolo v4 test 2"
[image7]: ./results_yolov4/test-image3.jpg "Yolo v4 test 3"
[image8]: ./results_yolov4/test-image4.jpg "Yolo v4 test 4"




# Face Mask Detector with YOLO

For this training I decided to go using a Docker image allocated [here](https://hub.docker.com/r/daisukekobayashi/darknet) by *daisukekobayashi*.

## Download Darknet

```
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
```

All following steps will be done inside the darknet folder.

## Downloading dataset

First step is to download an extract the dataset to dataset subfolder.

```
wget https://www.dropbox.com/s/uq0x32w70c390fb/mask_no-mask_dataset.zip?dl=1 -O dataset.zip
unzip dataset.zip -d dataset
```

## Prepare dataset

The dataset will be splitted between training and testing (validation) using the python script.

```
python prepare_dataset.py
```

The resulting files will be:

```
data_test.txt
data_train.txt
```

These files containing the paths to every image and its associated normalized bounding boxes with classes.

## Class definition

The file `class.names` contains the class definition 0 -> `mask` and 1 -> `No-mask`

```
Mask
No-mask
```

## Setup

For the training setup the `setup.data` contains the following definitions:

```
classes = 2
train = data_train.txt
valid = data_test.txt
names = class.names
backup = backup/

```

All these files must be stored in the root file of the `darknet` folder.

## YOLO v3

### Download weights

```
wget "https://www.dropbox.com/s/18dwbfth7prbf0h/darknet53.conv.74?dl=1" -O darknet53.conv.74
```

### Train and test condigurations

The files `mask_yolov3_test.cfg` and `mask_yolov3_train.cfg` were modified as follows:

- width=416
- height=416
- max_batches = 6000
- steps=4800,5400
- classes=2 (For every Yolo layer)
- filters=21 (For every convolutional layer beforo Yolo layer)

Batch and subdivision were set to 1 for testing and 64 and 32 for training.

### Training

Using the docker image the command is as follows:

```
docker run --runtime=nvidia --rm -v ${PWD}:/workspace -w /workspace daisukekobayashi/darknet:gpu \
        darknet detector train setup.data mask_yolov3_train.cfg ./darknet53.conv.74 -dont_show -map 2> train_log.txt
```

## YOLO v4

### Download weights

```
wget "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
```

### Train and test condigurations

The files `mask_yolov4_test.cfg` and `mask_yolov4_train.cfg` were modified as follows:

- width=416
- height=416
- max_batches = 6000
- steps=4800,5400
- classes=2 (For every Yolo layer)
- filters=21 (For every convolutional layer beforo Yolo layer)

Batch and subdivision were set to 1 for testing and 64 and 32 for training.

### Training

Using the docker image the command is as follows:

```
docker run --runtime=nvidia --rm -v ${PWD}:/workspace -w /workspace daisukekobayashi/darknet:gpu \
        darknet detector train setup.data mask_yolov4_train.cfg ./yolov4.conv.137 -dont_show -map 2> train_log.txt
```

## Running inference on images

```
docker run --runtime=nvidia --rm -v ${PWD}:/workspace -w /workspace daisukekobayashi/darknet:gpu \
	darknet detector test setup.data mask_yolov4_test.cfg backup/mask_yolov4_train_best.weights test-image1.jpg -thresh .6

```

## Running inference on videos

```
docker run --runtime=nvidia --rm -v ${PWD}:/workspace -w /workspace daisukekobayashi/darknet:gpu-cv \
	darknet detector demo setup.data mask_yolov4_test.cfg backup/mask_yolov4_train_best.weights test-video2.mp4 -thresh .6 -out_filename test-video2.mp4 -dont_show

```

## Results

### Images

|Detector| Image 1     | Image 2     | Image 3     | Image 4     |
|--------|-------------|-------------|-------------|-------------|
|Yolo v3 | ![][image1] | ![][image2] | ![][image3] | ![][image4] |
|Yolo v4 | ![][image5] | ![][image6] | ![][image7] | ![][image8] |

### Videos

| Detector | Video 1                      | Video 2                      |
|----------|------------------------------|------------------------------|
| Yolo v3  | https://youtu.be/OSJrziOfoU4 | https://youtu.be/1AW8TOsduMU |
| Yolo v4  | https://youtu.be/f3YmXntjT6s | https://youtu.be/Eay-L5saHP8 |