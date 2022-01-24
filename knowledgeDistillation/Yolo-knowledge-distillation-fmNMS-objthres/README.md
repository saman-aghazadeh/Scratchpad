# Yolo-knowledge-distillation

This repo implements [Object detection at 200 Frames Per Second](https://arxiv.org/pdf/1805.06361v1.pdf) and distills knowledge from larger teacher network to a much smaller student network.

## Quick start

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/GuiseAI/Yolo-knowledge-distillation/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/GuiseAI/Yolo-knowledge-distillation
$ cd Yolo-knowledge-distillation
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Training</summary>

1. Train teacher model on the training dataset using [yolov5](https://github.com/ultralytics/yolov5).

2. Create student model configuration for knowledge distillation as models/student.yaml

3. Run below command to distill teacher student knowledge on your custom dataset.
```bash
$ python train.py --data custom.yaml --cfg models/student.yaml --batch-size 64 --teacher-weights weights/yolov5l.pt
                                                                            40                   weights/yolov5x.pt
                                                                            24
                                                                            16
```
* student scratch training plots
  
![results_yolov5s](https://user-images.githubusercontent.com/65303956/134473649-21a465f5-aba0-4b39-887a-cfbbe7f974be.png)


* teacher training plots here.
  
![results](https://user-images.githubusercontent.com/65303956/134458999-747423b9-5947-4860-b81a-ca8fadbe0ce2.png)


* add student knowledge distillation plots here.
  
![results](https://user-images.githubusercontent.com/65303956/134459206-f417c0fc-70a9-4fd4-9d42-eb2cd91c6efd.png)

</details>  
