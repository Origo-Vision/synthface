# synthface
Playground for synthetic face/people detection.

## Install requirements in a local environment
```shell
> python -m venv env
> source env/bin/activate
> python -m pip install requirements.txt
```

## Download 1000 item dataset
```shell
> curl https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2021/dataset_1000.zip -o dataset_1000.zip
```

## Export training and validation data from the dataset (90% training 10% validation)
```shell
> python export.py dataset_1000.zip --datadir-train training-data --datadir-valid validation-data
```

## Explore samples from the training or the validation data
```shell
> python explore.py 10 --datadir training-data/
```

## Play (inference) with samples from the training or validation data
```shell
> python play.py 10 --datadir validation-data/ --model save/save_0.42.pth 
```

## Live inference using web camera
```shell
> python live.py 1 --model save/save_0.42.pth 
```