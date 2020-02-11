# How to run

1. Prepare the dataset `./data`:

```
./data/train.json
./data/valid.json
./data/test.json
./data/crawl-300d-2M.vec
```

2. Preprocess the data
```
cd src/
python make_dataset.py ../data/
```

3. To Train, run
```
python train.py ../models/model_rnn/
```

4. To predict, run
```
python predict.py ../models/model_rnn/
```
where `--epoch` specifies the save model of which epoch to use.

5. Plot
```
python plot.py
```
