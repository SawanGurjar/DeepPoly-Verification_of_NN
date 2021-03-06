### Installations required

Run the following commands(if needed) before generating models using tensorflow.<br>
```
	$ pip install tensorflow
	$ pip install onnxruntime
	$ pip install -U tf2onnx
	$ git clone https://github.com/onnx/tensorflow-onnx
	$ cd tensorflow-onnx
	$ python setup.py install
```

Run the following commands in this ```(./models)``` directory, it should create ```<net_name>``` directory in this ```(./models)``` directory.<br>
``` 
	$ python3 networks/<net_name>.py 
```

And, the following command should create a ```<net_name>.onnx``` model in the directory ```ERAN/nets/<dataset-name>```
```
	$ python3 -m tf2onnx.convert --saved-model ./<net_name> --output ../nets/<dataset-name>/<netname>.onnx
```
For example: <br>
```
	python3 -m tf2onnx.convert --saved-model ./net3 --output ../nets/mnist/net3.onnx 
```
