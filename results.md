# Experimental Results

### medmnist
| method | avg-per-acc| global rounds|local epochs|lr|optim|alpha|
| -------- | -------- |-------| -------- |-------------|----------|------|
|base w/o agg|0.7676|300|1|0.001|adam|0.01|
|fedavg|0.7807|300|1|0.001|adam|0.01|
|fedprox| 0.7504|300|1|0.001|sgd|0.01|
|fedbn|0.8173|300|1|0.01|sgd|0.01|
|fedbn|0.8309|300|1|0.001|adam|0.01|
|fedap|0.8237|300|1|0.01|sgd|0.01|
|MetaFed|0.8180|300|1|0.001|sgd|0.01|
|MetaFed|0.8357|300|1|0.01|sgd|0.01|
|**fedlp**|<strong>0.8389</strong>|300|1|0.001|adam|0.01|

