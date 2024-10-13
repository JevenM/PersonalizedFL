1. no aggregrate
python main.py --alg base --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.1
# Personalized test acc for each client: 0.7173,0.8097,0.7547,0.5627,0.8032,0.6756,0.8404,0.6542,0.7760,0.6693,0.6658,0.8347,0.8613,0.6818,0.8533,0.6944,0.6461,0.6979,0.8753,0.7239,
# Average accuracy: 0.7399

2. FedAvg
python main.py --alg fedavg --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01
# Personalized test acc for each client: 0.7287,0.7440,0.7040,0.7660,0.7660,0.6862,0.7040,0.7840,0.7280,0.7653,0.6827,0.7926,0.6693,0.8155,0.7573,0.7713,0.8138,0.6791,0.7166,0.8112,
# Average accuracy: 0.7443

python main.py --alg fedavg --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1
# Personalized test acc for each client: 0.7040,0.7936,0.8080,0.7040,0.7926,0.7265,0.8138,0.6676,0.7707,0.7067,0.6979,0.8107,0.8133,0.7139,0.8027,0.7399,0.7534,0.7861,0.8435,0.7131,
# Average accuracy: 0.7581

python main.py --alg fedavg --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 1
# Personalized test acc for each client: 0.7867,0.7694,0.7346,0.7947,0.7487,0.7968,0.7754,0.8011,0.7147,0.7787,0.7861,0.7513,0.7089,0.8075,0.7721,0.7914,0.7968,0.7534,0.8053,0.8005,
# Average accuracy: 0.7737

python main.py --alg fedavg --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01
# Personalized test acc for each client: 0.7048,0.7147,0.6133,0.7074,0.7261,0.6596,0.6613,0.7387,0.6453,0.7253,0.6133,0.7314,0.6133,0.7594,0.6907,0.6941,0.7394,0.6230,0.6524,0.7500,
# Average accuracy: 0.6882

python main.py --alg fedprox --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1 --mu 0.1
#Personalized test acc for each client: 0.7040,0.7936,0.8133,0.6987,0.7899,0.7158,0.8005,0.6756,0.7707,0.7120,0.6952,0.8133,0.8053,0.7139,0.8107,0.7373,0.7480,0.7807,0.8355,0.7131,
#Average accuracy: 0.7564

3. fedlp (latent prompt)
python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --pretrained_iters 150



python main.py --alg fedbn --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1
python main.py --alg fedap --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1 --model_momentum 0.3
python main.py --alg metafed --dataset medmnist --iters 300 --wk_iters 1 --threshold 1.1 --nosharebn --non_iid_alpha 0.1

python main.py --alg base --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.01

python main.py --alg fedprox --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --mu 0.01
python main.py --alg fedbn --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01
python main.py --alg fedap --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --model_momentum 0.3
python main.py --alg metafed --dataset medmnist --iters 50 --wk_iters 6 --threshold 1.1 --non_iid_alpha 0.01