1. no aggregrate
python main.py --alg base --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.1
# Personalized test acc for each client: 0.7173,0.8097,0.7547,0.5627,0.8032,0.6756,0.8404,0.6542,0.7760,0.6693,0.6658,0.8347,0.8613,0.6818,0.8533,0.6944,0.6461,0.6979,0.8753,0.7239,
# Average accuracy: 0.7399

python main.py --alg base --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.001


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


3. fedprox
python main.py --alg fedprox --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1 --mu 0.1
#Personalized test acc for each client: 0.7040,0.7936,0.8133,0.6987,0.7899,0.7158,0.8005,0.6756,0.7707,0.7120,0.6952,0.8133,0.8053,0.7139,0.8107,0.7373,0.7480,0.7807,0.8355,0.7131,
#Average accuracy: 0.7564
python main.py --alg fedprox --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --mu 0.1
# Personalized test acc for each client: 0.7420,0.7547,0.7200,0.7367,0.7447,0.7367,0.7333,0.7733,0.7253,0.7867,0.7120,0.7899,0.6933,0.8155,0.7760,0.7899,0.7979,0.6979,0.7487,0.8191,
# Average accuracy: 0.7547

4. fedlp (latent prompt)
python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --pretrained_iters 150

# prompt_dim = 8:
python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.0001
# Personalized test acc for each client: 0.6915,0.8213,0.7813,0.7234,0.7872,0.7314,0.8320,0.7200,0.6667,0.8000,0.7147,0.7207,0.7307,0.8048,0.7467,0.7287,0.7926,0.6791,0.6979,0.7367,
# Average accuracy: 0.7454

# prompt_dim = 16:
python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.0001
# Personalized test acc for each client: 0.2234,0.6427,0.4240,0.4894,0.6649,0.3484,0.3947,0.4827,0.3520,0.5333,0.2907,0.6941,0.2960,0.5615,0.4267,0.4814,0.6702,0.3155,0.2433,0.7181,
# Average accuracy: 0.4626

# prompt_dim = 8: pretrained lr = fixed 0.01 修改计算流程
python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.0001
# Personalized test acc for each client: 0.6676,0.8053,0.6773,0.8032,0.8059,0.7447,0.6773,0.7947,0.6453,0.8453,0.7520,0.7766,0.6827,0.8289,0.6987,0.7979,0.8351,0.6257,0.7059,0.8032,
# Average accuracy: 0.7487

# prompt_dim = 16:
python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.0001 --prompt_dim 16
# Personalized test acc for each client: 0.7367,0.7947,0.7307,0.7899,0.8032,0.7739,0.7520,0.7307,0.6907,0.8160,0.7707,0.8059,0.7173,0.7781,0.7573,0.8085,0.8005,0.6952,0.7219,0.7926,
# Average accuracy: 0.7633

# prompt_dim = 32: (adam 150预训练 之后交替训练 schedule)
python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.0001 --prompt_dim 32
# Personalized test acc for each client: 0.8324,0.9573,0.8587,0.8777,0.8670,0.8351,0.8667,0.8827,0.7920,0.8880,0.8160,0.9255,0.8587,0.8930,0.7573,0.8644,0.8644,0.7995,0.8048,0.9069,
# Average accuracy: 0.8574, best_epoch: 85


# 32 取消预训练 adam 发现很早就达到最好，训练loss为0，acc为100，所以减小round为100，交替训练 ×
python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.0001 --prompt_dim 32
# Personalized test acc for each client: 0.7048,0.8747,0.7840,0.8245,0.8005,0.6915,0.7120,0.7813,0.7093,0.7653,0.7280,0.8032,0.6720,0.8155,0.6720,0.7048,0.8138,0.6979,0.7086,0.8245,
# Average accuracy: 0.7544


# 预训练，无schedule：有无最终一样 (模型在验证和测试数据集上评估加上prompt prompt和backbone+cls交替训练)
python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 8 --lr 0.001
# Personalized test acc for each client: 0.7367,0.8907,0.7760,0.8617,0.8457,0.7926,0.7947,0.8267,0.7680,0.8240,0.7573,0.8750,0.7440,0.8262,0.7253,0.7713,0.8590,0.7406,0.7406,0.8777,
# Average accuracy: 0.8017, best_epoch: 56

# 预训练 固定backbone不变，只交替训练prompt和classifier 无schedule 在验证和测试数据集上评估加上prompt
python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 8 --lr 0.001
# Personalized test acc for each client: 0.7846,0.8933,0.8427,0.8351,0.8351,0.7846,0.8213,0.8373,0.7947,0.8267,0.7707,0.8777,0.7360,0.8877,0.7280,0.7713,0.8457,0.7567,0.7513,0.8963,
# Average accuracy: 0.8138, best_epoch: 57

# 预训练，不交替，固定backbone，训练prompt+classifier
python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 8
# Personalized test acc for each client: 0.2819,0.4933,0.2880,0.4894,0.4761,0.2793,0.2960,0.5013,0.2933,0.4480,0.2213,0.4681,0.3040,0.2968,0.2427,0.4920,0.4867,0.2968,0.2166,0.5027,
# Average accuracy: 0.3687, best_epoch: 9

# 预训练，一直固定classifier不变, 只交替训练prompt和backbone
python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 8 --lr 0.001
# Personalized test acc for each client: 0.7899,0.9147,0.7920,0.8431,0.8564,0.0000,0.8080,0.8347,0.7360,0.0000,0.0960,0.8963,0.7307,0.8476,0.7120,0.7793,0.8378,0.7299,0.0936,0.8883,
# Average accuracy: 0.6593, best_epoch: 97
# Personalized test acc for each client: 0.7819,0.9013,0.8053,0.8564,0.8484,0.7819,0.8347,0.8213,0.0907,0.8213,0.0960,0.8910,0.1067,0.8021,0.7173,0.7553,0.8298,0.0963,0.0936,0.8963,
# Average accuracy: 0.6414, best_epoch: 55





python main.py --alg fedbn --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1
python main.py --alg fedap --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1 --model_momentum 0.3
python main.py --alg metafed --dataset medmnist --iters 300 --wk_iters 1 --threshold 1.1 --nosharebn --non_iid_alpha 0.1

python main.py --alg base --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.01

python main.py --alg fedprox --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --mu 0.01
python main.py --alg fedbn --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01
python main.py --alg fedap --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --model_momentum 0.3
python main.py --alg metafed --dataset medmnist --iters 50 --wk_iters 6 --threshold 1.1 --non_iid_alpha 0.01