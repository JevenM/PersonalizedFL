1. no aggregrate
python main.py --alg base --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.1 (sgd)
# Personalized test acc for each client: 0.7173,0.8097,0.7547,0.5627,0.8032,0.6756,0.8404,0.6542,0.7760,0.6693,0.6658,0.8347,0.8613,0.6818,0.8533,0.6944,0.6461,0.6979,0.8753,0.7239,
# Average accuracy: 0.7399

python main.py --alg base --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7101,0.8827,0.7760,0.8218,0.8404,0.7287,0.7600,0.8267,0.6533,0.8213,0.7147,0.8351,0.6427,0.8262,0.6987,0.7473,0.8378,0.7086,0.6818,0.8378,
# Average accuracy: 0.7676, best_epoch: 70

python main.py --alg base --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.01 (sgd)

2. FedAvg
python main.py --alg fedavg --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 (sgd)
# Personalized test acc for each client: 0.7287,0.7440,0.7040,0.7660,0.7660,0.6862,0.7040,0.7840,0.7280,0.7653,0.6827,0.7926,0.6693,0.8155,0.7573,0.7713,0.8138,0.6791,0.7166,0.8112,
# Average accuracy: 0.7443

python main.py --alg fedavg --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7367,0.7920,0.7440,0.8138,0.8245,0.7713,0.7360,0.8107,0.7200,0.8080,0.7573,0.8085,0.7147,0.8610,0.7600,0.8005,0.8431,0.7059,0.7701,0.8351,
# Average accuracy: 0.7807, best_epoch: 155

python main.py --alg fedavg --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 1 (sgd)
# Personalized test acc for each client: 0.7867,0.7694,0.7346,0.7947,0.7487,0.7968,0.7754,0.8011,0.7147,0.7787,0.7861,0.7513,0.7089,0.8075,0.7721,0.7914,0.7968,0.7534,0.8053,0.8005,
# Average accuracy: 0.7737

python main.py --alg fedavg --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1 (sgd)
# Personalized test acc for each client: 0.7040,0.7936,0.8080,0.7040,0.7926,0.7265,0.8138,0.6676,0.7707,0.7067,0.6979,0.8107,0.8133,0.7139,0.8027,0.7399,0.7534,0.7861,0.8435,0.7131,
# Average accuracy: 0.7581

python main.py --alg fedavg --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 (sgd)
# Personalized test acc for each client: 0.7048,0.7147,0.6133,0.7074,0.7261,0.6596,0.6613,0.7387,0.6453,0.7253,0.6133,0.7314,0.6133,0.7594,0.6907,0.6941,0.7394,0.6230,0.6524,0.7500,
# Average accuracy: 0.6882


3. fedprox
python main.py --alg fedprox --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1 --mu 0.1 (sgd)
# Personalized test acc for each client: 0.7040,0.7936,0.8133,0.6987,0.7899,0.7158,0.8005,0.6756,0.7707,0.7120,0.6952,0.8133,0.8053,0.7139,0.8107,0.7373,0.7480,0.7807,0.8355,0.7131,
# Average accuracy: 0.7564

python main.py --alg fedprox --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --mu 0.01 (sgd)
# Personalized test acc for each client: 0.7340,0.7760,0.6933,0.7819,0.7713,0.6968,0.6827,0.8000,0.7227,0.7947,0.6987,0.7819,0.6720,0.8235,0.7573,0.7766,0.8351,0.6791,0.7246,0.8059,
# Average accuracy: 0.7504, best_epoch: 267

4. fedlp (latent prompt)
# pretrained lr = fixed 0.01 修改计算流程  adam 150预训练 之后交替训练 schedule

# first: 预训练 固定backbone不变，只交替训练prompt和classifier 无schedule 在验证和测试数据集上评估加上prompt 
# python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 8 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7846,0.8933,0.8427,0.8351,0.8351,0.7846,0.8213,0.8373,0.7947,0.8267,0.7707,0.8777,0.7360,0.8877,0.7280,0.7713,0.8457,0.7567,0.7513,0.8963,
# Average accuracy: 0.8138, best_epoch: 57

# python main.py --alg fedlp --dataset medmnist --iters 200 --wk_iters 3 --non_iid_alpha 0.01 --prompt_dim 16 --lr 0.0001 (adam)
# Personalized test acc for each client: 0.7500,0.9227,0.8080,0.8165,0.8644,0.7952,0.8427,0.8667,0.7787,0.8160,0.7840,0.8777,0.7680,0.8636,0.7200,0.7819,0.8590,0.7246,0.7620,0.8750,
# Average accuracy: 0.8138, best_epoch: 186 等同于lr=0.001时候进行100个round

# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 16 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8005,0.9147,0.8107,0.8404,0.8617,0.7926,0.8400,0.8480,0.7893,0.8373,0.7867,0.9043,0.7547,0.8690,0.7360,0.8085,0.8617,0.7299,0.7406,0.8750,
# Average accuracy: 0.8201, best_epoch: 119

# python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.01 --prompt_dim 16 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7766,0.8960,0.8160,0.8378,0.8617,0.8085,0.8453,0.8267,0.7653,0.8400,0.7573,0.8856,0.3040,0.8957,0.7413,0.7899,0.8537,0.7460,0.7567,0.8856,
# Average accuracy: 0.7945, best_epoch: 37

# 1由于修改模型结构，所以重新预训练，增加hyp缩放因子作用于prompt融合时, 性能少许降低
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 16 --lr 0.001 (adam)
#Personalized test acc for each client: 0.8085,0.8907,0.8373,0.8697,0.8511,0.7979,0.8187,0.8560,0.7787,0.8267,0.7413,0.8910,0.7680,0.8529,0.7227,0.7899,0.8324,0.7273,0.7781,0.8883,
#Average accuracy: 0.8164, best_epoch: 94

# 2增加weight_decay 1e-4
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 16 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8032,0.9200,0.8027,0.8936,0.8218,0.7819,0.8267,0.8347,0.7840,0.8107,0.7867,0.8644,0.7200,0.8930,0.7547,0.7952,0.8218,0.7620,0.7540,0.8723,
# Average accuracy: 0.8152, best_epoch: 93

# 3增加weight_list，在聚合时候根据val_acc_list的值设置反比例权重
#python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 16 --lr 0.001 (adam) 
#Personalized test acc for each client: 0.8085,0.9040,0.7867,0.8750,0.8564,0.7766,0.8133,0.8400,0.7333,0.8320,0.7733,0.8910,0.7413,0.8636,0.7493,0.7739,0.8245,0.7460,0.7754,0.8883,
#Average accuracy: 0.8126, best_epoch: 76

# 4增加weight_list, 去掉weight_decay，在聚合时候根据val_acc_list的值设置反比例权重
#python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 16 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7686,0.9093,0.8267,0.8511,0.8511,0.7553,0.8213,0.8480,0.7707,0.8187,0.7680,0.8856,0.7627,0.8850,0.7333,0.7580,0.8457,0.7406,0.7754,0.8777,
# Average accuracy: 0.8126, best_epoch: 128


# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 16 --lr 0.002 (adam)
# Personalized test acc for each client: 0.7846,0.4933,0.8027,0.8324,0.8484,0.7633,0.8213,0.8347,0.7520,0.8213,0.7467,0.8883,0.7467,0.8636,0.7093,0.7846,0.8431,0.7299,0.7299,0.8777,
# Average accuracy: 0.7837, best_epoch: 124



# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8005,0.9040,0.8160,0.8404,0.8324,0.7633,0.8400,0.8293,0.7920,0.8213,0.7627,0.9043,0.7520,0.8690,0.7627,0.7872,0.8457,0.7326,0.7594,0.9149,
# Average accuracy: 0.8165, best_epoch: 174

# wd=0, 但在聚合时候根据val_acc_list的值设置反比例权重，30个round开始，证明权重会导致性能下降
#python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7926,0.9040,0.8160,0.8298,0.8590,0.7473,0.8640,0.8427,0.7947,0.7947,0.7680,0.9096,0.7520,0.8529,0.7147,0.7739,0.8644,0.7781,0.7380,0.9016,
# Average accuracy: 0.8149, best_epoch: 129

# wd=0, 不用自己设置权重，权重为1/num_client
#python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8112,0.9040,0.7867,0.8351,0.8617,0.8005,0.8293,0.8720,0.8000,0.8320,0.7840,0.8830,0.7653,0.8342,0.7573,0.7793,0.8484,0.7647,0.7674,0.8989,
# Average accuracy: 0.8208, best_epoch: 140


# wd=1e-4, 不用自己设置权重，权重为默认的1/num_client，证明增加wd会导致性能下降
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7952,0.9200,0.7813,0.8351,0.8484,0.7766,0.8267,0.8320,0.7307,0.8000,0.7467,0.8883,0.7493,0.8342,0.6373,0.7633,0.8457,0.7460,0.7219,0.8750,
# Average accuracy: 0.7977, best_epoch: 278

# wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，100个round开始
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
#Personalized test acc for each client: 0.7846,0.8960,0.8160,0.8431,0.8670,0.7979,0.8507,0.8587,0.7893,0.8400,0.7840,0.9176,0.7867,0.8422,0.7440,0.7952,0.8617,0.7674,0.7701,0.8883,
#Average accuracy: 0.8250, best_epoch: 123

# wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，80个round开始
python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8138,0.9173,0.8027,0.8245,0.8564,0.8165,0.8427,0.8773,0.7973,0.8267,0.7707,0.8989,0.7973,0.8529,0.7573,0.8005,0.8484,0.7647,0.7701,0.9069,
# Average accuracy: 0.8271, best_epoch: 138


# wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，0.3*40=12个round开始，models结构修改，增加hyp的[0,1]约束和非线性, relu5由ReLu改为Tanh，学习率衰减0.9(每9个round)
# python main.py --alg fedlp --dataset medmnist --iters 40 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7926,0.9067,0.7813,0.8085,0.8404,0.7926,0.8453,0.8240,0.7920,0.8267,0.7520,0.9202,0.7760,0.8690,0.7627,0.7952,0.8537,0.7353,0.7540,0.8697,
# Average accuracy: 0.8149, best_epoch: 8

# wd=0, 平均聚合，models结构修改，增加hyp的[0,1]约束和非线性, relu5由ReLu改为Tanh，学习率衰减0.9(每9个round)
# python main.py --alg fedlp --dataset medmnist --iters 40 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8059,0.9093,0.8080,0.8085,0.8537,0.8005,0.8400,0.8507,0.7440,0.8533,0.7680,0.9016,0.7947,0.8583,0.7520,0.8005,0.8537,0.7594,0.7032,0.8989,
# Average accuracy: 0.8182, best_epoch: 19

# 不用预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，0.3*300=90个round开始，models结构修改，增加hyp的[0,1]约束和非线性, relu5的由Tanh改为ReLu，学习率衰减0.99(每30个round)
# 增加指数分母tau=3
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.6862,0.8320,0.7707,0.7872,0.7686,0.6410,0.7173,0.7707,0.6880,0.7493,0.6880,0.7846,0.6080,0.7701,0.7093,0.6596,0.7899,0.6711,0.6711,0.8005,
# Average accuracy: 0.7282, best_epoch: 56

# 预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，0.3*300=90个round开始，models结构修改，增加hyp的[0,1]约束和非线性, relu5的由Tanh改为ReLu，无学习率衰减
# 增加指数分母tau=0.1
# python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 16 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7872,0.9093,0.8080,0.8351,0.8457,0.8165,0.8613,0.8400,0.7600,0.8240,0.7840,0.8963,0.7653,0.8610,0.7440,0.7979,0.8059,0.7433,0.7674,0.8856,
# Average accuracy: 0.8169, best_epoch: 19

# 预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，0.01*300=3个round开始，models结构修改，增加hyp的[0,1]约束和非线性, relu5的由Tanh改为ReLu，无学习率衰减
# 增加指数分母tau=0.1
# python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8059,0.9333,0.8000,0.8298,0.8457,0.7979,0.8480,0.8507,0.7733,0.8240,0.7680,0.9069,0.7653,0.9037,0.6880,0.7606,0.8537,0.7701,0.7433,0.8723,
# Average accuracy: 0.8170, best_epoch: 14


# 预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，去掉指数负号（和val_acc成正比），90个round开始，models结构修改，增加hyp的[0,1]约束, relu5的由Tanh改为ReLu，无学习率衰减
# 增加指数分母tau=0.1, 去掉非线性output = self.relu5(output)，收敛速度减慢明显
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8165,0.9147,0.8187,0.8351,0.8564,0.7979,0.8427,0.8507,0.7920,0.8347,0.7787,0.8697,0.7813,0.8717,0.7653,0.7686,0.8537,0.7754,0.7567,0.8936,
# Average accuracy: 0.8237, best_epoch: 107

# 预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，去掉指数负号（和val_acc成正比）,从train_acc_avg>=0.9的round开始指数聚合，一开始平均聚合，models结构修改，增加hyp的[0,1]约束, 
# relu5的由Tanh改为ReLu，无学习率衰减，增加指数分母tau=0.1，去掉非线性，收敛速度减慢明显
python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7872,0.9013,0.8320,0.8351,0.8564,0.7952,0.8480,0.8507,0.7840,0.8400,0.7893,0.8963,0.7920,0.8717,0.7627,0.7926,0.8271,0.7701,0.7540,0.8963,
# Average accuracy: 0.8241, best_epoch: 102
# tau=1不如0.1: Personalized test acc for each client: 0.8005,0.9040,0.8240,0.8324,0.8723,0.7660,0.8400,0.8453,0.7840,0.8320,0.7893,0.9043,0.7707,0.8824,0.7627,0.7819,0.8670,0.7727,0.7326,0.9149,
# Average accuracy: 0.8240, best_epoch: 126

# 去除prompt和output的加权融合（变差）, 预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，去掉指数负号（和val_acc成正比）,从train_acc_avg>=0.9的round开始指数聚合，
# 一开始平均聚合，models结构修改，增加hyp的[0,1]约束, relu5的由Tanh改为ReLu，无学习率衰减，增加指数分母tau=0.1，去掉第一个relu5非线性，收敛速度减慢明显
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7793,0.9120,0.8400,0.8404,0.8457,0.8085,0.8453,0.8213,0.7733,0.8080,0.7600,0.9016,0.7547,0.8663,0.7147,0.7686,0.8457,0.7620,0.7380,0.8830,       
# Average accuracy: 0.8134, best_epoch: 194

# 最后的y和out_repeated的cat转为+, 后面必须加非线性函数，交替每个本地iter训练3次，模型性能增长速度慢，性能不好

# 去除prompt和output的加权融合（变差）, 预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，去掉指数负号（和val_acc成正比）,从train_acc_avg>=0.9的round开始指数聚合，
# 一开始平均聚合，models结构修改，增加hyp的[0,1]约束, relu5的由Tanh改为ReLu，无学习率衰减，增加指数分母tau=0.1，去掉第一个relu5非线性，收敛速度减慢明显
# 最后的y和out_repeated的cat转为+, 后面必须加非线性函数
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 3 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.6702,0.8587,0.7733,0.7021,0.8138,0.6489,0.7227,0.7333,0.6933,0.6880,0.6773,0.7633,0.6373,0.8075,0.6293,0.6223,0.7926,0.6738,0.2166,0.7128,
# Average accuracy: 0.6919, best_epoch: 165

# 修改models中流程，使用cat连接prompt和output，预训练降低至100round，--prompt_dim=8 一直平均聚合 不交替，训练fc+prompt lr=0.0001
# Personalized test acc for each client: 0.8138,0.9307,0.8027,0.8378,0.8564,0.7660,0.8560,0.8187,0.7573,0.8373,0.7520,0.9309,0.7493,0.8877,0.7440,0.7819,0.8404,0.7513,0.7513,0.8936,
# Average accuracy: 0.8180, best_epoch: 234

# 预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，加上指数负号（默认和val_acc成反比）,从train_acc_avg>=0.9的round开始指数聚合，一开始平均聚合，models结构修改，增加hyp的[0,1]约束, 
# relu5的由Tanh改为ReLu，无学习率衰减，增加指数分母tau=0.1，去掉非线性，收敛速度减慢明显
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8032,0.9147,0.8053,0.8271,0.8590,0.8032,0.8667,0.8613,0.7813,0.8347,0.7813,0.8830,0.7787,0.8636,0.7547,0.7793,0.8431,0.7620,0.7647,0.9016,
# Average accuracy: 0.8234, best_epoch: 107 可以看出正负号没什么区别

# 预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，去掉指数负号（和val_acc成正比）,从train_acc_avg>=0.9的round开始指数聚合，一开始平均聚合，models结构修改，增加hyp的[0,1]约束, 
# relu5的由Tanh改为ReLu，无学习率衰减，增加指数分母tau=0.01，去除所有RELU5非线性（包括维度拼接之后y=self.relu5(y)，性能大幅下降）
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# tau=0.01不如0.1


# 预训练，wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，去掉指数负号（和val_acc成正比），models结构修改，增加hyp的[0,1]约束, 去掉非线性，收敛速度减慢明显
# relu5的由Tanh改为ReLu，无学习率衰减，增加指数分母tau=0.1，翻转，从train_acc_avg>=0.9的round开始平均聚合，一开始是指数聚合
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8138,0.9147,0.7787,0.8404,0.8644,0.7846,0.8507,0.8613,0.7760,0.8293,0.7520,0.8910,0.7493,0.8690,0.7280,0.7766,0.8511,0.7727,0.7647,0.9096,
# Average accuracy: 0.8189, best_epoch: 127


# wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，60个round开始
# python main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7872,0.9173,0.8053,0.8324,0.8537,0.7926,0.8293,0.8613,0.7813,0.8347,0.7707,0.8537,0.7573,0.8797,0.7493,0.7606,0.8537,0.7861,0.7674,0.9149,
# Average accuracy: 0.8194, best_epoch: 124

# wd=0, 但在聚合时候根据val_acc_list的值设置指数变换权重，50个round开始
# python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7952,0.9253,0.7920,0.8404,0.8697,0.8191,0.8427,0.8507,0.7573,0.8267,0.7733,0.8856,0.7573,0.8610,0.7200,0.8005,0.8324,0.8048,0.7166,0.9122,
# Average accuracy: 0.8192, best_epoch: 63

# python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.005 (adam)
# Personalized test acc for each client: 0.2819,0.4933,0.2880,0.4894,0.4761,0.2793,0.2960,0.5013,0.2933,0.4480,0.2213,0.4681,0.3040,0.2968,0.2427,0.4920,0.4867,0.2968,0.2166,0.5027,
# Average accuracy: 0.3687, best_epoch: 20




# second: 预训练，无schedule：有无最终一样 (模型在验证和测试数据集上评估加上prompt prompt和backbone+cls交替训练)
# python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 8 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7367,0.8907,0.7760,0.8617,0.8457,0.7926,0.7947,0.8267,0.7680,0.8240,0.7573,0.8750,0.7440,0.8262,0.7253,0.7713,0.8590,0.7406,0.7406,0.8777,
# Average accuracy: 0.8017, best_epoch: 56

# third: 预训练，固定classifier不变, 只交替训练prompt和backbone
# python main.py --alg fedlp --dataset medmnist --iters 100 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 8 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8138,0.9200,0.7467,0.8298,0.8404,0.8191,0.8053,0.8213,0.0907,0.0000,0.7627,0.8484,0.7013,0.8262,0.7333,0.7713,0.8564,0.7086,0.0936,0.8697,
# Average accuracy: 0.6929, best_epoch: 65

# wd=0, 默认平均聚合
pymao main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7766,0.9120,0.8240,0.8218,0.8750,0.8059,0.8347,0.8507,0.7867,0.8347,0.7893,0.8883,0.7653,0.8422,0.7173,0.7819,0.8271,0.7807,0.7647,0.8910,
# Average accuracy: 0.8185, best_epoch: 132

# wd=0, SVD之后平均聚合
pymao main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.7899,0.9120,0.8293,0.8165,0.8590,0.7926,0.8320,0.8400,0.7813,0.8267,0.7653,0.8856,0.7760,0.8877,0.7493,0.7952,0.8511,0.7406,0.7139,0.8883,
# Average accuracy: 0.8166, best_epoch: 96

# wd=0, 在聚合时候根据val_acc_list的值设置指数变换权重，80个round开始, 指数的tau=0.1
pymao main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8218,0.9013,0.8133,0.8590,0.8590,0.8085,0.8560,0.8507,0.7653,0.8267,0.7733,0.8830,0.7680,0.8583,0.7653,0.7952,0.8590,0.7594,0.7513,0.9016,
# Average accuracy: 0.8238, best_epoch: 80


# wd=0, 在聚合时候根据val_acc_list的值设置指数变换权重，80个round开始, 指数的tau=0.01
pymao main.py --alg fedlp --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --prompt_dim 32 --lr 0.001 (adam)



5. fedbn
# python main.py --alg fedbn --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --lr 0.001 (adam)
# Personalized test acc for each client: 0.8457,0.9067,0.8320,0.8830,0.8564,0.8324,0.8507,0.8533,0.7680,0.8533,0.7920,0.8431,0.7813,0.8770,0.7787,0.7899,0.8723,0.7807,0.7701,0.8511,
# Average accuracy: 0.8309, best_epoch: 260

python main.py --alg fedbn --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 (sgd)
# Personalized test acc for each client: 0.7793,0.8800,0.8267,0.8670,0.8457,0.7846,0.8587,0.8373,0.7733,0.8160,0.8000,0.8883,0.7813,0.8797,0.7867,0.7500,0.8245,0.7834,0.7219,0.8617,
# Average accuracy: 0.8173, best_epoch: 279

6. fedap
python main.py --alg fedap --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --model_momentum 0.3 (sgd)
# Personalized test acc for each client: 0.8245,0.8933,0.8160,0.8644,0.8245,0.7314,0.8587,0.8533,0.7973,0.8400,0.7867,0.8590,0.7947,0.8610,0.7733,0.7926,0.8564,0.7914,0.7807,0.8750,
# Average accuracy: 0.8237, best_epoch: 269


7. metafed

# python main.py --alg metafed --dataset medmnist --iters 300 --wk_iters 1 --threshold 1.1 --nosharebn --non_iid_alpha 0.01 (sgd)
# Personalized test acc for each client: 0.8112,0.9307,0.8293,0.8697,0.8644,0.7819,0.8640,0.8373,0.8027,0.8693,0.8107,0.8670,0.8187,0.8877,0.7707,0.8245,0.8723,0.7513,0.7834,0.8670,
# Average accuracy: 0.8357, best_epoch: 24

# python main.py --alg metafed --dataset medmnist --iters 300 --wk_iters 1 --threshold 1.1 --non_iid_alpha 0.01 --lr 0.001 (sgd)
# # Personalized test acc for each client: 0.7686,0.8373,0.8267,0.7952,0.8085,0.7287,0.8427,0.7760,0.7387,0.8347,0.8187,0.8351,0.8027,0.7807,0.7867,0.8005,0.8351,0.7406,0.7781,0.8324,
# # Average accuracy: 0.7984, best_epoch: 169

python main.py --alg metafed --dataset medmnist --iters 300 --wk_iters 1 --threshold 1.1 --nosharebn --non_iid_alpha 0.01 --lr 0.001 (sgd)
# Personalized test acc for each client: 0.7846,0.9120,0.8027,0.8617,0.8245,0.7819,0.8587,0.8560,0.7813,0.8533,0.7973,0.8457,0.7813,0.8529,0.7680,0.7793,0.8644,0.7540,0.7380,0.8617,
# Average accuracy: 0.8180, best_epoch: 171