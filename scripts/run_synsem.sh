#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --ntasks=1
#SBATCH --mem=30000M
#SBATCH --output=synsem.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chakraba@mila.quebec
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
###########################
python -u main.py --cuda --epochs 40

cd analogy_tasks/

echo "****************WordSim Task *******************"
echo "Full Embeddings"
python -u main.py --sim-task --emb ../embeddings/final_emb_data_300_en.pb --vocab ../embeddings/common_vocab_data_300_en.pb
echo "Syntactic Embeddings"
python -u main.py --sim-task --emb ../embeddings/syn_emb_data_300_en.pb --vocab ../embeddings/common_vocab_data_300_en.pb
echo "Semantic Embeddings"
python -u main.py --sim-task --emb ../embeddings/sem_emb_data_300_en.pb --vocab ../embeddings/common_vocab_data_300_en.pb

echo "***************Analogy Task ********************"
echo "Full Embeddings"
python -u main.py --analogy-task --emb ../embeddings/final_emb_data_300_en.pb --vocab ../embeddings/common_vocab_data_300_en.pb
echo "Syntactic Embeddings"
python -u main.py --analogy-task --emb ../embeddings/syn_emb_data_300_en.pb --vocab ../embeddings/common_vocab_data_300_en.pb
echo "Semantic Embeddings"
python -u main.py --analogy-task --emb ../embeddings/sem_emb_data_300_en.pb --vocab ../embeddings/common_vocab_data_300_en.pb
