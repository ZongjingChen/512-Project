export CUDA_VISIBLE_DEVICES=0

python src/trainer.py --dataset yelp --n_clusters 10 --lr 5e-4 --cluster_weight 0.1 --seed 42 --epochs 1 --is_hierarchical --do_inference
# python src/trainer.py --dataset yelp --n_clusters 10 --lr 5e-4 --cluster_weight 0.1 --seed 42 --epochs 1 
# python src/trainer.py --dataset nyt --n_clusters 10 --lr 5e-4 --cluster_weight 0.1 --seed 42 --do_cluster --is_hierarchical
