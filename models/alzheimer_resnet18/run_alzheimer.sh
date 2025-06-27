python3 alzheimer_resnet18.py \
  --train_path /home/infres/xnguyen-24/cluster_cam/datasets/alzheimer/Data/train.parquet \
  --test_path /home/infres/xnguyen-24/cluster_cam/datasets/alzheimer/Data/test.parquet \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --patience 7 \
  --checkpoint best_model.pth \
  --save_test_dir /home/infres/xnguyen-24/cluster_cam/datasets/alzheimer/test_imgs
