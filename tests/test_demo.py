# Generate test metric
from tests.test_performance import run_test

checkpoint_path = r"/Users/permanj/Desktop/Cambridge Research/research_with_kyle/reproduce_PointProNets/checkpoints/denoising/advanced-10-168/best-model-epoch=46-val_loss=0.0152.ckpt"
in_channels = 1
num_layers = 10
num_feat = 168
run_test(checkpoint_path, in_channels, num_layers, num_feat)