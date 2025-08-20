# Generate test metric
from tests.test_performance import run_test

checkpoint_path = r"/Users/permanj/Desktop/Cambridge Research/research_with_kyle/reproduce_PointProNets/checkpoints/denoising/TrivialEquiv-L2+ssim+grad-epoch=48-val_loss=0.0088.ckpt"
in_channels = 1
num_layers = 10
num_feat = 128
run_test(checkpoint_path, in_channels, num_layers, num_feat)