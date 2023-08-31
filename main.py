import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")

parser.add_argument("--height", default=64, type=int, help="Up-scaled image height")
parser.add_argument("--width", default=64, type=int, help="Up-scaled image width")
parser.add_argument("--channels", default=3, type=int, help="Up-scaled image channels")
parser.add_argument("--cnn_channels", default=32, type=int, help="CNN channels in the first stage.")
parser.add_argument("--downscale", default=4, type=int, help="Downscale factor")

parser.add_argument("--stages", default=4, type=int, help="UNet scaling stages")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage")

parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum.")