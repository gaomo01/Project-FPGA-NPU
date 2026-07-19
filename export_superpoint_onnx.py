import argparse
from pathlib import Path

import torch
import torch.nn as nn


class SuperPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        semi = self.convPb(self.relu(self.convPa(x)))
        semi = self.softmax(semi)
        desc = self.convDb(self.relu(self.convDa(x)))
        return semi, desc


def load_weights(model, weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    state_dict = {}
    for key, value in checkpoint.items():
        clean_key = key
        for prefix in ("module.", "superpoint.", "net."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        state_dict[clean_key] = value

    model.load_state_dict(state_dict, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Export SuperPoint encoder and heads to ONNX.")
    parser.add_argument("--weights", required=True, help="Path to .pt/.pth SuperPoint weights.")
    parser.add_argument("--output", default="superpoint_test.onnx", help="Output ONNX path.")
    parser.add_argument("--height", type=int, required=True, help="Input image height.")
    parser.add_argument("--width", type=int, required=True, help="Input image width.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    output_path = Path(args.output)

    model = SuperPointNet().eval()
    load_weights(model, weights_path)

    dummy = torch.randn(1, 1, args.height, args.width, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path.as_posix(),
        input_names=["image"],
        output_names=["semi", "desc"],
        opset_version=12,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported: {output_path.resolve()}")


if __name__ == "__main__":
    main()
