import torch
import torch.onnx
from net import UNet  # 导入你的UNet模型定义

if __name__ == '__main__':
    # 加载训练好的模型
    model = UNet()  # 根据你的模型参数调整
    model.load_state_dict(torch.load('params/unet.pth'))
    model.eval()

    # 创建一个虚拟输入（尺寸需与模型期望输入一致）
    dummy_input = torch.randn(1, 3, 256, 256)  # BCHW格式

    # 导出为ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "unet_model.onnx",
        export_params=True,
        opset_version=19,  # 根据模型使用的算子选择合适的版本
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 可选：支持动态batch_size
    )