import torch
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path="water_model.pth"):

    # checkpoint = torch.load(model_path, map_location=DEVICE)
    checkpoint = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=False  # allow full checkpoint loading
)

    model = smp.DeepLabV3(
        encoder_name="efficientnet-b7",
        encoder_weights=None,
        in_channels=15,
        classes=1
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, checkpoint["band_mins"], checkpoint["band_maxs"]