import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

JPEG_CONTENT_TYPE = 'image/jpeg'


def model_fn(model_dir):
    # Load the saved model state_dict
    pth_path = os.path.join(model_dir, "model.pth")
    model = torch.load(pth_path)
    model.eval()
    return model

# deserializing input
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))
    
    raise Exception('Received unsupported ContentType: {}'.format(content_type))

# normalizing input data and performing predictions
def predict_fn(input_object, model):    
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction