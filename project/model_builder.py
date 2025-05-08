# ------------------------------
# model_builder.py (loads and builds models from Keras applications)
# ------------------------------
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras import layers, models

# This function builds a model based on the specified architecture and returns it along with the appropriate preprocessing function
def get_model(model_name, input_shape=(64, 64, 3), num_classes=25, pretrained=False):
    weights = 'imagenet' if pretrained else None

    if model_name == 'resnet50':
        base_model = ResNet50(input_shape=input_shape, weights=weights, include_top=False)
        preprocess = resnet_preprocess
    elif model_name == 'resnet101':
        base_model = ResNet101(input_shape=input_shape, weights=weights, include_top=False)
        preprocess = resnet_preprocess
    elif model_name == 'resnet152':
        base_model = ResNet152(input_shape=input_shape, weights=weights, include_top=False)
        preprocess = resnet_preprocess
    elif model_name == 'densenet121':
        base_model = DenseNet121(input_shape=input_shape, weights=weights, include_top=False)
        preprocess = densenet_preprocess
    elif model_name == 'densenet169':
        base_model = DenseNet169(input_shape=input_shape, weights=weights, include_top=False)
        preprocess = densenet_preprocess
    elif model_name == 'densenet201':
        base_model = DenseNet201(input_shape=input_shape, weights=weights, include_top=False)
        preprocess = densenet_preprocess
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, preprocess