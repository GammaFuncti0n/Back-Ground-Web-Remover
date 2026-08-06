"""DeepLabV3 builders (MobileNetV2 only), adapted from VainF/DeepLabV3Plus-Pytorch."""

from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import mobilenetv2, resnet

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_mobilenet(name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(
        pretrained=pretrained_backbone, output_stride=output_stride
    )

    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320

    if name != "deeplabv3":
        raise NotImplementedError(f"Unsupported arch: {name}")

    return_layers = {"high_level_features": "out"}
    classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return DeepLabV3(backbone, classifier)


def deeplabv3_mobilenet(num_classes=21, output_stride=16, pretrained_backbone=True, **kwargs):
    """DeepLabV3 with MobileNetV2 backbone (Pascal VOC, 21 classes)."""
    return _segm_mobilenet(
        "deeplabv3",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )

def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.
    """
    return _segm_resnet(
        "deeplabv3plus",
        "resnet101",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )
