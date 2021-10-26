# Robust Video Matting
[Robust Video Matting](https://peterl1n.github.io/RobustVideoMatting/) for human segmentation. This package requires [NatML](https://github.com/natsuite/NatML).

## Matting People in an Image
First, create the Robust Video Matting predictor:
```csharp
// Fetch the model data from Hub
var accessKey = "<HUB ACCESS KEY>"; // Get your access key from https://hub.natsuite.io/profile
var modelData = await MLModelData.FromHub("@natsuite/robust-video-matting", accessKey);
// Deserialize the model
var model = modelData.Deserialize();
// Create the Robust Video Matting predictor
var predictor = new RobustVideoMattingPredictor(model);
```

Predict the matte for an image:
```csharp
// Compute the matte
Texture2D image = ...; // This can also be a WebCamTexture or an MLImageFeature
RobustVideoMattingPredictor.Matte matte = predictor.Predict(image);
```

Finally, render the predicted matte to a `RenderTexture`:
```csharp
// Visualize the matte in a `RenderTexture`
var result = new RenderTexture(image.width, image.height, 0);
matte.Render(result);
```

___

## Requirements
- Unity 2019.2+
- [NatML 1.0.4+](https://github.com/natsuite/NatML)

## Quick Tips
- See the [NatML documentation](https://docs.natsuite.io/natml).
- Join the [NatSuite community on Discord](https://discord.gg/y5vwgXkz2f).
- Discuss [NatML on Unity Forums](https://forum.unity.com/threads/open-beta-natml-machine-learning-runtime.1109339/).
- Contact us at [hi@natsuite.io](mailto:hi@natsuite.io).

Thank you very much!