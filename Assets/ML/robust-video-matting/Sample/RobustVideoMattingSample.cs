/* 
*   Robust Video Matting
*   Copyright (c) 2021 Yusuf Olokoba.
*/

namespace NatSuite.Examples {

    using System.Threading.Tasks;
    using UnityEngine;
    using UnityEngine.UI;
    using NatSuite.ML;
    using NatSuite.ML.Vision;

    public sealed class RobustVideoMattingSample : MonoBehaviour {

        [Header(@"NatML Hub")]
        public string accessKey;

        [Header(@"UI")]
        public RawImage rawImage;
        public AspectRatioFitter aspectFitter;

        WebCamTexture webCamTexture;
        RenderTexture segmentationImage;

        MLModelData modelData;
        MLModel model;
        RobustVideoMattingPredictor predictor;

        async void Start () {
            // Fetch model data from NatML Hub
            Debug.Log("Fetching model data from NatML Hub");
            modelData = await MLModelData.FromHub("@natsuite/robust-video-matting", accessKey);
            // Deserialize the model
            model = modelData.Deserialize();
            // Create the predictor
            predictor = new RobustVideoMattingPredictor(model);
            // Start the webcam
            webCamTexture = new WebCamTexture();
            webCamTexture.Play();
            // Create and display the destination segmentation image
            while (webCamTexture.width == 16 || webCamTexture.height == 16)
                await Task.Yield();
            segmentationImage = new RenderTexture(webCamTexture.width, webCamTexture.height, 0);
            rawImage.texture = segmentationImage;
            aspectFitter.aspectRatio = (float)webCamTexture.width / webCamTexture.height;
        }

        void Update () {
            // Check that the segmentation image has been created
            if (!segmentationImage)
                return;
            // Check that the camera frame updated
            if (!webCamTexture.didUpdateThisFrame)
                return;
            // Predict
            var matte = predictor.Predict(webCamTexture);
            matte.Render(segmentationImage);
        }

        void OnDisable () {
            // Dispose the predictor
            predictor?.Dispose();
            // Dispose the model
            model?.Dispose();
        }
    }
}