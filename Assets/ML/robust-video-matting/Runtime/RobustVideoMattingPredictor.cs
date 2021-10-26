/* 
*   Robust Video Matting
*   Copyright (c) 2021 Yusuf Olokoba.
*/

namespace NatSuite.ML.Vision {

    using System;
    using NatSuite.ML.Extensions;
    using NatSuite.ML.Features;
    using NatSuite.ML.Internal;
    using NatSuite.ML.Types;

    /// <summary>
    /// Robust Video Matting for human segmentation.
    /// </summary>
    public sealed partial class RobustVideoMattingPredictor : IMLPredictor<RobustVideoMattingPredictor.Matte> {

        #region --Client API--
        /// <summary>
        /// Create the Robust Video Matting predictor.
        /// </summary>
        /// <param name="model">Robust Video Matting ML model.</param>
        /// <param name="downsampleRatio">Downsample ratio for inference.</param>
        public RobustVideoMattingPredictor (MLModel model, float downsampleRatio = 0.5f) {
            this.model = model;
            this.recurrentState = new IntPtr[4];
            this.downsampleRatio = new MLArrayFeature<float>(new [] { downsampleRatio });
            this.initialRecurrentType = new MLArrayType(typeof(float), new [] { 1, 1, 1, 1 });
        }

        /// <summary>
        /// Compute a human alpha matte on an image.
        /// </summary>
        /// <param name="inputs">Input image.</param>
        /// <returns>Alpha matte.</returns>
        public unsafe Matte Predict (params MLFeature[] inputs) {
            // Check
            if (inputs.Length != 1)
                throw new ArgumentException(@"Robust Video Matting predictor expects a single feature", nameof(inputs));
            // Check type
            var input = inputs[0];
            if (!input.GetImageSize(out var imageWidth, out var imageHeight))
                throw new ArgumentException(@"Robust Video Matting predictor expects an an array or image feature", nameof(inputs));
            // Create input features
            var imageType = model.inputs[0];
            var ratioType = model.inputs[5];
            var imageFeature = (input as IMLFeature).Create(imageType);
            var ratioFeature = (downsampleRatio as IMLFeature).Create(ratioType);
            var initialRecurrentState = 0f;
            for (var i = 0; i < recurrentState.Length; ++i)
                if (recurrentState[i] == IntPtr.Zero)
                    recurrentState[i] = (new MLArrayFeature<float>(&initialRecurrentState) as IMLFeature).Create(initialRecurrentType);
            // Predict
            var outputFeatures = model.Predict(
                imageFeature,
                recurrentState[0],
                recurrentState[1],
                recurrentState[2],
                recurrentState[3],
                ratioFeature
            );
            imageFeature.ReleaseFeature();
            ratioFeature.ReleaseFeature();
            // Update recurrent state
            for (var i = 0; i < recurrentState.Length; ++i) {
                recurrentState[i].ReleaseFeature();
                recurrentState[i] = outputFeatures[i + 2];
            }
            // Marshal
            var matte = new MLArrayFeature<float>(outputFeatures[1]);   // (N,1,H,W)
            var matteData = new float[matte.elementCount];
            matte.CopyTo(matteData);
            var result = new Matte(matte.shape[3], matte.shape[2], matteData);
            // Release
            outputFeatures[0].ReleaseFeature();
            outputFeatures[1].ReleaseFeature();
            // Return
            return result;
        }

        /// <summary>
        /// Dispose the predictor and release resources.
        /// </summary>
        public void Dispose () {
            foreach (var rs in recurrentState)
                rs.ReleaseFeature();
        }
        #endregion


        #region --Operations--
        private readonly IMLModel model;
        private IntPtr[] recurrentState;
        private readonly MLArrayFeature<float> downsampleRatio;
        private readonly MLArrayType initialRecurrentType;
        #endregion
    }
}