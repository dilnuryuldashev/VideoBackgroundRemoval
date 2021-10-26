/* 
*   Robust Video Matting
*   Copyright (c) 2021 Yusuf Olokoba.
*/

namespace NatSuite.ML.Vision {

    using System;
    using UnityEngine;

    public sealed partial class RobustVideoMattingPredictor {

        /// <summary>
        /// Alpha matte.
        /// Each pixel in the map returns a probability of that pixel location being a person (~1.0) or background (~0.0).
        /// </summary>
        public sealed class Matte {

            #region --Client API--
            /// <summary>
            /// Map width.
            /// </summary>
            public readonly int width;

            /// <summary>
            /// Map height.
            /// </summary>
            public readonly int height;

            /// <summary>
            /// Render the probability map to a texture.
            /// Each pixel will have value `(p, p, p, 1.0)` where `p` is the foreground probability for that pixel.
            /// </summary>
            /// <param name="destination">Destination texture.</param>
            public unsafe void Render (RenderTexture destination) {
                // Check texture
                if (!destination)
                    throw new ArgumentNullException(nameof(destination));
                // Create buffer
                using (var matteBuffer = new ComputeBuffer(width * height, sizeof(float))) {
                    // Upload
                    matteBuffer.SetData(data);
                    // Create temporary                
                    var descriptor = new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGB32, 0) {
                        enableRandomWrite = true
                    };
                    var tempBuffer = RenderTexture.GetTemporary(descriptor);
                    tempBuffer.Create();
                    // Render
                    renderer = renderer ?? (ComputeShader)Resources.Load(@"RobustVideoMattingRenderer");
                    renderer.SetBuffer(0, "Matte", matteBuffer);
                    renderer.SetTexture(0, "Result", tempBuffer);
                    renderer.GetKernelThreadGroupSizes(0, out var gx, out var gy, out var _);
                    renderer.Dispatch(0, Mathf.CeilToInt((float)width / gx), Mathf.CeilToInt((float)height / gy), 1);
                    // Blit to destination
                    Graphics.Blit(tempBuffer, destination);
                    RenderTexture.ReleaseTemporary(tempBuffer);
                }
            }
            #endregion


            #region --Operations--
            private readonly float[] data;
            private static ComputeShader renderer;

            internal Matte (int width, int height, float[] data) {
                this.width = width;
                this.height = height;
                this.data = data;
            }
            #endregion
        }
    }
}