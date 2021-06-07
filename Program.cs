using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace OnnxSR
{
    class Program
    {
        static void Main(string[] args)
        {
            var ressourcePath = Path.Combine(Environment.CurrentDirectory, "Ressources");


            // Read paths
            string modelFilePath = Path.Combine(ressourcePath,"super_resolution.onnx");
            string imageFilePath = Path.Combine(ressourcePath,"demo.png");

            MLContext mlContext = new MLContext();









            // Read image
            using Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>(imageFilePath);
            using Image<Rgb24> imageBic = image.Clone();

            image.Save("./demooGray.png");
            imageBic.Mutate(x => x.Resize(new SixLabors.ImageSharp.Size(image.Width * 3, image.Height * 3)));


            // Resize image

            // Preprocess image
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 1, image.Height, image.Width });
            Tensor<float> inputBic = new DenseTensor<float>(new[] { 1, 3, imageBic.Height, imageBic.Width });

            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var conv = new SixLabors.ImageSharp.ColorSpaces.Conversion.ColorSpaceConverter();
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < image.Height; y++)
            {
                Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                for (int x = 0; x < image.Width; x++)
                {
                    var xxxx = new SixLabors.ImageSharp.ColorSpaces.Conversion.ColorSpaceConverter().Adapt(image[x, y]);
                    var xxx = conv.ToYCbCr(xxxx);
                    input[0, 0, y, x] = xxx.Y;

                }
            }

            for (int y = 0; y < imageBic.Height; y++)
            {
                Span<Rgb24> pixelSpan = imageBic.GetPixelRowSpan(y);
                for (int x = 0; x < imageBic.Width; x++)
                {
                    var xx = conv.ToYCbCr(new SixLabors.ImageSharp.ColorSpaces.Rgb(pixelSpan[x].R, pixelSpan[x].G, pixelSpan[x].B));
                    inputBic[0, 0, y, x] = xx.Y;
                    inputBic[0, 1, y, x] = xx.Cb;
                    inputBic[0, 2, y, x] = xx.Cr;
                }
            }







            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", input)
            };

            using var session1 = new InferenceSession(modelFilePath);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session1.Run(inputs);
            Tensor<float> output = results.First().AsTensor<float>();
            Image<Rgb24> neww = new Image<Rgb24>(image.Width * 3, image.Height * 3);

            for (int y = 0; y < neww.Height; y++)
            {
                for (int x = 0; x < neww.Width; x++)
                {
                    var bicRGB = new SixLabors.ImageSharp.ColorSpaces.Conversion.ColorSpaceConverter().Adapt(imageBic[x, y]);
                    var bicYCbCr = conv.ToYCbCr(bicRGB);

                    var ycbcr = new SixLabors.ImageSharp.ColorSpaces.YCbCr(output[0, 0, y, x], bicYCbCr.Cb, bicYCbCr.Cr);


                    var tt = new SixLabors.ImageSharp.ColorSpaces.Conversion.ColorSpaceConverter().ToRgb(ycbcr);



                    neww[x, y] = tt;
                }
            }
            neww.SaveAsJpeg(Path.Combine(ressourcePath,"FINAL.jpeg"));
            imageBic.SaveAsJpeg(Path.Combine(ressourcePath, "Bicc.jpeg"));

            Console.WriteLine("Super Resolution using ONNX!");
        }
    }
}
