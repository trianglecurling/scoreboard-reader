using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.ML;

namespace Contours
{
    class Program
    {
        static void Main(string[] args)
        {
            var im = Cv2.ImRead("train.png");
            var imOrig = im.Clone();

            Cv2.CvtColor(im, im, ColorConversionCodes.BGR2GRAY);
            Cv2.GaussianBlur(im, im , new Size(5, 5), 0);
            Cv2.AdaptiveThreshold(im, im, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 11, 2.0d);
            
            var contours = new Mat[0];
            var hierarchy = new Mat();
            Cv2.FindContours(im, out contours, hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
            var boxes = contours
                .Where(c => Cv2.ContourArea(c) > 50.0 && Cv2.ContourArea(c) < 2000.0 )
                .Select(c => Cv2.BoundingRect(c))
                .OrderBy(r => (int)r.Y / 28) //magic rounding numbers for rows and cols
                .ThenBy(r => (int)r.X / 20);

            var cachedResponses = File.ReadAllText("responses.data");

            var samples = new Mat();
            var responses = new Mat();
            int i = 0;

            foreach (var b in boxes){
                if (b.Height > 28)
                {
                    var imDisp = imOrig.Clone();
                    Cv2.Rectangle(imDisp, b, Scalar.Red, 2);
                    var roi = im[b];
                    Cv2.Resize(roi, roi, new Size(10, 10));
                    Cv2.ImShow("norm", imDisp);
                    int key = Int32.Parse(""+cachedResponses[i++]);
                    responses.PushBack(key);
                    var sample = roi.Reshape(1, 100);
                    samples.PushBack(sample);
                }
            }
            Console.WriteLine("training complete");

            var samplesFloat = new Mat();
            var responseFloat = new Mat();
            samples.ConvertTo(samplesFloat, MatType.CV_32FC1);
            responses.ConvertTo(responseFloat, MatType.CV_32FC1);
            samplesFloat = samplesFloat.Reshape(1, 125);
            var model = KNearest.Create();
            model.Train(samplesFloat, SampleTypes.RowSample, responseFloat);

//=================== START TEST IMAGE DIGIT DETECTION ================
            var dists = new Mat();
            var results = new Mat();
            var neighborResponses = new Mat();
            
            var test = Cv2.ImRead("scoreboard.png");
            var testOrig = test.Clone();
            
            var start = DateTime.Now;
            Cv2.CvtColor(test, test, ColorConversionCodes.BGR2GRAY);
            Cv2.GaussianBlur(test, test , new Size(5, 5), 0);
            Cv2.AdaptiveThreshold(test, test, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 11, 2.0d);
            
            contours = new Mat[0];
            hierarchy = new Mat();
            Cv2.FindContours(test, out contours, hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
            foreach (var cnt in contours)
            {
                var area = Cv2.ContourArea(cnt);
                if( area > 50.0 && area < 400) // 400 filters out red team scoreboard "label"
                {
                    var rect = Cv2.BoundingRect(cnt);

                    if (rect.Height > 28)
                    {
                        Cv2.Rectangle(testOrig, rect, Scalar.Red, 2);
                        var roi = test[rect];
                        Cv2.Resize(roi, roi, new Size(10, 10));
                        roi.ConvertTo(roi, MatType.CV_32FC1);
                        roi = roi.Reshape(1, 1);
                        Cv2.PutText(testOrig, $"{model.FindNearest(roi, 1, results, neighborResponses, dists)}", rect.TopLeft, HersheyFonts.HersheyPlain, 1, Scalar.White, 1);
                    }
                }
            }
            Console.WriteLine(DateTime.Now - start);
            Cv2.ImShow("norm", testOrig);
            Cv2.WaitKey(0);
        }
    }
}
