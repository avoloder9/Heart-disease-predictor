using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using System.IO;
namespace Heart_Disease_Predictor
{
    public class HeartModel
    {
        private readonly string _dataPath;
        private readonly string _modelPath;
        private readonly MLContext _mlContext;
        private ITransformer _trainedModel;

        public HeartModel(string dataPath, string modelPath)
        {
            _dataPath = dataPath;
            _modelPath = modelPath;
            _mlContext = new MLContext();
        }
        public void TrainAndSaveModel()
        {
            var data = _mlContext.Data.LoadFromTextFile<HeartData>(
                path: _dataPath,
                hasHeader: true,
                separatorChar: ',');

            var split = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var pipeline = _mlContext.Transforms.Concatenate("Features", new[]
                {
                    nameof(HeartData.Age), nameof(HeartData.Sex), nameof(HeartData.Cp),
                    nameof(HeartData.Trestbps), nameof(HeartData.Chol), nameof(HeartData.Fbs),
                    nameof(HeartData.Restecg), nameof(HeartData.Thalach), nameof(HeartData.Exang),
                    nameof(HeartData.Oldpeak), nameof(HeartData.Slope), nameof(HeartData.Ca),
                    nameof(HeartData.Thal)
                }).Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression());
            _trainedModel = pipeline.Fit(split.TrainSet);

            var predictions = _trainedModel.Transform(split.TestSet);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine($"Model trained. Accuracy: {metrics.Accuracy:P2}, AUC: {metrics.AreaUnderRocCurve:P2}");
            _mlContext.Model.Save(_trainedModel, data.Schema, _modelPath);
        }

        public float Predict(HeartData input)
        {
            var predictor = _mlContext.Model.Load(_modelPath, out var _);
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<HeartData, HeartPrediction>(predictor);
            var prediction = predictionEngine.Predict(input);
            var probability = Sigmoid(prediction.Probability);
            AppendToCsv(_dataPath, input, probability);
            return probability;
        }

        public static HeartData GetUserInput()
        {
            return new HeartData
            {
                Age = ReadFloat("Age (18-120)", 18, 120),
                Sex = ReadInt("Sex (0: Female, 1: Male)", 0, 1),
                Cp = ReadInt("Chest Pain Type (0-3)", 0, 3),
                Trestbps = ReadFloat("Resting Blood Pressure (50-250)", 50, 250),
                Chol = ReadFloat("Cholesterol (0-10)", 0, 10),
                Fbs = ReadInt("Fasting Blood Sugar > 120 mg/dl or 3.1 mmol/L (1: true, 0: false)", 0, 1),
                Restecg = ReadInt("Resting ECG Results (0-2)", 0, 2),
                Thalach = ReadFloat("Maximum Heart Rate Achieved (50-250)", 50, 250),
                Exang = ReadInt("Exercise Induced Angina (1: yes, 0: no)", 0, 1),
                Oldpeak = ReadFloat("ST Depression (0 to 6)", 0, 6),
                Slope = ReadInt("Slope of ST Segment (0-2)", 0, 2),
                Ca = ReadInt("Number of Major Vessels (0-3)", 0, 3),
                Thal = ReadInt("Thalassemia (1-3)", 1, 3)
            };
        }
        private float Sigmoid(float score)
        {
            return 1 / (1 + (float)Math.Exp(-score));
        }

        private static float ReadFloat(string prompt, float minValue, float maxValue)
        {
            float result;
            do
            {
                Console.Write($"{prompt}: ");
                if (float.TryParse(Console.ReadLine(), out result) && result >= minValue && result <= maxValue)
                    return result;
                Console.WriteLine($"Please enter a valid value between {minValue} and {maxValue}.");
            } while (true);
        }
        private static int ReadInt(string prompt, int minValue, int maxValue)
        {
            int result;
            do
            {
                Console.Write($"{prompt}: ");
                if (int.TryParse(Console.ReadLine(), out result) && result >= minValue && result <= maxValue)
                    return result;
                Console.WriteLine($"Please enter a valid value between {minValue} and {maxValue}.");
            } while (true);
        }
        public static void AppendToCsv(string filePath, HeartData input, float prediction)
        {
            string dataLine = $"{input.Age},{input.Sex},{input.Cp},{input.Trestbps},{input.Chol},{input.Fbs},{input.Restecg},{input.Thalach},{input.Exang},{input.Oldpeak},{input.Slope},{input.Ca},{input.Thal},{(prediction >= 0.5 ? 1 : 0)}";
            File.AppendAllText(filePath, dataLine + Environment.NewLine);
        }
    }


}
