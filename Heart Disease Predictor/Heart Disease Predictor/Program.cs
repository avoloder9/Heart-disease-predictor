using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Heart_Disease_Predictor
{
    class Program
    {
        static void Main(string[] args)
        {
            string dataPath = "C:\\Users\\Adnan\\Desktop\\Heart-disease-predictor\\heart.csv";
            string modelPath = "heart_model.zip";

            Console.WriteLine("Training the model");
            var modelHandler = new HeartModel(dataPath, modelPath);
            modelHandler.TrainAndSaveModel();

            while (true)
            {
                Console.WriteLine("Enter new patient data");
                var input = HeartModel.GetUserInput();
                var prediction = modelHandler.Predict(input);
                Console.WriteLine($"Probability of heart disease: {prediction}");
                HeartModel.AppendToCsv(dataPath, input, prediction);

                Console.WriteLine("Do you want to enter data for another user?  (y/n): ");
                string answer = Console.ReadLine()?.ToLower();

                if (answer != "y")
                {
                    Console.WriteLine("Thank you for using the system.");
                    System.Threading.Thread.Sleep(2000);
                    break;
                }
            }
        }
    }
}
