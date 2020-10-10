using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

using System.Data.SqlClient;

using System.Threading.Tasks;
using PredicaoBiblioteca;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.Azure.Cosmos.Table;

namespace RegressaoLogistica
{
    class Program
    {
        private static string _modelPath;
        private static MLContext _context;
        static readonly string _dataPath = Path.Combine(@"C:\Users\fabri\source\repos\TesteBinary\bin\Debug\Data", @"Pesquisa sobre Gastos Pessoais.csv");
        static async Task Main(string[] args)
        {
            Console.WriteLine("\nPredição de inadimplência\n");
            MLContext mlContext = new MLContext();

            TrainTestData splitDataView = LoadData(mlContext);

            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            // 1. load data and create data pipeline

            Evaluate(mlContext, model, splitDataView.TestSet);

            UseModelWithSingleItem(mlContext, model);

            Console.WriteLine();
            Console.WriteLine("=============== End of process ===============");
            Console.ReadLine();

            //_context = new MLContext();
            //_modelPath = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "model.zip");
            //ITransformer model;
            //DataViewSchema schema;
            //if (!System.IO.File.Exists(_modelPath))
            //{
            //    CloudStorageAccount storageAccount = CloudStorageAccount.Parse("DefaultEndpointsProtocol=https;AccountName=tccinadimplanecia;AccountKey=kNEa4mJH9VYGlHa/PXcpl+rDBBLCwx/i3DI2AA/WpYFEP/sMvGMCBbyRXqwd2OomBTPZiZnE7u/CrKJ96EwyBQ==;EndpointSuffix=core.windows.net");
            //    Microsoft.Azure.Storage.Blob.CloudBlobClient client = Microsoft.Azure.Storage.Blob.BlobAccountExtensions.CreateCloudBlobClient();
            //    var container = client.GetContainerReference("models");

            //    var blob = container.GetBlockBlobReference("model.zip");

            //    await blob.DownloadToFileAsync(_modelPath, System.IO.FileMode.CreateNew);

            //}
            //using (var stream = System.IO.File.OpenRead(_modelPath))
            //{
            //    model = _context.Model.Load(stream, out schema);
            //}
            //ModelInput sampleStatement = new ModelInput
            //{
            //    Idade = 20,
            //    Sexo = 1,
            //    Escolaridade = 2,
            //    Fl_InadimplentePassado = 1,
            //    Renda = 5000,
            //    Desp_Fixas = 1000,
            //    Desp_Variaveis = 3000,
            //    Contas_Atrasadas = 1
            //};

            //var predictionEngine = _context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            //var prediction = predictionEngine.Predict(sampleStatement);

            //Console.WriteLine("A");

        }



        public static TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<Usuario>(_dataPath, hasHeader: true, separatorChar: ';');

            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;


        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var Idade = mlContext.Transforms.Categorical.OneHotEncoding(new[]
          { new InputOutputColumnPair("Idade", "Idade") });
            var Sexo = mlContext.Transforms.Categorical.OneHotEncoding(new[]
        { new InputOutputColumnPair("Sexo", "Sexo") });
            var Escolaridade = mlContext.Transforms.Categorical.OneHotEncoding(new[]
        { new InputOutputColumnPair("Escolaridade", "Escolaridade") });
            var Fl_InadimplentePassado = mlContext.Transforms.Categorical.OneHotEncoding(new[]
        { new InputOutputColumnPair("Fl_InadimplentePassado", "Fl_InadimplentePassado") });
            var Renda = mlContext.Transforms.Categorical.OneHotEncoding(new[]
        { new InputOutputColumnPair("Renda", "Renda") });
            var Desp_Fixas = mlContext.Transforms.Categorical.OneHotEncoding(new[]
        { new InputOutputColumnPair("Desp_Fixas", "Desp_Fixas") });
            var Desp_Variaveis = mlContext.Transforms.Categorical.OneHotEncoding(new[]
        { new InputOutputColumnPair("Desp_Variaveis", "Desp_Variaveis") });
            var Contas_Atrasadas = mlContext.Transforms.Categorical.OneHotEncoding(new[]
        { new InputOutputColumnPair("Contas_Atrasadas", "Contas_Atrasadas") });


            var c = mlContext.Transforms.Concatenate("Features", new[]
              { "Idade", "Sexo", "Escolaridade"
                        , "Fl_InadimplentePassado", "Renda", "Desp_Fixas", "Desp_Variaveis", "Contas_Atrasadas"
               });
            var dataPipe = Idade.Append(Idade).Append(Sexo)
                .Append(Escolaridade).Append(Fl_InadimplentePassado).Append(Renda)
                .Append(Desp_Fixas).Append(Desp_Variaveis)
                .Append(Contas_Atrasadas).Append(c);

            var estimator = dataPipe.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);

            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<Usuario, Predicao> predictionFunction = mlContext.Model.CreatePredictionEngine<Usuario, Predicao>(model);

            Usuario sampleStatement = new Usuario
            {
                Idade = 20,
                Sexo = 1,
                Escolaridade = 2,
                Fl_InadimplentePassado = 1,
                Renda = 5000,
                Desp_Fixas = 1000,
                Desp_Variaveis = 3000,
                Contas_Atrasadas = 1
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Prediction: {(Convert.ToBoolean(resultPrediction.predicao) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        // Main
    } // Program

    class ModelInput
    {
        [LoadColumn(0)]
        public int ID;

        [LoadColumn(1)]
        public int Idade;
        [LoadColumn(2)]
        public int Sexo;
        [LoadColumn(3)]
        public int Escolaridade;
        [LoadColumn(4), ColumnName("Label")]
        public bool Fl_Inadimplente;
        [LoadColumn(5)]
        public int Fl_InadimplentePassado;

        [LoadColumn(6)]
        public int Renda;
        [LoadColumn(7)]
        public int Desp_Fixas;
        [LoadColumn(8)]
        public int Desp_Variaveis;
        [LoadColumn(9)]
        public int Contas_Atrasadas;


    }

    class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }


        public float Score { get; set; }

        public float Probability { get; set; }

    }


}


