using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using mnist.DataStructures;
using Microsoft.ML.Transforms;
/// <summary>
///    基于 https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/MulticlassClassification_MNIST
/// 
///    Lex 2020-5-17修改
/// </summary>
namespace mnist
{
    class Program
    {
        private static string BaseDatasetsRelativePath = @"../../../Data";
        private static string TrianDataRealtivePath = $"{BaseDatasetsRelativePath}/optdigits-train.csv";
        private static string TestDataRealtivePath = $"{BaseDatasetsRelativePath}/optdigits-val.csv";

        private static string TrainDataPath = GetAbsolutePath(TrianDataRealtivePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRealtivePath);

        private static string BaseModelsRelativePath = @"../../../MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/Model.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            // MLContext 类, ML.NET核心类，初始化后 会创建一个新的ML.NET运算环境
            //提供了一种创建数据准备、功能引擎、训练、预测、模型评估的组件的方法。
            //还允许日志记录、执行控制和设置可重复的随机数的能力。
            //可在模型创建工作流对象之间共享该环境。 
            MLContext mlContext = new MLContext();

            Train(mlContext);
            TestSomePredictions(mlContext);

            Console.WriteLine("按任意键退出！");
            Console.ReadKey();
        }

        public static void Train(MLContext mlContext)
        {
            try
            {
                //步骤1：通用数据加载配置
                var trainData = mlContext.Data.LoadFromTextFile(path: TrainDataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );
                Console.WriteLine(DateTime.Now.ToString() + " >> " + "=============== 读取训练数据 ===============");

                var testData = mlContext.Data.LoadFromTextFile(path: TestDataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );
                Console.WriteLine(DateTime.Now.ToString() + " >> " + "=============== 读取测试数据 ===============");

                //步骤2：具有管道数据转换的通用数据过程配置
                //对中小型数据集使用内存中缓存以减少训练时间。处理非常大的数据集时，请勿使用它（删除.AppendCacheCheckpoint（））。
                var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue).
                    Append(mlContext.Transforms.Concatenate("Features", nameof(InputData.PixelValues)).AppendCacheCheckpoint(mlContext));


                //步骤3：设置训练算法，然后创建并配置modelBuilder
                var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
                var trainingPipeline = dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));


                //步骤4：训练模型拟合数据集

                Console.WriteLine(DateTime.Now.ToString() + " >> " + "=============== 训练模型 ===============");
                ITransformer trainedModel = trainingPipeline.Fit(trainData);

                Console.WriteLine(DateTime.Now.ToString() + " >> " + "===== 使用测试数据评估模型的准确性 =====");
                var predictions = trainedModel.Transform(testData);
                var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");

                Common.ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

                mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);

                Console.WriteLine(DateTime.Now.ToString() + " >> " + "模型已保存到： {0}", ModelPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                //return null;
            }
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        private static void TestSomePredictions(MLContext mlContext)
        {
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(trainedModel);

            var resultprediction1 = predEngine.Predict(SampleMNISTData.MNIST1);

            Console.WriteLine($"测试: 1     预测概率               :       0:  {resultprediction1.Score[0]:0.####}");
            Console.WriteLine($"                                           1:  {resultprediction1.Score[1]:0.####}");
            Console.WriteLine($"                                           2:  {resultprediction1.Score[2]:0.####}");
            Console.WriteLine($"                                           3:  {resultprediction1.Score[3]:0.####}");
            Console.WriteLine($"                                           4:  {resultprediction1.Score[4]:0.####}");
            Console.WriteLine($"                                           5:  {resultprediction1.Score[5]:0.####}");
            Console.WriteLine($"                                           6:  {resultprediction1.Score[6]:0.####}");
            Console.WriteLine($"                                           7:  {resultprediction1.Score[7]:0.####}");
            Console.WriteLine($"                                           8:  {resultprediction1.Score[8]:0.####}");
            Console.WriteLine($"                                           9:  {resultprediction1.Score[9]:0.####}");
            Console.WriteLine();

            var resultprediction2 = predEngine.Predict(SampleMNISTData.MNIST2);

            Console.WriteLine($"测试: 7     预测概率              :        0:  {resultprediction2.Score[0]:0.####}");
            Console.WriteLine($"                                           1:  {resultprediction2.Score[1]:0.####}");
            Console.WriteLine($"                                           2:  {resultprediction2.Score[2]:0.####}");
            Console.WriteLine($"                                           3:  {resultprediction2.Score[3]:0.####}");
            Console.WriteLine($"                                           4:  {resultprediction2.Score[4]:0.####}");
            Console.WriteLine($"                                           5:  {resultprediction2.Score[5]:0.####}");
            Console.WriteLine($"                                           6:  {resultprediction2.Score[6]:0.####}");
            Console.WriteLine($"                                           7:  {resultprediction2.Score[7]:0.####}");
            Console.WriteLine($"                                           8:  {resultprediction2.Score[8]:0.####}");
            Console.WriteLine($"                                           9:  {resultprediction2.Score[9]:0.####}");
            Console.WriteLine();

            var resultprediction3 = predEngine.Predict(SampleMNISTData.MNIST3);

            Console.WriteLine($"测试: 9    预测概率               :        0:  {resultprediction3.Score[0]:0.####}");
            Console.WriteLine($"                                           1:  {resultprediction3.Score[1]:0.####}");
            Console.WriteLine($"                                           2:  {resultprediction3.Score[2]:0.####}");
            Console.WriteLine($"                                           3:  {resultprediction3.Score[3]:0.####}");
            Console.WriteLine($"                                           4:  {resultprediction3.Score[4]:0.####}");
            Console.WriteLine($"                                           5:  {resultprediction3.Score[5]:0.####}");
            Console.WriteLine($"                                           6:  {resultprediction3.Score[6]:0.####}");
            Console.WriteLine($"                                           7:  {resultprediction3.Score[7]:0.####}");
            Console.WriteLine($"                                           8:  {resultprediction3.Score[8]:0.####}");
            Console.WriteLine($"                                           9:  {resultprediction3.Score[9]:0.####}");
            Console.WriteLine();
        }
    }
}
