# C#开发人工智能应用：使用SDCA算法 进行MNIST 分类


## 引言：
ML.NET是.NET Foundation(.NET基金会)的一个项目，为开发者提供了基于.NET开发跨平台机器学习应用的平台。ML.NET 是面向.NET开发人员的开源和跨平台机器学习框架。 ML.NET 还包括Model Builder （一个简单的UI工具）和 CLI ，使用自动机器学习（AutoML）构建自定义机器学习（ML）模型变得非常容易。本篇文章将会介绍ML.NET开发环境的安装和使用SDCA算进行MNIST 分类示例。
开发环境
1.安装Visual Studio 
安装Visual Studio 2017 更高版本，建议安装 Visual Studio 2019（https://visualstudio.microsoft.com/zh-hans/vs/community/），社区版是免费的，安装的时候需要选择安装.NetCore功能。

2.安装ML.net模型生成器
模型生成器是开发人员在其 .NET 应用程序中构建、训练和运送自定义机器学习模型的简单 UI 工具。没有 ML 专业知识的开发人员可以使用 Visual Studio 中的这个简单的可视化界面连接到存储在文件或 SQL Server 中的数据，训练模型，并生成用于模型培训和使用的代码。安装ML.net模型构建器（微软官方下载地址 ML.NET Model Builder tool，https://marketplace.visualstudio.com/items?itemName=MLNET.07），下载以后直接双击安装即可。安装需要关闭Visual Studio。也可通过Visual Studio中的扩展功能选项安装。安装完成后工程的ADD菜单将会出现Machine Learning选项。

3.新建C# 命令行项目 通过NuGet添加程序包
新建一个.NET Core命令行程序项目
在解决方案资源管理器项目名称上单击鼠标右键，选择“管理NuGet程序包”

## 应用实例
在本示例中，您将了解如何使用 ML.NET将MNIST 数据集中的手写数字进行从 0-9分类。这是一个多类分类问题，我们将使用SDCA（随机双坐标上升）算法来解决。
1.MNIST数据集
MNIST 数据集包含数字的手写图像，范围从 0 到 9。数据来源：UCI 机器学习存储库http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

我们使用的 MNIST 数据集包含 65 列数字。每行的前 64 列是 0 到 16 范围内的整数值。这些值的计算方法是将 32 x 32 位图划分为 4 x 4 的非重叠块。ON 像素的数量在每个块中计数，这些块生成 8 x 8 的输入矩阵。每行中的最后一列是由前 64 列中的值表示的数字。前 64 列是我们的功能，我们的 ML 模型将使用这些功能对测试图像进行分类。我们培训和验证数据集的最后一列是标签 - 我们将使用 ML 模型预测的实际数字。我们将构建的 ML 模型将返回给定图像的概率，该图像是 0 到 9 的数字之一。

2.关键代码
一个机器学习应用的建立和使用： 需要经过  构建-》训练-》评估-》部署 四个步骤。
要解决手写数字识别问题，首先我们将构建一个 ML 模型。然后，我们将根据现有数据对模型进行训练，评估其良好程度，最后我们将使用模型来预测给定图像表示的数字。

关键对象
```CSharp
            // MLContext 类, ML.NET核心类，初始化后 会创建一个新的ML.NET运算环境
            //提供了一种创建数据准备、功能引擎、训练、预测、模型评估的组件的方法。
            //还允许日志记录、执行控制和设置可重复的随机数的能力。
            //可在模型创建工作流对象之间共享该环境。 
            MLContext mlContext = new MLContext();
```

数据加载和模型训练：
```CSharp
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
```

模型测试：
```CSharp
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
```
测试，查看预测概率

## 示例代码 https://github.com/LEX-2025/MNIST-APP


 

 
