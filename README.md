# C#�����˹�����Ӧ�ã�ʹ��SDCA�����MNIST ����

| ML.NET version | API type          | Status                        | App Type    | Data type | Scenario            | ML Task                   | Algorithms                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4           | Dynamic API | Up-to-date | Console app | .csv files | MNIST classification | Multi-class classification | Sdca Multi-class |

In this introductory sample, you'll see how to use [ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) to classify handwritten digits from 0 to 9 using the MNIST dataset. This is a **multiclass classification** problem that we will solve using SDCA (Stochastic Dual Coordinate Ascent) algorithm.

## ���ԣ�
ML.NET��.NET Foundation(.NET�����)��һ����Ŀ��Ϊ�������ṩ�˻���.NET������ƽ̨����ѧϰӦ�õ�ƽ̨��ML.NET ������.NET������Ա�Ŀ�Դ�Ϳ�ƽ̨����ѧϰ��ܡ� ML.NET ������Model Builder ��һ���򵥵�UI���ߣ��� CLI ��ʹ���Զ�����ѧϰ��AutoML�������Զ������ѧϰ��ML��ģ�ͱ�÷ǳ����ס���ƪ���½������ML.NET���������İ�װ��ʹ��SDCA�����MNIST ����ʾ����
��������
1.��װVisual Studio 
��װVisual Studio 2017 ���߰汾�����鰲װ Visual Studio 2019��https://visualstudio.microsoft.com/zh-hans/vs/community/��������������ѵģ���װ��ʱ����Ҫѡ��װ.NetCore���ܡ�

2.��װML.netģ��������
ģ���������ǿ�����Ա���� .NET Ӧ�ó����й�����ѵ���������Զ������ѧϰģ�͵ļ� UI ���ߡ�û�� ML רҵ֪ʶ�Ŀ�����Ա����ʹ�� Visual Studio �е�����򵥵Ŀ��ӻ��������ӵ��洢���ļ��� SQL Server �е����ݣ�ѵ��ģ�ͣ�����������ģ����ѵ��ʹ�õĴ��롣��װML.netģ�͹�������΢��ٷ����ص�ַ ML.NET Model Builder tool��https://marketplace.visualstudio.com/items?itemName=MLNET.07���������Ժ�ֱ��˫����װ���ɡ���װ��Ҫ�ر�Visual Studio��Ҳ��ͨ��Visual Studio�е���չ����ѡ�װ����װ��ɺ󹤳̵�ADD�˵��������Machine Learningѡ�

## Ӧ��ʵ��
�ڱ�ʾ���У������˽����ʹ�� ML.NET��MNIST ���ݼ��е���д���ֽ��д� 0-9���ࡣ����һ������������⣬���ǽ�ʹ��SDCA�����˫�����������㷨�������
1.MNIST���ݼ�
MNIST ���ݼ��������ֵ���дͼ�񣬷�Χ�� 0 �� 9��������Դ��UCI ����ѧϰ�洢��http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

����ʹ�õ� MNIST ���ݼ����� 65 �����֡�ÿ�е�ǰ 64 ���� 0 �� 16 ��Χ�ڵ�����ֵ����Щֵ�ļ��㷽���ǽ� 32 x 32 λͼ����Ϊ 4 x 4 �ķ��ص��顣ON ���ص�������ÿ�����м�������Щ������ 8 x 8 ���������ÿ���е����һ������ǰ 64 ���е�ֵ��ʾ�����֡�ǰ 64 �������ǵĹ��ܣ����ǵ� ML ģ�ͽ�ʹ����Щ���ܶԲ���ͼ����з��ࡣ������ѵ����֤���ݼ������һ���Ǳ�ǩ - ���ǽ�ʹ�� ML ģ��Ԥ���ʵ�����֡����ǽ������� ML ģ�ͽ����ظ���ͼ��ĸ��ʣ���ͼ���� 0 �� 9 ������֮һ��

2.�ؼ�����
һ������ѧϰӦ�õĽ�����ʹ�ã� ��Ҫ����  ����-��ѵ��-������-������ �ĸ����衣
Ҫ�����д����ʶ�����⣬�������ǽ�����һ�� ML ģ�͡�Ȼ�����ǽ������������ݶ�ģ�ͽ���ѵ�������������ó̶ȣ�������ǽ�ʹ��ģ����Ԥ�����ͼ���ʾ�����֡�

�ؼ�����
```CSharp
            // MLContext ��, ML.NET�����࣬��ʼ���� �ᴴ��һ���µ�ML.NET���㻷��
            //�ṩ��һ�ִ�������׼�����������桢ѵ����Ԥ�⡢ģ������������ķ�����
            //��������־��¼��ִ�п��ƺ����ÿ��ظ����������������
            //����ģ�ʹ�������������֮�乲��û����� 
            MLContext mlContext = new MLContext();
```

���ݼ��غ�ģ��ѵ����
```CSharp
           try
            {
                //����1��ͨ�����ݼ�������
                var trainData = mlContext.Data.LoadFromTextFile(path: TrainDataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );
                Console.WriteLine(DateTime.Now.ToString() + " >> " + "=============== ��ȡѵ������ ===============");

                var testData = mlContext.Data.LoadFromTextFile(path: TestDataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );
                Console.WriteLine(DateTime.Now.ToString() + " >> " + "=============== ��ȡ�������� ===============");

                //����2�����йܵ�����ת����ͨ�����ݹ�������
                //����С�����ݼ�ʹ���ڴ��л����Լ���ѵ��ʱ�䡣����ǳ�������ݼ�ʱ������ʹ������ɾ��.AppendCacheCheckpoint��������
                var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue).
                    Append(mlContext.Transforms.Concatenate("Features", nameof(InputData.PixelValues)).AppendCacheCheckpoint(mlContext));


                //����3������ѵ���㷨��Ȼ�󴴽�������modelBuilder
                var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
                var trainingPipeline = dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));


                //����4��ѵ��ģ��������ݼ�
                Console.WriteLine(DateTime.Now.ToString() + " >> " + "=============== ѵ��ģ�� ===============");
                ITransformer trainedModel = trainingPipeline.Fit(trainData);

                Console.WriteLine(DateTime.Now.ToString() + " >> " + "===== ʹ�ò�����������ģ�͵�׼ȷ�� =====");
                var predictions = trainedModel.Transform(testData);
                var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");

                Common.ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

                mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);

                Console.WriteLine(DateTime.Now.ToString() + " >> " + "ģ���ѱ��浽�� {0}", ModelPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                //return null;
            }
```

ģ�Ͳ��ԣ�
```CSharp
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(trainedModel);

            var resultprediction1 = predEngine.Predict(SampleMNISTData.MNIST1);

            Console.WriteLine($"����: 1     Ԥ�����               :       0:  {resultprediction1.Score[0]:0.####}");
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
���ԣ��鿴Ԥ�����

## ʾ������ https://github.com/LEX-2025/MNIST-APP


 

 
