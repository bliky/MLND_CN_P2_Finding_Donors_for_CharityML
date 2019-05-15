#!/usr/bin/env python
# coding: utf-8

# # 机器学习纳米学位
# ## 监督学习
# ## 项目2: 为*CharityML*寻找捐献者

# 欢迎来到机器学习工程师纳米学位的第二个项目！在此文件中，有些示例代码已经提供给你，但你还需要实现更多的功能让项目成功运行。除非有明确要求，你无须修改任何已给出的代码。以**'练习'**开始的标题表示接下来的代码部分中有你必须要实现的功能。每一部分都会有详细的指导，需要实现的部分也会在注释中以'TODO'标出。请仔细阅读所有的提示！
# 
# 除了实现代码外，你还必须回答一些与项目和你的实现有关的问题。每一个需要你回答的问题都会以**'问题 X'**为标题。请仔细阅读每个问题，并且在问题后的**'回答'**文字框中写出完整的答案。我们将根据你对问题的回答和撰写代码所实现的功能来对你提交的项目进行评分。
# >**提示：**Code 和 Markdown 区域可通过**Shift + Enter**快捷键运行。此外，Markdown可以通过双击进入编辑模式。

# ## 开始
# 
# 在这个项目中，你将使用1994年美国人口普查收集的数据，选用几个监督学习算法以准确地建模被调查者的收入。然后，你将根据初步结果从中选择出最佳的候选算法，并进一步优化该算法以最好地建模这些数据。你的目标是建立一个能够准确地预测被调查者年收入是否超过50000美元的模型。这种类型的任务会出现在那些依赖于捐款而存在的非营利性组织。了解人群的收入情况可以帮助一个非营利性的机构更好地了解他们要多大的捐赠，或是否他们应该接触这些人。虽然我们很难直接从公开的资源中推断出一个人的一般收入阶层，但是我们可以（也正是我们将要做的）从其他的一些公开的可获得的资源中获得一些特征从而推断出该值。
# 
# 这个项目的数据集来自[UCI机器学习知识库](https://archive.ics.uci.edu/ml/datasets/Census+Income)。这个数据集是由Ron Kohavi和Barry Becker在发表文章_"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_之后捐赠的，你可以在Ron Kohavi提供的[在线版本](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf)中找到这个文章。我们在这里探索的数据集相比于原有的数据集有一些小小的改变，比如说移除了特征`'fnlwgt'` 以及一些遗失的或者是格式不正确的记录。

# ----
# ## 探索数据
# 运行下面的代码单元以载入需要的Python库并导入人口普查数据。注意数据集的最后一列`'income'`将是我们需要预测的列（表示被调查者的年收入会大于或者是最多50,000美元），人口普查数据中的每一列都将是关于被调查者的特征。

# In[1]:


# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # 允许为DataFrame使用display()

# 导入附加的可视化代码visuals.py
import visuals as vs

# 为notebook提供更加漂亮的可视化
get_ipython().run_line_magic('matplotlib', 'inline')

# 导入人口普查数据
data = pd.read_csv("census.csv")

# 成功 - 显示第一条记录
display(data.head(n=1))


# ### 练习：数据探索
# 首先我们对数据集进行一个粗略的探索，我们将看看每一个类别里会有多少被调查者？并且告诉我们这些里面多大比例是年收入大于50,000美元的。在下面的代码单元中，你将需要计算以下量：
# 
# - 总的记录数量，`'n_records'`
# - 年收入大于50,000美元的人数，`'n_greater_50k'`.
# - 年收入最多为50,000美元的人数 `'n_at_most_50k'`.
# - 年收入大于50,000美元的人所占的比例， `'greater_percent'`.
# 
# **提示：** 您可能需要查看上面的生成的表，以了解`'income'`条目的格式是什么样的。 

# In[9]:


# TODO：总的记录数
n_records = data.shape[0]

# TODO：被调查者的收入大于$50,000的人数
n_greater_50k = data[data.income == '>50K'].shape[0]

# TODO：被调查者的收入最多为$50,000的人数
n_at_most_50k = data[data.income == '<=50K'].shape[0]

# TODO：被调查者收入大于$50,000所占的比例
greater_percent = n_greater_50k / n_records

# 打印结果
print ("Total number of records: {}".format(n_records))
print ("Individuals making more than $50,000: {}".format(n_greater_50k))
print ("Individuals making at most $50,000: {}".format(n_at_most_50k))
print ("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))


# ----
# ## 准备数据
# 在数据能够被作为输入提供给机器学习算法之前，它经常需要被清洗，格式化，和重新组织 - 这通常被叫做**预处理**。幸运的是，对于这个数据集，没有我们必须处理的无效或丢失的条目，然而，由于某一些特征存在的特性我们必须进行一定的调整。这个预处理都可以极大地帮助我们提升几乎所有的学习算法的结果和预测能力。
# 
# ### 获得特征和标签
# `income` 列是我们需要的标签，记录一个人的年收入是否高于50K。 因此我们应该把他从数据中剥离出来，单独存放。

# In[10]:


# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis = 1)


# ### 转换倾斜的连续特征
# 
# 一个数据集有时可能包含至少一个靠近某个数字的特征，但有时也会有一些相对来说存在极大值或者极小值的不平凡分布的的特征。算法对这种分布的数据会十分敏感，并且如果这种数据没有能够很好地规一化处理会使得算法表现不佳。在人口普查数据集的两个特征符合这个描述：'`capital-gain'`和`'capital-loss'`。
# 
# 运行下面的代码单元以创建一个关于这两个特征的条形图。请注意当前的值的范围和它们是如何分布的。

# In[11]:


# 可视化 'capital-gain'和'capital-loss' 两个特征
vs.distribution(features_raw)


# 对于高度倾斜分布的特征如`'capital-gain'`和`'capital-loss'`，常见的做法是对数据施加一个<a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">对数转换</a>，将数据转换成对数，这样非常大和非常小的值不会对学习算法产生负面的影响。并且使用对数变换显著降低了由于异常值所造成的数据范围异常。但是在应用这个变换时必须小心：因为0的对数是没有定义的，所以我们必须先将数据处理成一个比0稍微大一点的数以成功完成对数转换。
# 
# 运行下面的代码单元来执行数据的转换和可视化结果。再次，注意值的范围和它们是如何分布的。

# In[12]:


# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 可视化对数转换后 'capital-gain'和'capital-loss' 两个特征
vs.distribution(features_raw, transformed = True)


# ### 规一化数字特征
# 除了对于高度倾斜的特征施加转换，对数值特征施加一些形式的缩放通常会是一个好的习惯。在数据上面施加一个缩放并不会改变数据分布的形式（比如上面说的'capital-gain' or 'capital-loss'）；但是，规一化保证了每一个特征在使用监督学习器的时候能够被平等的对待。注意一旦使用了缩放，观察数据的原始形式不再具有它本来的意义了，就像下面的例子展示的。
# 
# 运行下面的代码单元来规一化每一个数字特征。我们将使用[`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)来完成这个任务。

# In[13]:


from sklearn.preprocessing import MinMaxScaler

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 显示一个经过缩放的样例记录
display(features_raw.head(n = 1))


# ### 练习：数据预处理
# 
# 从上面的**数据探索**中的表中，我们可以看到有几个属性的每一条记录都是非数字的。通常情况下，学习算法期望输入是数字的，这要求非数字的特征（称为类别变量）被转换。转换类别变量的一种流行的方法是使用**独热编码**方案。独热编码为每一个非数字特征的每一个可能的类别创建一个_“虚拟”_变量。例如，假设`someFeature`有三个可能的取值`A`，`B`或者`C`，。我们将把这个特征编码成`someFeature_A`, `someFeature_B`和`someFeature_C`.
# 
# | 特征X |                    | 特征X_A | 特征X_B | 特征X_C |
# | :-: |                            | :-: | :-: | :-: |
# |  B  |  | 0 | 1 | 0 |
# |  C  | ----> 独热编码 ----> | 0 | 0 | 1 |
# |  A  |  | 1 | 0 | 0 |
# 
# 此外，对于非数字的特征，我们需要将非数字的标签`'income'`转换成数值以保证学习算法能够正常工作。因为这个标签只有两种可能的类别（"<=50K"和">50K"），我们不必要使用独热编码，可以直接将他们编码分别成两个类`0`和`1`，在下面的代码单元中你将实现以下功能：
#  - 使用[`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies)对`'features_raw'`数据来施加一个独热编码。
#  - 将目标标签`'income_raw'`转换成数字项。
#    - 将"<=50K"转换成`0`；将">50K"转换成`1`。

# In[56]:


# cols = list(set([col.split('_')[0] for col in pd.get_dummies(features_raw).columns if col not in features_raw.columns]))

# print(cols)

# cols[2] = 'education_level'

print(cols)
print(features_raw.shape)
print(pd.get_dummies(features_raw).shape)
print([col for col in features_raw.columns if col not in pd.get_dummies(features_raw).columns])

features_raw[cols].head()

# income_raw.apply(lambda x: 1 if x=='>50K' else 0)


# In[58]:


# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)

# TODO：将'income_raw'编码成数字值
income = income_raw.apply(lambda x: 1 if x=='>50K' else 0)

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))

# 移除下面一行的注释以观察编码的特征名字
# print(encoded)


# ### 混洗和切分数据
# 现在所有的 _类别变量_ 已被转换成数值特征，而且所有的数值特征已被规一化。和我们一般情况下做的一样，我们现在将数据（包括特征和它们的标签）切分成训练和测试集。其中80%的数据将用于训练和20%的数据用于测试。然后再进一步把训练数据分为训练集和验证集，用来选择和优化模型。
# 
# 运行下面的代码单元来完成切分。

# In[59]:


# 导入 train_test_split
from sklearn.model_selection import train_test_split

# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0,
                                                    stratify = income)
# 将'X_train'和'y_train'进一步切分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                    stratify = y_train)

# 显示切分的结果
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Validation set has {} samples.".format(X_val.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))


# ----
# ## 评价模型性能
# 在这一部分中，我们将尝试四种不同的算法，并确定哪一个能够最好地建模数据。四种算法包含一个*天真的预测器* 和三个你选择的监督学习器。

# ### 评价方法和朴素的预测器
# *CharityML*通过他们的研究人员知道被调查者的年收入大于\$50,000最有可能向他们捐款。因为这个原因*CharityML*对于准确预测谁能够获得\$50,000以上收入尤其有兴趣。这样看起来使用**准确率**作为评价模型的标准是合适的。另外，把*没有*收入大于\$50,000的人识别成年收入大于\$50,000对于*CharityML*来说是有害的，因为他想要找到的是有意愿捐款的用户。这样，我们期望的模型具有准确预测那些能够年收入大于\$50,000的能力比模型去**查全**这些被调查者*更重要*。我们能够使用**F-beta score**作为评价指标，这样能够同时考虑查准率和查全率：
# 
# $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
# 
# 
# 尤其是，当 $\beta = 0.5$ 的时候更多的强调查准率，这叫做**F$_{0.5}$ score** （或者为了简单叫做F-score）。

# ### 问题 1 - 天真的预测器的性能
# 
# 通过查看收入超过和不超过 \$50,000 的人数，我们能发现多数被调查者年收入没有超过 \$50,000。如果我们简单地预测说*“这个人的收入没有超过 \$50,000”*，我们就可以得到一个 准确率超过 50% 的预测。这样我们甚至不用看数据就能做到一个准确率超过 50%。这样一个预测被称作是天真的。通常对数据使用一个*天真的预测器*是十分重要的，这样能够帮助建立一个模型表现是否好的基准。 使用下面的代码单元计算天真的预测器的相关性能。将你的计算结果赋值给`'accuracy'`, `‘precision’`, `‘recall’` 和 `'fscore'`，这些值会在后面被使用，请注意这里不能使用scikit-learn，你需要根据公式自己实现相关计算。
# 
# *如果我们选择一个无论什么情况都预测被调查者年收入大于 \$50,000 的模型，那么这个模型在**验证集上**的准确率，查准率，查全率和 F-score是多少？*  
# 

# In[80]:


y_val[y_val==1].shape[0] / y_val.shape[0]


# In[66]:


#不能使用scikit-learn，你需要根据公式自己实现相关计算。

#TODO： 计算准确率
accuracy = y_val[y_val==1].shape[0] / y_val.shape[0]

# TODO： 计算查准率 Precision
precision = y_val[y_val==1].shape[0] / y_val.shape[0]

# TODO： 计算查全率 Recall
recall = 1

# TODO： 使用上面的公式，设置beta=0.5，计算F-score
fscore = (1 + 0.25) * precision * recall/(0.25*precision + recall)

# 打印结果
print ("Naive Predictor on validation data: \n     Accuracy score: {:.4f} \n     Precision: {:.4f} \n     Recall: {:.4f} \n     F-score: {:.4f}".format(accuracy, precision, recall, fscore))


# ## 监督学习模型
# ### 问题 2 - 模型应用
# 
# 你能够在 [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) 中选择以下监督学习模型
# - 高斯朴素贝叶斯 (GaussianNB)
# - 决策树 (DecisionTree)
# - 集成方法 (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K近邻 (K Nearest Neighbors)
# - 随机梯度下降分类器 (SGDC)
# - 支撑向量机 (SVM)
# - Logistic回归（LogisticRegression）
# 
# 从上面的监督学习模型中选择三个适合我们这个问题的模型，并回答相应问题。

# ### 模型1
# 
# **模型名称**
# 
# 回答：高斯朴素贝叶斯 (GaussianNB)
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：文本分类，文档过滤，博客订阅源过滤，垃圾邮件检测
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：1）发源于古典数学理论，有稳定的分类效率； 2）对小规模的数据表现很好，能处理多分类任务，适合增量式训练，尤其数据超出内存时，可以一批批地增量训练； 3）对缺失数据不太敏感，算法比较简单
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# 回答：1）理论上，与其他分类算法相比具有最小的误差率，但实际上并非总是如此，这是因为朴素贝叶斯模型给定输出类别的情况下，假设属性之间相互独立，但这个假设在实际应用中往往是不成立的，在属性个数比较多或者属性之间相关性较大时，分类效果不好；2）需要知道先验概率，且先验概率很多时候取决于假设，假设的模型可以有很多种，因此在某些时候会由于假设的先验模型的原因导致预测效果不佳；3）由于我们是通过先验和数据来决定后验的概率从而决定分类，所以分类决策存在一定的错误率；4）对输入数据的表达方式很敏感
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：人口普查数据量大，影响预测目标的特征量多，和其他分类算法相比，朴素贝叶斯具有的一大主要优势是能够处理大量特征，比较简单，能够处理大规模样本集，并且训练和预测速度很快

# ### 模型2
# 
# **模型名称**
# 
# 回答：决策树 (DecisionTree)
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：由于决策树具有易于解释的特点，因此它是商务分析，医疗决策和政策制定领域里应用最为广泛的数据挖掘方法之一。通常，决策树的构造是自动进行的，专家们可以利用生成的决策树来理解问题的某些关键因素，然后对其加以改进，以便更好地与他的观点相匹配。这一过程允许机器协助专家进行决策，并清晰地展示出推导的路径，从而我们可以据此来判断预测的质量。如今，决策树以这样的形式被广泛运用于众多应用系统中，包括顾客调查，金融风险分析，辅助诊断和交通预测。—— 引自 书籍《集体智慧编程》
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：1）决策树算法中学习简单的决策规则建立决策树模型的过程非常容易理解；2）决策树模型可以可视化，非常直观；3）应用范围广，可用于分类和回归，而且非常容易做多类别的分类；4）能够处理数值型和连续的样本特征
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# 回答：1）很容易在训练数据中生成复杂的树结构，造成过拟合；2）基于启发式的贪心算法建立，这种算法不能保证建立全局最优的决策树
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：首先，特征包含了分类特征和数值类特征，决策树更好的处理多类型特性；其次，决策树使得我们的模型更具解释性，以方便我们更好地向CharityML解释我们的预测结果，同时方便我们找到于最终预测结果最最相关的特征

# ### 模型3
# 
# **模型名称**
# 
# 回答：支撑向量机 (SVM)
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：因为支持向量机在高维数据集上有不错的表现，因此它们时常被用于解决数据量很大的科学问题，以及其他需要处理极复杂数据集的问题。其中的一些例子如：1）对面部表情进行分类；2）使用军事数据侦测入侵者；3）根据蛋白质序列预测蛋白质结构；4）笔迹识别；5）确定地震期间的潜在危害 —— 引自 书籍《集体智慧编程》
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：1）非线性映射是SVM的理论基础，SVM利用内积核函数代替高维空间的非线性映射；2）对特征空间划分的最优超平面是SVM的目标，最大化分类边际的思想是SVM方法的核心；3）支持向量是SVM的训练结果，在SVM分类决策中起决定作用的是支持向量；4）SVM是一种有坚实理论基础的新颖的小样本学习方法，它基本不涉及概率测度和大数定律等，因此，不同于现有的统计方法。从本质上看，它避开了从归纳到演绎的传统过程，实现了高效的从训练样本到预测样本“转导推理”大大简化了通常的分类和回归问题；5）SVM的最终决策函数只由少数的支持向量决定，计算的复杂度取决于支持向量的数目，而不是样本空间的维数，这在某种意义上避免了“维数灾难”；6）少数支持向量决定了最终结果，这不但可以帮助我们可以抓住关键样本，“剔除”大量冗余样本，而且注定了该方法不但算法简单，而且具有较好的鲁棒性；
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# 回答：1）SVM对大规模训练样本难以实施；2）用SVM解决多分类问题存在困难
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：这是个二元分类问题，特征空间复杂度高，而支持向量机在这方面的分类效果比较好

# ### 练习 - 创建一个训练和预测的流水线
# 为了正确评估你选择的每一个模型的性能，创建一个能够帮助你快速有效地使用不同大小的训练集并在验证集上做预测的训练和验证的流水线是十分重要的。
# 你在这里实现的功能将会在接下来的部分中被用到。在下面的代码单元中，你将实现以下功能：
# 
#  - 从[`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)中导入`fbeta_score`和`accuracy_score`。
#  - 用训练集拟合学习器，并记录训练时间。
#  - 对训练集的前300个数据点和验证集进行预测并记录预测时间。
#  - 计算预测训练集的前300个数据点的准确率和F-score。
#  - 计算预测验证集的准确率和F-score。

# In[81]:


# TODO：从sklearn中导入两个评价指标 - fbeta_score和accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_val, y_val): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_val: features validation set
       - y_val: income validation set
    '''
    
    results = {}
    
    # TODO：使用sample_size大小的训练数据来拟合学习器
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # 获得程序开始时间
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # 获得程序结束时间
    
    # TODO：计算训练时间
    results['train_time'] = end - start
    
    # TODO: 得到在验证集上的预测值
    #       然后得到对前300个训练数据的预测结果
    start = time() # 获得程序开始时间
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train)
    end = time() # 获得程序结束时间
    
    # TODO：计算预测用时
    results['pred_time'] = end - start
            
    # TODO：计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train[:300])
        
    # TODO：计算在验证上的准确率
    results['acc_val'] = accuracy_score(y_val, predictions_val)
    
    # TODO：计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train[:300], predictions_train[:300], beta=0.5)
        
    # TODO：计算验证集上的F-score
    results['f_val'] = fbeta_score(y_val, predictions_val, beta=0.5)
       
    # 成功
    print ("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # 返回结果
    return results


# ### 练习：初始模型的评估
# 在下面的代码单元中，您将需要实现以下功能：             
# - 导入你在前面讨论的三个监督学习模型。             
# - 初始化三个模型并存储在`'clf_A'`，`'clf_B'`和`'clf_C'`中。
#   - 使用模型的默认参数值，在接下来的部分中你将需要对某一个模型的参数进行调整。             
#   - 设置`random_state`  (如果有这个参数)。       
# - 计算1%， 10%， 100%的训练数据分别对应多少个数据点，并将这些值存储在`'samples_1'`, `'samples_10'`, `'samples_100'`中
# 
# **注意：**取决于你选择的算法，下面实现的代码可能需要一些时间来运行！

# In[82]:


# TODO：从sklearn中导入三个监督学习模型
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# TODO：初始化三个模型
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier()
clf_C = SVC()

# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = int(X_train.shape[0] * 0.01)
samples_10 = int(X_train.shape[0] * 0.1)
samples_100 = X_train.shape[0]

# 收集学习器的结果
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)

# 对选择的三个模型得到的评价结果进行可视化
vs.evaluate(results, accuracy, fscore)


# ----
# ## 提高效果
# 
# 在这最后一节中，您将从三个有监督的学习模型中选择 *最好的* 模型来使用学生数据。你将在整个训练集（`X_train`和`y_train`）上使用网格搜索优化至少调节一个参数以获得一个比没有调节之前更好的 F-score。

# ### 问题 3 - 选择最佳的模型
# 
# *基于你前面做的评价，用一到两段话向 *CharityML* 解释这三个模型中哪一个对于判断被调查者的年收入大于 \$50,000 是最合适的。*             
# **提示：**你的答案应该包括评价指标，预测/训练时间，以及该算法是否适合这里的数据。

# **回答：** 三个模型中，朴素贝叶斯除了具备最短的训练时间外，在训练集和测试的表现，无论是准确率还是F-score都明显逊色于其他两个模型，仅仅比朴素预测器性能好，不难发现在训练集上，决策树表现最好，几乎可以完美预测目标，而在测试集，支持向量机性能与决策树旗鼓相当，甚至更好，相比之下，决策树容易过拟合的缺点暴露出来，支持向量机在训练集和测试集都表现出了较好的鲁棒性，但是随着训练样本的增加，模型在训练时间的开销将急剧增加，在实际庞大人口样本的数据中，可实施性大幅下降，因此，综合比较，决策树模型对于判断被调查者收入大于$50,0000是最合适的。

# ### 问题 4 - 用通俗的话解释模型
# 
# *用一到两段话，向 *CharityML* 用外行也听得懂的话来解释最终模型是如何工作的。你需要解释所选模型的主要特点。例如，这个模型是怎样被训练的，它又是如何做出预测的。避免使用高级的数学或技术术语，不要使用公式或特定的算法名词。*

# **回答： ** 决策树是一种直观易懂的分类算法，它会自动找出决定被调查者收入大于$50,000的最重要的特征，根据特征一步步将推导出调查者是否是我们需要找的人。训练决策树的过程，简单来说，依次根据特征划分训练集，找出分类能力最强的特征，最终形成一个树状结构的决策模型，根据模型，在预测前，我们就能容易理解决策树分类机制，并能协助我们发现影响决策最关键的因素。

# ### 练习：模型调优
# 调节选择的模型的参数。使用网格搜索（GridSearchCV）来至少调整模型的重要参数（至少调整一个），这个参数至少需尝试3个不同的值。你要使用整个训练集来完成这个过程。在接下来的代码单元中，你需要实现以下功能：
# 
# - 导入[`sklearn.model_selection.GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 和 [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - 初始化你选择的分类器，并将其存储在`clf`中。
#  - 设置`random_state` (如果有这个参数)。
# - 创建一个对于这个模型你希望调整参数的字典。
#  - 例如: parameters = {'parameter' : [list of values]}。
#  - **注意：** 如果你的学习器有 `max_features` 参数，请不要调节它！
# - 使用`make_scorer`来创建一个`fbeta_score`评分对象（设置$\beta = 0.5$）。
# - 在分类器clf上用'scorer'作为评价函数运行网格搜索，并将结果存储在grid_obj中。
# - 用训练集（X_train, y_train）训练grid search object,并将结果存储在`grid_fit`中。
# 
# **注意：** 取决于你选择的参数列表，下面实现的代码可能需要花一些时间运行！

# In[85]:


# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO：初始化分类器
clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, min_samples_split=10, random_state=42)

# TODO：创建你希望调节的参数列表
parameters = {'max_depth':[2,4,6,8,10],'min_samples_leaf':[2,4,6,8,10,12,16], 'min_samples_split':[2,4,6,8,10]}

# TODO：创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# TODO：用训练数据拟合网格搜索对象并找到最佳参数
grid_obj = grid_obj.fit(X_train, y_train)

# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
predictions = (clf.fit(X_train, y_train)).predict(X_val)
best_predictions = best_clf.predict(X_val)

# 汇报调优后的模型
print ("best_clf\n------")
print (best_clf)

# 汇报调参前和调参后的分数
print ("\nUnoptimized model\n------")
print ("Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions)))
print ("F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5)))
print ("\nOptimized Model\n------")
print ("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions)))
print ("Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5)))


# ### 问题 5 - 最终模型评估
# 
# _你的最优模型在测试数据上的准确率和 F-score 是多少？这些分数比没有优化的模型好还是差？_
# **注意：**请在下面的表格中填写你的结果，然后在答案框中提供讨论。

# #### 结果:
#  
# | 评价指标         |  未优化的模型        | 优化的模型        |
# | :------------: |  :---------------: | :-------------: | 
# | 准确率          |      0.8581        |   0.8630        |
# | F-score        |      0.7408        |   0.7437        |

# **回答：** 未优化前，模型在验证集的准确率为 0.8581 F-score为0.7408，网格搜索算法在给定参数范围内找出了，评分最高的参数，最终优化的模型在验证集的准确率提高到0.8630，F-score提高到0.7437，模型性能得到改善。

# ----
# ## 特征的重要性
# 
# 在数据上（比如我们这里使用的人口普查的数据）使用监督学习算法的一个重要的任务是决定哪些特征能够提供最强的预测能力。专注于少量的有效特征和标签之间的关系，我们能够更加简单地理解这些现象，这在很多情况下都是十分有用的。在这个项目的情境下这表示我们希望选择一小部分特征，这些特征能够在预测被调查者是否年收入大于\$50,000这个问题上有很强的预测能力。
# 
# 选择一个有 `'feature_importance_'` 属性的scikit学习分类器（例如 AdaBoost，随机森林）。`'feature_importance_'` 属性是对特征的重要性排序的函数。在下一个代码单元中用这个分类器拟合训练集数据并使用这个属性来决定人口普查数据中最重要的5个特征。

# ### 问题 6 - 观察特征相关性
# 
# 当**探索数据**的时候，它显示在这个人口普查数据集中每一条记录我们有十三个可用的特征。             
# _在这十三个记录中，你认为哪五个特征对于预测是最重要的，选择每个特征的理由是什么？你会怎样对他们排序？_

# **回答：**
# 
# - 特征1: capital-gain 资本增益，有资本收益，会投资更有可能获得更多财富
# - 特征2: education_level 受教育程度，学历高更有可能获得更多高薪机会
# - 特征3: occupation 职业，不同行业市场情况不同，会直接影响到收入
# - 特征4: workclass 工作类型，不同工作类型，创造的价值不一样，薪资也会收到影响
# - 特征5: sex 性别，相对而言，男性更容易获得更多工作机会，更容易接受更大挑战，薪资也更高

# ### 练习 - 提取特征重要性
# 
# 选择一个`scikit-learn`中有`feature_importance_`属性的监督学习分类器，这个属性是一个在做预测的时候根据所选择的算法来对特征重要性进行排序的功能。
# 
# 在下面的代码单元中，你将要实现以下功能：
#  - 如果这个模型和你前面使用的三个模型不一样的话从sklearn中导入一个监督学习模型。
#  - 在整个训练集上训练一个监督学习模型。
#  - 使用模型中的 `'feature_importances_'`提取特征的重要性。

# In[87]:


# TODO：导入一个有'feature_importances_'的监督学习模型
from sklearn.ensemble import AdaBoostClassifier

# TODO：在训练集上训练一个监督学习模型
model = AdaBoostClassifier().fit(X_train, y_train)

# TODO： 提取特征重要性
importances = model.feature_importances_

# 绘图
vs.feature_plot(importances, X_train, y_train)


# ### 问题 7 - 提取特征重要性
# 观察上面创建的展示五个用于预测被调查者年收入是否大于\$50,000最相关的特征的可视化图像。
# 
# _这五个特征的权重加起来是否超过了0.5?_<br>
# _这五个特征和你在**问题 6**中讨论的特征比较怎么样？_<br>
# _如果说你的答案和这里的相近，那么这个可视化怎样佐证了你的想法？_<br>
# _如果你的选择不相近，那么为什么你觉得这些特征更加相关？_

# **回答：** 1）超过了2）这五个特征于6中只有capital-gain是相同的 3）不相近，我是根据我国情推测的，这里是美国国情，所以不完全一样是情理之中的

# ### 特征选择
# 
# 如果我们只是用可用特征的一个子集的话模型表现会怎么样？通过使用更少的特征来训练，在评价指标的角度来看我们的期望是训练和预测的时间会更少。从上面的可视化来看，我们可以看到前五个最重要的特征贡献了数据中**所有**特征中超过一半的重要性。这提示我们可以尝试去**减小特征空间**，简化模型需要学习的信息。下面代码单元将使用你前面发现的优化模型，并**只使用五个最重要的特征**在相同的训练集上训练模型。

# In[88]:


# 导入克隆模型的功能
from sklearn.base import clone

# 减小特征空间
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_val_reduced = X_val[X_val.columns.values[(np.argsort(importances)[::-1])[:5]]]

# 在前面的网格搜索的基础上训练一个“最好的”模型
clf_on_reduced = (clone(best_clf)).fit(X_train_reduced, y_train)

# 做一个新的预测
reduced_predictions = clf_on_reduced.predict(X_val_reduced)

# 对于每一个版本的数据汇报最终模型的分数
print ("Final Model trained on full data\n------")
print ("Accuracy on validation data: {:.4f}".format(accuracy_score(y_val, best_predictions)))
print ("F-score on validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5)))
print ("\nFinal Model trained on reduced data\n------")
print ("Accuracy on validation data: {:.4f}".format(accuracy_score(y_val, reduced_predictions)))
print ("F-score on validation data: {:.4f}".format(fbeta_score(y_val, reduced_predictions, beta = 0.5)))


# ### 问题 8 - 特征选择的影响
# 
# *最终模型在只是用五个特征的数据上和使用所有的特征数据上的 F-score 和准确率相比怎么样？*  
# *如果训练时间是一个要考虑的因素，你会考虑使用部分特征的数据作为你的训练集吗？*

# **回答：** 使用删减后的特征会全特征训练的模型，准确率和F-score有所降低，但是差别不大；如果训练时间考虑在内，我会考虑使用部分特征的数据作为训练集，然后随机组合，获得更强分类器，同时能有效降低模型过拟合的情况发生。

# ### 问题 9 - 在测试集上测试你的模型
# 
# 终于到了测试的时候，记住，测试集只能用一次。
# 
# *使用你最有信心的模型，在测试集上测试，计算出准确率和 F-score。*
# *简述你选择这个模型的原因，并分析测试结果*

# In[96]:


#TODO test your model on testing data and report accuracy and F score
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
test_fscore = fbeta_score(y_test, test_predictions, beta=0.5)

print('test_accuracy : {}'.format(test_accuracy))
print('test_fscore : {}'.format(test_fscore))


# 答：选择使用AdaBoostClassifier分类模型作为最终模型在测试集测试，因为Boostging组合算法，通过将多个弱分类器，通过依次训练，将前一个弱分类器分错的数据，增加权重，使下一个弱分类器更多关注分错的数据，充分发挥模型的能力，达到更好的分类效果。

# > **注意：** 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)**把这个 HTML 和这个 iPython notebook 一起做为你的作业提交。
