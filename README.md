![logo](pics/logo.png)

<h1 style="text-align:center; font-weight:bold;">Generative AI Quant: The Future or Folly?</h1>



##  🔍  概览



**Generative AI Quant旨在探究生成式AI Agents在量化交易领域的应用，同时也关注当前对现有大语言模型在量化交易应用中的能力是否过于乐观。**



具体而言，我们设计了四种功能特性各异的模型架构：



1. 我们首先采用一个未引入语言模型代理的传统量化交易程序，它是基于遗传规划方法进行操作的；

2. 在此基础上，我们引入了一个研究代理，负责组织和执行遗传规划任务，并对其进行周期性反思，最终由该研究代理确定最终策略（这被我们称为"语言模型代理-反思属性"模型）；

3. 再进一步，我们引入了多个代理来替代遗传规划环节并处理多模态数据，依然是由代理们确定最终策略（我们称之为"语言模型代理-多模态属性"模型）；

4. 在此基础上，我们又增加了双层的反思模块（"多模态 + 反思"模型）；

5. 最后，我们将未经修改和调整的模型直接应用到其他市场上，观察其效力（"语言模型代理-转移属性"模型）。

   

通过比较这些模型的测试结果，我们可探究诸如多代理化、反思、多模态及转移等语言模型特性是否能够助力量化交易并提高其表现。



## 💫 项目结构

- models - 不同架构的实验模型

- data - 股票特征以及多模态数据

  - graph
  - news
  - price

- data_process - 用于生成各种数据集以及debug

- papers - 研究论文

- results

- tools

  - backtest.py - 回测框架
  - common.py - 常用工具
  - CTA_GEP.py - 普通遗传规划框架
  - agent_GEP.py - 支持agent的遗传规划框架
  - numerical_GEP.py - 遗传规划测试框架
  - evaluate_and_visualizations.py - 结果可视化
  - Indicators.py - 指标计算
  - more_operators.py - 自定义算子
  - multimodal.py - 与多模态相关的工具

  project.env - 环境配置文件



## 🚀 使用方法

首先按序号运行data_process，第一个ipynb生成测试股票数据，第二个ipynb帮助生成一些常见的技术指标作为因子，这些数据都将保存成data文件夹里。进一步的[3], [4], [5]文件用于生成LLM所需要用到的多态数据，包括每一个时间点的价格数据，图片数据与新闻数据。文件[6]以后是对封装函数的测试，以方便进一步的debug。



生成数据并且正确设置project.env后，可以按顺序在models里运行不同架构的实验模型以观察不同结果。



.env环境文件里应该包含至少以下内容

```python
OPENAI_API_KEY = ''

LANGCHAIN_TRACING_V2 = 'true'
LANGCHAIN_API_KEY = ''
LANGCHAIN_PROJECT = ''
```



## 📝 Citation

如果对你有帮助，请以以下格式引用我们：

```
@misc{2024Generative AI Quant,
      title={Generative AI Quant: The Future or Folly?},
      author={},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR}
}
```

