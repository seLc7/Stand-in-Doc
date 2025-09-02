# Stand-in-Doc
A LLMs-driven framework for generating high-fidelity surrogate documents for cyber deception defense.

Our source code and experiment scripts are publicly available at [https://github.com/seLc7/Stand-in-Doc](https://github.com/seLc7/Stand-in-Doc). The released code is distributed under the MIT License.


### **Data**

We also upload datasets under the `data/` folder. Due to the large size of the data files, we compress them into multi-part archives.

**English Dataset**

- News Domain: We utilize two widely used open-source fake news datasets.  
    - [The Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset): This dataset contains four fields: title, text, subject, and date. We use the real news subset (True.csv) as validation samples, which consists of 21,192 unique news articles. The subject field indicates that 53% of the content belongs to political news, and 47% covers world news.

    - [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification): This dataset consists of 72,134 news articles, among which 35,028 are real news articles.

- Government Reports Domain: We collect 21,428 government report documents from the [Data.gov](https://data.gov/) platform, which provides open access to U.S. government datasets.
    
- Paper Abstract Domain: We use the [econ_paper_abstracts dataset](https://huggingface.co/datasets/onurkeles/econ_paper_abstracts) from HuggingFace, which contains 7,070 research paper abstracts in the field of econometrics collected via the arXiv API.

**Chinese Dataset**

- News Domain: We use the [SmoothNLP](https://github.com/smoothnlp/FinancialDatasets) dataset, which contains 20,000 financial news articles and column information.

- Government Reports Domain: We utilize [GovDoc-CN](https://github.com/RuilinXu/GovDoc-CN), an official document dataset collected from the Chinese State Council Policy Document Database, containing 1,737 documents with a total of 6,816 pages.

- Paper Abstract Domain: We use the [Chinese_Paper_Abstract dataset](https://huggingface.co/datasets/yuyijiong/Chinese_Paper_Abstract), which provides title, full text, and Chinese abstract information. We select 20,000 samples for experiments.

### **Reproducibility**

Our experiments are conducted primarily in a Python environment (Python 3.10.11).

The two main execution programs are:

- **ChatGPT-TiShen**:  
  Implemented in `chatgpt_tishen_mark2.py`.  
  Requires a configuration file (`config.json`), where users must insert their own API key and optionally change the invoked model (our default is ChatGPT-4o).

- **DeepSeek-TiShen**:  
  Implemented in `deepseek_tishen_mark2.py`.  
  Given the model's open-source nature, we recommend local deployment. Our experiments use DeepSeek-R1-70B.

These two distinct programs implement the core pipeline of surrogate document generation from source materials. One leverages the ChatGPT model, and the other employs the DeepSeek model as the underlying large language model (LLM) for content synthesis and prompt refinement.

Both programs support batch processing: they can automatically read original document files in bulk from a specified directory and generate corresponding stand-in documents on a one-to-one basis. Each generated surrogate is saved in a dedicated subfolder, preserving the original document structure while applying context-aware replacements to sensitive content. The tools also integrate iterative prompt refinement logic to ensure the obfuscation quality improves over training epochs. The implementations are modular, allowing users to configure the prompt template, replacement strategy, number of iterations, and target language model. This enables reproducible experiments and flexible adaptation to different datasets or threat models in research and practical applications.

In addition to the surrogate document generation pipeline, we provide two auxiliary scripts to support dataset preparation and evaluation:

- `data_preprocess.py`:  
  This script processes structured source data in `.xlsx` format. It reads each row (e.g., title and content fields) and extracts them into individual `.txt` files. Each file corresponds to a single document and serves as a clean, modular input for surrogate generation. This facilitates scalable and standardized document preparation across various datasets.

- `CI_similarity_comparison.py`:  
  This script is designed to evaluate the concealment quality of generated surrogate documents. It compares each Stand-in document with its original counterpart, focusing on the preservation or obfuscation of critical information entities. The script outputs key evaluation metrics such as entity match rate, field-level similarity, and optionally computes the Critical Information Concealment Rate (CICR) for each instance.

All provided scripts and datasets are structured for plug-and-play reproducibility, enabling researchers to replicate our experiments or adapt the framework to alternative LLMs and threat scenarios. We encourage the community to explore the codebase, extend the surrogate generation pipeline, and evaluate new models or document formats under similar conditions. This supports not only scientific transparency but also practical advancement in the proactive defense of critical information systems.