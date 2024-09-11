[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[![CC BY 4.0][cc-by-shield]][cc-by]

Published in September 12th, 2024

Reference: [AWS Certified AI Practitioner - Exam Prep Course](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/19554/exam-prep-standard-course-aws-certified-ai-practitioner-aif-c01)

# AWS AI Practitioner Certification - CheatSheet

This reference guide should provide a structured way for you to organize and review the material needed to prepare for the AWS AI Practitioner Certification. The guide will not replace the available content in [AWS Skillbuilder Exam Preparation Course](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/19554/exam-prep-standard-course-aws-certified-ai-practitioner-aif-c01), but will serve as a content index to help you keep track of all the knowledge needed to successfully pass the exam.

----

# Domain 01 - Fundamentals Of AI And ML

## Basic AI Concepts And Terminology

### Definitions
Artificial Inteligence, or AI, is the field of computer science dedicated to solve cognitive problems commonly associated with human intelligence such as learning, creation, image recognition. The goal with AI is to create self-learning system that derives meaning from data.

#### Machine Learning (ML)
Machine learning is a branch of AI and computer science that focuses on use of data and algorithms to imitate the way humans learn. It gradually improves its accuracy to build computer systems that learn from data. Machine learning models are trained by using large datasets to identify patterns and make predictions. Example: Product recommendation for a customer who's shopping online.

#### Deep Learning 
Deep learning is a type of machine learning model that is inspired by human brains using layers of neural networks to process information. Example: Recognizing human speech and objects and images.


### What AI Can Do?
#### Recognizing Patterns And Make Predictions
An AI model can analyze the patterns in the past and forecast how many customers will be in the store on a given day in the future. Predictions that AI makes are called **inferences**. Note that an inference is basically an educated guess, so the model gives a probabilistic result. 

#### Detect Anomalies
Because AI can recognize patterns in data, it can also detect when there is a deviation from the expected pattern known as an **anomaly**. For example, the number of calls a customer service team receives might vary throughout the day in a predictable manner. When something happens, like the call center application goes offline, AI can detect the drop in calls and notify the IT department.

#### Image Processing
Computer vision applications use AI to process images and video for object identification and facial \recognition, as well as classification, recommendation, monitoring, and detection. For example, a computer vision model can detect scratches on a surface and mark it with a red box around them on the image. In a more advanced application, the model can identify a missing capacitor on a circuit board.

#### Translation
AI can translate text from one language to another without human involvement. It goes beyond simple word-to-word translation. It analyzes all text elements and recognizes how the words influence one another to be able to communicate the meaning of the phrase accurately in the target language.

#### Natural Language Processing (NLP)
Natural language processing, NLP, is what allows machines to understand, interpret, and generate human language in a natural-sounding way. This is the technology that powers Alexa devices and those chatbots that let you book a hotel.

#### Generative AI
Generative AI is the next step in artificial intelligence. Generative AI can have seemingly intelligent conversations and generate original content like stories, images, videos, and even music.

----

## Machine Learning (ML)
Machine Learning (ML) is the science of developing algorithms and statistic models that computer systems use to perform complex tasks without explicit instructions.
- It process large quantities of historical data and identify data patterns.
- It starts with mathematical algorithms that takes data input and generates an output.
- To train the algorithm to produce the expected output, known data is continously provided, which consists of features
- Features can be columns in a table, pixels in a image.
- It tasks is to find correlation betwwen input data features and the known expected output.    
- Adjustments are made to the model by changing internal parameter values until the model reliably produces the expected output.
- A trained model is then able to make accurate predictions and produce output from new data that it hasn't seen during training. This is known as **inference**.

### Data Types
ML models can be trained on different types of data from various sources.

#### Structured Data
- This type of data is stored as rows in a table with columns, which can serve as the features for an ML model.
- Examples: CSV text files, Relational Databases.

#### Semi-Structured Data
- Semi-structured data doesn't completely follow the rules of structured tabular data.
- Unlike data in a table, semi-structured data elements can have different attributes or missing attributes.
- Examples: JSON text file

#### Unstructured Data
- Data that doesn't conform to any specific data model and can't be stored in table format.
- Examples include images, video, and text files, or social media posts.
- Typically stored as objects in an object storage system like Amazon S3.

Time series data is important for training models that need to predict future trends. Each data record is labeled with a timestamp, and stored sequentially. This example shows the performance metrics for microservice, including the used memory, CPU percentage, and transactions per second. A machine learning model could discover the patterns in this data. It could then use it to proactively scale out the infrastructure for the service before load is expected to increase. Depending on the sampling rate, time series data captured for long periods can get quite large and be stored in Amazon S3 for model training.

To create a machine learning model, we need to start with an algorithm which defines the mathematical relationship between outputs and inputs. In this simple example of linear regression, we want to find the best fit for a line to match the input data. Model parameters that are adjusted iteratively during the training process to find the best-fitting model. To determine the best fit, we look for the parameter values that minimize the errors. When the model training is complete, it's ready to begin making inferences.

----

## Inferencing

The training process produces model artifacts, which typically consists of trained parameters, a model definition that describes how to compute inferences and other metadata. The model artifacts, which are normally stored in Amazon S3, are packaged together with inference code to make a deployable model. Inference code is the software that implements the model, by reading the artifacts. 

There are two general options for hosting a model:

- The first is where an endpoint is always available to accept inference requests in real time.
- And the second is where a batch job is performing inference.

### Real-Time Inference

- Ideal for online inferences that have low latency and high throughput requirements.
- Model is deployed on a persistent endpoint to handle a sustained flow of requests.
- Clients send input data to the model and receive back and inference very quickly.
- Compute resources must be available for requests at any time.

### Batch Job

- Suitable for offline processing when large amounts of data are available upfront.
- Also for large number of inferences and it's ok to wait for the results.
- Cost-effective, since the compute resources are shutdown after processing. Therefore a persistent endpoint is not needed.

----

There are several distinct machine learning styles that can be used depending on the expected output and the input type.

### Supervised Learning

- Model trained with pre-labeled data.
- Internal parameters of the algorithm are adjusted during the training process.

> [!NOTE]
> ML inferences are not always accurate, whilst uses probability to generate results. The challenge with supervised learning is in labeling the data. The model might need to be trained on many thousands of pictures of fish before it makes reliable predictions. This involves people who must look at an image and label it.

> [!TIP]
> To address this challenge, Amazon offers a labeling service, Amazon SageMaker Ground Truth. SageMaker Ground Truth can leverage crowdsourcing service called Amazon Mechanical Turk that provides access to a large pool of affordable labor spread across the globe.

### Unsupervised Learning

- Useful for use cases such as pattern recognition, anomaly detection, and automatically grouping data into categories. 
- Does not require data labeling, setup is straightforward.
- Can also be used to clean and process data for further modeling automatically (clustering)

### Reinforcement Learning

- Method that is focused on autonomous decision making by an agent.
- Agent takes actions within an environment to achieve specific goals.
- Actions that an agent takes that move it closer to achieving the goal are rewarded.
- To encourage learning during training, the learning agent must be allowed to sometimes pursue actions that might not result in rewards.
- The model learns through trial and error.
- Does not require labeled input.  

> [!NOTE]
> Both unsupervised and reinforcement learning work without labeled data. Unsupervised learning algorithms receive inputs with no specified outputs during the training process. However, reinforcement learning has a predetermined end goal. While it takes an exploratory approach, the explorations are continuously validated and improved to increase the probability of reaching the end goal.

----

## Troubleshooting Model Performance

### Overfitting

- When a model performs better on training data than it does on new data, it is called **overfitting**.
- It is said that the model does not recognize well.
- The model fits the training data too well, so when it sees something slightly different, it thinks the probability is low.
- Usually, the best way to correct a model that is overfitting is to train it with data that is more diverse.

> [!IMPORTANT]
> Sometimes, if you train your model for too long, it will start to <u>overemphasize</u> unimportant features called **noise**, which is another way of overfitting.

### Underfitting

- The error that occurs when the model cannot determine a meaningful relationship between the input and output data.
- Models give inaccurate results for both the training dataset and new data.
- This can happen if you haven't trained the model long enough or with a large enough dataset.

> [!NOTE]
> Because training for too long can cause overfitting, data scientists try to find the sweet spot for training time where the model doesn't underfit or overfit.

### Bias And Fairness

- Disparities in the performance of a model across different groups.
- The results are skewed in favor of or against an outcome for a particular class.


The quality of a model depends on the underlying data quality and quantity. Also, if a model is showing bias, the weight of features that are introducing noise can be directly adjusted by the data scientists. For example, it could completely remove gender consideration by the model. Fairness constraints, such as age and sex discrimination, should be identified at the beginning before creating a model. Training data should be inspected and evaluated for potential bias, and models need to be continually evaluated by checking their results for fairness.

----

## Deep Learning

Deep learning is a type of machine learning that uses algorithmic structures called neural networks. These are based upon the structure of the human brain. In our brains, brain cells called neurons form a complex network where they send electrical signals to each other to help us process information. In deep learning models, we use software modules called nodes to simulate the behavior of neurons.

Deep neural networks comprise layers of nodes, including an input layer, several hidden layers, and an output layer of nodes. Every node in the neural network autonomously assigns weights to each feature. <u>Information flows through the network in a forward direction from input to output</u>.

During training, the difference between the predicted output and the actual output is then calculated. The weights of the neurons are repeatedly adjusted to minimize the error. Deep learning can excel at tasks like image classification and natural language processing where there is a need to identify the complex relationship between data objects.

> [!NOTE]
>
> The concept of deep learning with neural networks has existed for some time. However, the required computing power wasn't visible for most businesses to obtain until the arrival of low-cost cloud computing. Because anyone can now readily use powerful computing resources in the cloud, neural networks have become the standard algorithmic approach to computer vision.

A big advantage of deep learning models for computer vision is that they don't need the relevant features given to them. They can identify patterns in images and extract the important features on their own. However, we might need to give a deep learning model millions of pictures of fish before it can accurately detect and label a fish in an image. And the compute infrastructure to train a deep learning model repeatedly on such a large dataset is going to cost more than the traditional approach.

> [!IMPORTANT]
>
> The decision to use traditional machine learning or deep learning depends on the type of data you need to process. Traditional machine learning algorithms will generally perform well and be efficient when it comes to identifying patterns from structured data and labeled data. Examples include classification and recommendation systems.
>
> For instance, a cell phone company can use ML to predict when a customer will change carriers based on previous customer churn data. On the other hand, deep learning solutions are more suitable for unstructured data like images, videos, and text.
>
> Tasks for deep learning include image classification and natural language processing, where the is a need to identify the complex relationships between pixels and words. For example, a deep learning solution can analyze social media mentions or product feedback to determine user sentiment.
>
> Both types of machine learning use statistical algorithms, but only deep learning uses neural networks to simulate human intelligence. As we've seen, deep learning models self-learn patterns, so they don't require as much work on selecting and extracting features.
>
> However, their infrastructure costs are significantly higher. 
----
### Generative AI
Generative AI is accomplished by using deep learning models that are pre-trained on extremely large datasets containing strings of text or, in AI terms, sequences. They use transformer neural networks, which change an input sequence, in Gen AI known as prompt, into an output sequence, which is the response to your prompt. Neural networks process the elements of a sequence sequentially one word at a time. Transformers process the sequence in parallel, which speeds up the training and allows much bigger datasets to be used.
Large language models contain many billions of features, which captures a wide range of human knowledge. With all this training, large models are very flexible in the tasks they can perform. They outperform other ML approaches to natural language processing.

They excel at understanding human language so they can read long articles and summarize them. They are also great at generating text that's similar to the way a human would. As a result, they are good at language translation and even writing original stories, letters, articles, and poetry.

They even know computer programming languages and can write code for software developers.

> [!TIP]
>
> If you want to try Amazon Bedrock for yourself for free, you can build your own AI app at partyrock.aws.



## Practical Use Cases for AI

### Consider AI/ML For...

- Increasing business efficiency
  - Automating time-consuming tasks
  - Streamlining self-service processes
- Solving complex problems
  - Fraud detection
  - Waste reduction
- Making better decisions
  - Forecasting future actions
  - Getting answers fast

### Consider AI/ML Alternatives When...

- Costs outweighs benefits
  - Perform a cost-benefit analysis
- Models cannot meet the interpretability requirements
  - Consider a rules-based system (deterministic)
- Systems must be deterministics rather than probabilistics

### Machine Learning Problem Types

- If your dataset consists of features or attributes as inputs with <u>labeled target values</u> as outputs, then you have a **supervised learning** problem. In this type of problem, you train your model with data containing known inputs and outputs.

  - If your target values are **categorical**, for example, one or more discrete values, then you have a **classification** problem. 

  - If these target values you're trying to predict are mathematically **continuous**, then you have a **regression** problem. 

- If your dataset consists of features or attributes as inputs that <u>do not contain labels or target values</u>, then you have an **unsupervised learning** problem. In this type of problem, the output must be predicted based on the pattern discovered in the input data. The goal in unsupervised learning problems is to discover patterns, such as groupings, within the data.
  - When your data needs to be **separated** **into** discrete **groups**, you have a **clustering** problem. 
  - If you are seeking to **spot outliers** in your data, then you have an **anomaly detection** problem.

#### Supervised Learning - Classification

- Binary

  - Probability that something is true

    Binary classification assigns an input to one of two predefined and mutually exclusive classes based on its attributes. A medical diagnosis for whether an individual has a disease or not, based on the results of diagnostic tests is an example of binary classification.

- Multiclass

  - Prediction of a class

    Multiclass classification assigns an input to one of several classes based on the input attributes. An example is the prediction of the topic most relevant to a tax document. A document might be classified as being about religion, politics, or finance, or as about one of several other predefined topic classes.

#### Supervised Regression

When your target values are mathematically continuous, then you have a regression problem. Regression estimates the value of dependent target variable based on one or more other variables, or attributes that are correlated with it. Linear regression is when there is a direct linear relationship between the inputs and output.

- Simple linear regression

  - Single independent variable

    Simple linear regression uses a single independent variable, such as weight, to predict someone's height

- Multiple linear regression

  - Multiple independent variables

    Multiple independent variables, such as weight and age, then we have a multiple linear regression problem. Another example is the prediction of house prices using features like the number of bathrooms, bedrooms, the square footage of the house and garden. Regression analysis can create a model that takes one or more of these features as an input to predict the price of a house.

- Logistic regression

  - Prediction between 0 and 1

  - Single or multiple independent variables

    Logistic regression is used to measure the probability of an event occurring. The prediction is a value between zero and one, where zero indicates an event that is unlikely to happen, and one indicates a maximum likelihood that it will happen. Logistic equations use logarithmic functions to compute the regression line. It can use one or multiple independent variables. An example is predicting if a person will get heart disease based on body mass index, BMI, smoking status, and genetic predisposition. Another example would be predicting whether a financial transaction is fraud when there are transactions labeled as fraud and not fraud used for training.

> [!TIP]
>
> Both logistic regression and linear regression require a significant amount of labeled data for the models to become accurate in predictions. 

#### Cluster Analysis

Cluster analysis is a class of techniques that are used to classify data objects into groups, called clusters. It attempts to find discrete groupings within data. Members of a group are similar as possible to one another, and as different as possible from members of other groups. You define the features or attributes that you want the algorithm to use to determine similarity. Then you select a distance function to measure similarity and specify the number of clusters, or groups, you want for the analysis.

An example of clustering is segmenting customers into groups by purchase history or clickstream activity.

#### Anomaly Detection

Anomaly detection is the identification of rare items, events, or observations in the data, which raise suspicions, because they differ significantly from the rest of the data. The identification of anomalous items can be used, for example, to detect failed sensors or medical errors.

### AWS Services

AWS offers several pre-trained AI services that are accessible through APIs. Before embarking on the effort and cost to build a custom model, you should investigate whether an existing service for your use case already exists. 

#### Computer Vision

##### Amazon Reckognition

Amazon Rekognition is a pre-trained deep learning service for computer vision. It meets the needs of several common computer vision use cases without requiring customers to train their own models. It works with both images and videos, including streaming videos.

- Facial comparison and analysis

  It can be used to verify someone's identity by comparing their image with a reference image, like an employee's badge or a driver's license. One of the things you can do with Amazon Rekognition is give it a collection of labeled images of faces. For example, a company's employees. It'll automatically recognize and find them in images or stored in streaming videos. 

- Object detection and labeling

  Amazon Rekognition can detect and label objects, which can be used to make an image or video library searchable. It can also be used in security system to detect and identify objects in real-time streaming video and send out alerts. You can get Amazon Rekognition to recognize custom or proprietary objects by giving it some labeled images to learn from. 

  - Custom labeling

    You can get Amazon Rekognition to recognize custom or proprietary objects by giving it some labeled images to learn from. 

- Text detection

- Content moderation

  Typically, companies that allow users to upload content to their application need to employ people to screen content before letting it get published. Amazon Rekognition can detect and filter explicit, inappropriate, and violent content in images and videos, and also flag content that should be reviewed by humans.

#### Text And Document Analysis

##### Amazon Textract

More than just optical character recognition, Amazon Textract extracts text, handwriting, forms, and tabular data from scanned documents. 

##### Amazon Comprehend

Amazon Comprehend is a natural language processing service that helps discover insights and relationships in text. A common use case to classify the sentiment of customer feedback. For example, AWS uses Amazon Comprehend to analyze the comments left on Certification exams. 

> [!NOTE]
>
> Frequently, Amazon Textract and Amazon Comprehend are used together. Content extracted by Amazon Textract can be given to Amazon Comprehend for sentiment analysis. A common ML use case is detecting personal identifiable information, PII, in text. If you were collecting data for training an ML model to detect spam emails, you would want to be able to find PII and remove it from training data. Amazon Comprehend is pre-trained to find PII. 

#### Language AI

##### Amazon Lex

Amazon Lex helps build voice and text interfaces to engage with customers. It uses the same technology that powers Amazon Alexa devices. Some common use cases are customer service chatbots and interactive voice response systems, that route calls to the proper agent in a call center.

##### Amazon Transcribe

Amazon Transcribe is an automatic speech recognition service that supports over 100 languages. Transcribe is designed to process live and recorded audio or video input to provide high quality transcriptions for search and analysis. A common use case is to caption streaming audio in real time.

##### Amazon Polly

Amazon Polly turns text into natural-sounding speech in dozens of languages. It uses deep learning technologies to synthesize human speech. Common use cases include converting articles to speech and prompting callers in interactive voice response systems. The ability of AI to create natural-sounding voices is enabling companies to increase engagement with their products and be more accessible to visually impaired customers.

#### Customer Experience

##### Amazon Kendra

Amazon Kendra uses machine learning to perform an intelligent search of enterprise systems to quickly find content. It uses natural language processing to understand questions like, "How do I connect my Echo Plus to my network?" It responds with results based on an intelligent understanding of the question.

##### Amazon Personalize

Amazon Personalize allows businesses to automatically generate personalized recommendations for their customers in industries such as retail, media and, entertainment. For example, an ecommerce app can include a section called "you might also like" with personalized product recommendations to customers who will likely be interested in those products. Businesses can also use Amazon Personalize to run more effective marketing campaigns by segmenting customers according to their preferences.

##### Amazon Translate

Amazon Translate fluently translates text between 75 different languages. It is built on a neural network that considers the entire context of the source sentence and the translation it has generated so far. It uses this information to create more accurate and fluent translations. One use case is real-time translation in an online chat application.

#### Business Metrics

##### Amazon Forecast

Amazon Forecast is an AI service for time series forecasting. By providing Amazon Forecast with historical time series data, you can predict future points in the series. Time series forecasting is useful in multiple domains, including retail, financial planning, supply chain, and healthcare. For example, you can use it to project sales and manage inventory levels.

##### Amazon Fraud Detector

Amazon Fraud Detector helps to identify potentially fraudulent online activities such as online payment fraud and creation of fake accounts. It features pre-trained data models to detect fraud in online transactions, product reviews, checkout and payments, new accounts, and account takeovers.

#### Generative AI

##### Amazon Bedrock

- Foundation Models (FMs)
- Can customize with training data or Retrieval Augmented Generation

Amazon Bedrock is a fully managed service to build generative AI applications on AWS. Amazon Bedrock lets you choose from high performing foundation models trained by Amazon, Meta, and leading AI startups. You can customize a foundation model by providing your own training data or creating a knowledge base for the model to query.

> [!TIP]
>
> When a generative AI model calls an external knowledge system to retrieve information outside its training data, this is called **Retrieval Augmented Generation** or RAG for short.

#### Model Development

> [!TIP]
>
> Use the Amazon SageMaker family of services when you need more customized machine learning models or workflows that go beyond the prebuilt functionalities offered by the core AI services.

##### Amazon SageMaker

- Fully managed ML service
- Data preparation and labeling
- Model training and evaluation
- Model deployment and monitoring
- Pre-trained models

Amazon SageMaker provides machine learning capabilities for data scientists and developers to prepare, build, train, and deploy high-quality ML models efficiently. It comprises several services that are optimized for building and training custom machine learning models. These include data preparation and labeling, large-scale parallel training on multiple instances or GPU clusters, model deployment, and real-time inference endpoints To accelerate the development process, SageMaker offers pre-trained models that you can use as a starting point and reduce the resources needed for data preparation and model training.



## ML Development Lifecycle

A machine learning pipeline is a series of interconnected steps that start with a business goal and finish with operating a deployed ML model. It starts with defining the problem, collecting and preparing training data, training the model, deploying, and finally, monitoring it. Some of these steps are an iterative process, and this is repeated until certain objectives are accomplished.

<img src="/Users/akzsato/Library/Application Support/typora-user-images/image-20240907113105129.png" alt="image-20240907113105129" />

Machine learning models are dynamic by design. They are re-trained with new data, continually evaluated against performance and business metrics, monitoring for drifts and bias, and adjusted or rebuilt as needed. As a result, many prefer to think of the ML pipeline as a lifecycle, where parts, or even all of it, are repeated even after the model is deployed.

<img src="/Users/akzsato/Library/Application Support/typora-user-images/image-20240907113327181.png" alt="image-20240907113327181" style="zoom:50%;" />

### Defining The Problem

- Identify the business goal

  The development of an ML model should always begin with identifying the business goal. An organization considering ML should have a clear idea of the problem to be solved and the business value to be gained.

  - Success criteria

    More than just an idea, you must be able to measure business value against specific business objectives and success criteria. Without clear success criteria, you won't be able to evaluate the model or even determine if ML is the best solution.

  - Align stakeholders

    You'll need to align stakeholders to gain consensus on what the goal of the project is. After you determine your criteria for success, evaluate your organization's ability to move toward the target.

- Frame the ML problem

  The target should be achievable and provide a clear path to production. Determine if ML is the appropriate approach for delivering your business goal. Evaluate all the options that you have available for achieving the goal. Determine how accurate the resulting outcomes would be while considering the cost and scalability of each approach. Ensure that enough relevant high-quality training data is available to the algorithm. 

  - Defining the ML tasks: inputs, outputs and metrics

    Carefully evaluate the data to make sure that the correct data sources are available and accessible. Formulate the ML question in terms of input, desired outputs, and the performance metric to be optimized.

  - Feasibility

    With the ML problem in mind, investigate all available options.

  - Starting with the simplest models options

    Start with the simplest solution before determining that more complexity is required to meet the business objectives.

  - Cost-benefit analysis

    Remember to perform a cost-benefit analysis to see if the project should move to the next phase.

    

  > [!IMPORTANT]
  >
  > AWS has introduced a number of AI services to democratize ML and make it accessible to anyone. They have identified many common use cases and developed easy, consumable, and fully trained ML models that are fully hosted by them. Because these services are pay as you go, it makes sense to evaluate them to see if they can meet the business goals. Many of these services allow you to customize their outputs.
  >
  > For example, with <u>Amazon Comprehend</u>, you can create a custom classifier that uses your own categories by supplying it with your training data. If a hosted service doesn't achieve the objectives, the next consideration should be building your own model by starting with an existing one.
  >
  > For example, for generative AI use cases, <u>Amazon Bedrock</u> lets you start with a fully trained foundation model. You can fine-tune this model with your own data using transfer learning.
  >
  > For other use cases, <u>Amazon SageMaker</u> has a number of open source pre-trained models to jumpstart your model development. The most difficult and costly approach is to train your own model from scratch. As we will see in later sections, this is not only the most technically challenging, but also requires the most responsibility for security and compliance.
  >
  > <u>SageMaker JumpStart</u> provides pre-trained AI foundation models and task-specific models for computer vision and natural language processing problem types. These are pre-trained on large public datasets. You have the option of fine-tuning the model with incremental training using your own dataset. This is a process known as **transfer learning**. Using a pre-trained model is a large savings in cost and development time over creating a custom model from scratch. 



### Processing Data

- Collecting training data

  - Data sources

    Start by identifying the data needed and determining the options for collecting the data. You'll need to know what training data you will need to develop your model and where it is generated and stored. To collect the data, you'll need to know if it's streaming data or whether you can load it in batch process. 

  - Data ingestion - ETL

    You'll need to configure a process known as extract, transform, and load, ETL, to collect the data from possibly multiple sources and store it in a centralized repository. Remember that models should be re-trained frequently with new data, so your process needs to be repeatable.

  - Labels

    You will need to know if the data is labeled or how you will be able to label it. This can be one of the longest parts of the process because accurately labeled data likely does not already exist.

- Pre-process data

  Data preparation includes data pre-processing and feature engineering.

  - Exploratory data analysis

    Exploratory data analysis, EDA, with visualization tools can help to quickly gain a deeper understanding of data. You can use data wrangling tools for interactive data analysis and to prepare data for model building.

  - Clean

    Data with missing or anomalous values might need to be filtered out or repaired. PII data should be masked or removed. After reprocessing your data, you are almost ready to start training.

  - Split - train, validate test

    But first, you need to decide how best to split up your data. Typically, you will need to create three datasets from the available data. A common recommendation is that about 80% of the data should be used for training the model, 10% should be set aside for model evaluation, and 10% for performing the final test before deploying the model to production.

- Feature engineering

  Finally, you need to determine which characteristics of the dataset should be used as features to train the model. This is the subset that is relevant and contributes to minimizing the error rate of a trained model.

  - Select features

    You should reduce the features in your training data to only those that are needed for inference. Features can be combined to further reduce the number of features. Reducing the number of features reduces the amount of memory and computing power required for training.

  > [!IMPORTANT]
  >
  > Now let's look at some of the many AWS services available for data ingestion and preparation.
  >
  > <u>AWS Glue</u> is a fully managed ETL service. You can create and run an ETL job with a few clicks in the AWS Management Console. You simply point AWS Glue to your data stored on AWS. Then AWS Glue discovers your data and stores the associated metadata, the table definition and schema, in the <u>AWS Glue Data Catalog</u>. After it's cataloged, your data is immediately searchable, queryable, and available for ETL. AWS Glue generates the code to execute your data transformations and data loading processes. In addition to defining your own data transformation, AWS Glue has built-in transformations for things like dropping duplicate records, filling in missing values, and splitting your dataset. AWS Glue can extract, transform, and load data from a large variety of data stores. These include relational databases, data warehouses, and other cloud, or even streaming services, such as <u>Amazon Managed Streaming for Apache Kafka</u>, or Amazon MSK, and <u>Amazon Kinesis</u>.
  >
  > The AWS Glue Data Catalog contains references to data that is used as sources and targets of your ETL jobs in AWS Glue. The AWS Glue Data Catalog tables include an index to the location, schema, and runtime metrics of your data. You use the information in the Data Catalog to create and monitor your ETL jobs. Typically, you run a crawler to take inventory of the data in your data stores, but you can also enter the information in the tables manually. AWS Glue can crawl your data sources and automatically determine the data schema by using classifiers. It writes the schema to tables in the Data Catalog. It's important to understand that the source data itself is not written to the data. Only metadata, such as the location and schema, are stored in the Data Catalog.
  >
  > The AWS Glue ETL jobs use this information to collect, transform, and store the data in the target data store, which is typically an S3 bucket. AWS Glue DataBrew is a visual data preparation tool that enables users to clean and normalize data without writing any code. You can interactively discover, visualize, clean, and transform raw data. DataBrew makes smart suggestions to help you identify data quality issues that can be difficult to find and time-consuming to fix. You can save transformation steps in a recipe, which you can update or reuse later with other datasets and deploy on a continuing basis.
  >
  > DataBrew provides more than 250 built-in transformations, with a visual point-and-click interface for creating and managing data transformation jobs. These include removing nulls, replacing missing values, fixing schema inconsistencies, creating column-based functions and more. You can also use DataBrew to evaluate the quality of your data by defining rule sets and running profiling jobs.
  >
  > To train a supervised machine learning model, you need a large, high-quality labeled dataset. <u>SageMaker Ground Truth</u> helps you build high-quality training datasets for your machine learning models. SageMaker Ground Truth active learning uses machine learning model to label your training data. It will automatically label data that it can label, and the rest is given to a human workforce. You can use human workers from <u>Amazon Mechanical Turk</u>, a workforce with over 500,000 independent contractors. Or use an internal private workforce that uses your own employees or contractors. 
  >
  > You can use <u>Amazon SageMaker Canvas</u> to prepare, featurize, and analyze your data. With Amazon SageMaker Canvas, you can simplify the feature engineering process by using a single visual interface.
  >
  > Using the SageMaker Data Wrangler data selection tool, you can choose the raw data that you want from various data sources and import it with a single click. SageMaker Canvas contains over 300 built-in transformations so that you can quickly normalize, transform, and combine features without having to write any code.
  >
  > <u>Amazon SageMaker Feature Store</u> is a centralized store for features and associated metadata, so features can be easily discovered and reused. Feature Store makes it easy to create, share, and manage features for ML development. Feature Store accelerates this process by reducing repetitive data processing and curation work required to convert raw data into features for training an ML algorithm. You can create workflow pipelines that convert raw data into features and add them to feature groups. 



### Teach The Model: Train, Tune, And Evaluate

The next phase of the pipeline is where we get to train, tune, and evaluate the model. In this phase, we are teaching the model, through an iterative process of training, tuning, and evaluating.

- Train

  During training, the machine learning algorithm updates a set of numbers, known as parameters or weights. The goal is to update the parameters in the model in such a way that the inference matches the expected output. This can't be done in one iteration, because the algorithm has not learned yet. It has no knowledge of how changing weights will shift the output closer toward the expected value. Therefore, it watches the weights and outputs from previous iterations, and shifts the weights to a direction that lowers the error in generated output. This iterative process stops either when a defined number of iterations have been run, or when the change in error is below a target value.

  - Iteractive process
  - Tune parameters

- Run experiments

  There are usually multiple algorithms to consider for a model. The best practice is to run many training jobs in parallel, by using different algorithms and settings. This is known as running experiments, which helps you land on the best-performing solution.

  - Tune hyperparameters

    Each algorithm has a set of external parameters that affect its performance, known as hyperparameters. These are set by the data scientists before training the model. These include adjusting things like how many neural layers and nodes there will be in a deep learning model. The optimal values for the hyperparameters can only be determined by running multiple experiments with different settings.

    > [!IMPORTANT]
    >
    > To train your model using <u>SageMaker</u>, you create a training job which runs your training code on a fleet of ML compute instances managed by SageMaker.
    >
    > You specify the algorithm by giving SageMaker the path to a Docker container image that contains the training algorithm. In the <u>Amazon Elastic Container Registry</u>, Amazon ECR, you can specify the location of SageMaker provided algorithms and deep learning containers, or the location of your custom container, containing a custom algorithm.
    >
    > You also need to set the hyperparameters required by the algorithm. After you create the training job, SageMaker launches the ML compute instances, and uses the training code and the training dataset to train the model. It saves the resulting model artifacts and other outputs in the S3 bucket you specified for that purpose. You need to experiment with multiple combinations of data, algorithms, and parameters, all while observing the impact of incremental changes on model accuracy. This iterative experimentation can result in thousands of model training runs and model versions.
    >
    > <u>Amazon SageMaker experiments</u> is a capability of Amazon SageMaker that lets you create, manage, analyze, and compare your machine learning experiments. An **experiment is a group of training runs, each with different inputs, parameters, and configurations**. It features a visual interface to browse your active and past experiments, compare runs on key performance metrics, and identify the best-performing models.
    >
    > <u>Amazon SageMaker automatic model tuning</u>, AMT, also known as hyperparameter tuning, finds the best version of a model, by running many training jobs on your dataset. To do this, AMT uses the algorithm and ranges of hyperparameters that you specify. It then chooses the hyperparameter values that create a model that performs it best, as measured by a metric that you choose. For example, suppose that you are tuning a binary classification model. You can have automatic model tuning find the combination of hyperparameters that maximizes a metric known as the area under the curve. To use automatic model tuning, you can figure a tuning job that runs several training jobs inside a loop. You specify completion criteria as the number of jobs that are no longer improving the metric. The job will run until the completion criteria are satisfied.

    > [!TIP]
    >
    > **Amazon SageMaker training**
    >
    > - Training jobs
    >   - Training data location
    >   - ML compute instances
    >   - Training image
    >   - Hyperparameters
    >   - S3 output buckets
    >
    > **Amazon Sagemaker experiments**
    >
    > - Tracks inputs, parameter, configurations as separate runs
    > - Compares runs against performance metrics
    > - Visual interface
    >
    > **Amazon SageMaker automatic model tunning**
    >
    > - Automatically adjusts hyperparameters
    > - Uses specified objective metrics
    > - Tuning jobs
    >   - Runs training jobs inside a loop
    >   - Satisfies completion criteria

- Model Evaluation

  - Metrics targets

### Deploy The Model

Once you have a fully trained, tuned, and evaluated model, you need to make it available for use. The first decision is whether you need batch or real-time inferencing or something in between.

- Considerations:

  - Batch inference

    When you need a large number of inferences, and it's okay to wait for the results. This is the most cost-effective approach because cloud computing resources are only running once per day.

  - Real-time inference

     With real-time inference, you deploy your model so that it can respond to requests immediately.

  - Self-managed

    > [!TIP]
    >
    > Clients interact with your model by using a REST application programming interface, API. An API is a set of actions that are made available over an HTTP connection. For example, a web application can send a POST request containing input data and your endpoint, which will pass the request to a compute resource that is running the model. The resulting model output is sent back to the client in the response to the request. In this example, Amazon API Gateway can serve as the interface with the clients and forward requests to an AWS Lambda function, which is running the model.

    - AWS Batch

    - Amazon ECS

    - Amazon EKS

    - Amazon EC2

    - AWS Lambda

      Depending on the service these options will require you to configure and manage the inference endpoint, which might also include managing updates, patches, scalability, network routing, and security.

  - Hosted

    For reduced operational overhead you can choose to host your model with Amazon SageMaker. Amazon SageMaker can automatically deploy your model on hosted endpoints that it fully manages on your behalf.

    - SageMaker inference

      To use SageMaker inferencing, you just point SageMaker to your model artifacts in an S3 bucket and a Docker container image in Amazon ECR. You select which inference option such as batch, asynchronous, serverless, or real time, and SageMaker creates the endpoint and installs your model code. For real-time, asynchronous, and batch inference, SageMaker runs the model on EC2 ML instances, which can be inside an auto scaling group. You select the number and instance type of the ML instances that you want to use. There is also an inference recommender tool within SageMaker that can test out different configuration options with your model, so you can pick the best one.

      For the serverless inference option, SageMaker runs your code on Lambda functions. When you create an endpoint or endpoint configuration, you must choose an inference option. Amazon SageMaker supports four option types. The best choice depends on the business requirements of your ML inference workload. These endpoints are fully managed and support auto scaling.

      - Batch transform

        Batch transform provides offline inference for large datasets. It's useful when running inference if a persistent endpoint is not required and you can wait for the results. It can support large datasets that are gigabytes in size.

      - Asynchronous

        Asynchronous inference is ideal when you want to queue request and have large payloads with processing times. SageMaker will scale your endpoint down to zero so that you aren't charged for periods without requests.

      - Serverless

        Serverless inference can be used to serve model inference requests in real time without directly provisioning compute instances, or configuring scaling policies to handle traffic variations. Because it uses Lambda, you only pay when functions are running or pre-provisioned, so it is a good choice if your model has periods without requests.

      - Real-time

        Real-time inference is ideal for inference workloads where you need real-time interactive responses from your model. Use real-time inference for a persistent and fully managed endpoint REST API that can handle sustained traffic backed by the instance type of your choice. The ML instances remain available to receive requests and return a response in real time.

### Monitor The Model

No matter how great your model performs initially, model performance could degrade over time for reasons such as data quality, model quality, and model bias. The final stage of the ML pipeline is to monitor your model.

- Continuously monitor the quality of ML model in real-time

  The model monitoring system must capture data, compare the data to the training set, define rules to detect issues, and send alerts.

- Identify the right time and frequency to retrain and update the model

  For most ML models, a simple scheduled approach for re-training daily, weekly, or monthly is usually enough.

- Configure alerts to notify and initiate actions if any drift in model performance is observed

  This process repeats on a defined schedule when initiated by an event or when initiated by human intervention. The monitoring system should detect data and concept drifts, initiate an alert, and send it to an alarm manager system, which could automatically start a re-training cycle

Data drift is when there are significant changes to the data distribution compared to the data used for training. Concept drift is when the properties of the target variables change. Any kind of drift results in model performance degradation.

#### Amazon SageMaker Model Monitor

Amazon SageMaker Model Monitor, which is a capability of Amazon SageMaker, monitors models in production and detects errors so you can take remedial actions. 

You define a monitoring schedule that collects data from your endpoints and detects changes against the baseline. It analyzes the data based upon built-in rules or rules that you define. You can view the results in Amazon SageMaker Studio and see which rules were violated. The results are also sent to Amazon CloudWatch, where you can configure alarms to take remedial actions, such as starting a re-training process.

#### MLOps

Automation is an important part of implementing and operating repeatable and reliable business processes. So let's look at how we can use automation in our ML pipelines. MLOps is about using these established best practices of software engineering and applying them to machine learning model development. It's about automating manual tasks, testing, and evaluating code before release, and responding automatically to incidents. 

MLOps can streamline model delivery across the machine learning development lifecycle. Because the cloud uses API based services, everything is treated as software. This includes the infrastructure used in ML pipelines. The entire infrastructure can be described in software and deployed and redeployed in repeatable fashion. This lets data scientists quickly spin up the infrastructure needed to build and test a model so they can run experiments and make continual improvements.

Like DevOps, version control is critical for tracking lineage and being able to inspect a past configuration. With MLOps, everything gets versioned, including the training data. Other key MLOps principles are monitoring deployments to detect potential issues and automating re-training because of issues or data and code changes. One of the benefits of MLOps is productivity, automation and providing self-service environments and infrastructure let data engineers and data scientists move forward.

Another benefit is repeatability. Automating all the steps in the ML lifecycle helps ensure a repeatable process, including how the model is trained, evaluated, version, and deployed. This also improves reliability because it provides the ability to deploy not only quickly, but with increased quality and consistency. For compliance, MLOps can improve auditability by versioning all inputs and outputs from data science experiments to source data to trained models. This means that we can demonstrate exactly how the model was built and where it was deployed.

The final benefit is improvements to data and model quality. MLOps lets us enforce policies that guard against model bias and track changes to data statistical properties, and model quality over time.

**Amazon SageMaker Pipelines** offers the ability to orchestrate SageMaker jobs and author reproducible ML pipelines. SageMaker Pipelines can deploy custom built models for inference in real time with low latency, run offline inferences with batch transform and track lineage of artifacts.

They can institute sound operational practices in deploying and monitoring production workflows, deploying model artifacts, and tracking artifact lineage through a simple interface. You can create a pipeline using the SageMaker SDK for Python or define the pipeline using JSON. The pipeline can contain all the steps to build and deploy a model, and can also include conditional branches based on the output of a previous step. Pipelines can be viewed in SageMaker Studio. 

It's important to mention a few other services for MLOps:

##### AWS CodeCommit (deprecated)

Repositories are where you keep versions of your code and models. AWS CodeCommit is a source code repository that you can use for storing your inference code. It is comparable to GitHub, a third-party source code repository.

##### SageMaker Feature Store

SageMaker Feature Store is a repository for the feature definitions of your training data.

##### SageMaker Model Registry

SageMaker Model Registry is a centralized repository for your trained models and history. 

##### AWS Step Functions

AWS Step Functions, which lets you define a workflow with a visual drag-and-drop interface. It gives you the ability to build serverless workflows that integrate various AWS services and custom application logic.

##### Amazon Managed Workflows for Apache Airflow

Apache Airflow is an open source tool used to programmatically author, schedule, and monitor sequences of processes and tasks referred to as workflows. With Amazon Managed Workflows for Apache Airflow, you can use Apache Airflow and Python to create workflows without having to manage the underlying infrastructure for scalability, availability, and security. 

#### Confusion Matrix

A confusion matrix is used to summarize the performance of a classification model when it's evaluated against task data. The simplest way would be a binary classification model where the output is a simple binary result, yes or a no, positive or a negative. A confusion matrix is a table with actual data typically across the top and the predicted values on the left. The confusion matrix shows the number of true and false positives and negatives. 

##### Accuracy

One metric that is sometimes used to judge a model's performance is accuracy, which is simply the percentage of correct predictions. Accuracy measures how close the predicted class values are to the actual values. Values for accuracy metrics vary between zero and one. A value of one indicates perfect accuracy and zero indicates complete inaccuracy. The formula for accuracy is the number of true positives plus true negatives divided by the total number of predictions.

> [!IMPORTANT]
>
> Though accuracy is understandable, it is not a good metric when the dataset is imbalanced.

##### Precision

Precision measures how well an algorithm predicts true positives out of all the positives that it identifies. The formula is the number of true positives divided by the number of true positives, plus the number of false positives. This is a good quality metric to use when your goal is to minimize the number of false positives. Precision measures how well an algorithm predicts true positives out of all the positives that it identifies.

##### Recall

If we want to minimize the false negatives, then we can use a metric known as recall. For example, we want to make sure that we don't miss if someone has a disease and we say they don't. The formula is the number of true positives divided by the number of true positives plus the number of false negatives. Recall is also known as sensitivity or the true positive rate.

> [!IMPORTANT]
>
> There is a tradeoff between precision and recall because you can't optimize a model for both.

##### F1 Score

However, if recall and precision are both important to us, the F1 score balances precision and recall by combining them in a single metric.

##### False Positive Rate (FPR)

Another metric we can calculate from our confusion matrix is the false positive rate, which is the false positives divided by the sum of the false positives and true negatives.

##### Specificity, or True Negative Rate (TNR)

Closely related to the false positive rate is the true negative rate, which is the ratio of the true negatives to the sum of the false  positives and true negatives.

##### Receiver Operating Characteristics (ROC)

<img src="/Users/akzsato/Library/Application Support/typora-user-images/image-20240908095121981.png" alt="image-20240908095121981" style="zoom:50%;" />

The area under the curve, also known as AUC metric, is used to compare  and evaluate binary classification by algorithms that return  probabilities, such as logistic regression. To map the probabilities  into discrete predictions such as true or false, these are compared  against a threshold value. A threshold is a value that the model uses to make a decision between the two possible classes. It can converts the  probability of a sample being part of a class into a binary decision. 

The true positive rate is plotted against the false positive rate for  increasing threshold values. The threshold values are represented by the red dashed line in the graph. The relevant curve is called the receiver operating characteristic curve. You can see that increasing the  threshold results in fewer false positives, but more false negatives.  AUC is the area under this receiver operating characteristic curve. AUC  provides an aggregated measure of the model performance across the full  range of thresholds. AUC scores vary between zero and one. A score of  one indicates perfect accuracy and a score of one half, or 0.5,  indicates that the prediction is no better than a random classifier.

#### Regression Model Errors

Recall that in linear regression, we're fitting a line to the points in a dataset. The distance between the line and the actual values is the  error.

- Mean Squared Error (MSE)

  A metric that we can use to evaluate a linear regression model is called the mean squared error, MSE. To compute it, we take the difference  between the prediction and actual value, square the difference, and then compute the average of all square differences. MSE values are always  positive. The better a model is at predicting the actual values, the  smaller the MSE value is.

  - Average of the square of the errors

- Root Mean Squared Error (RMSE)

  - Square root of MSE

    Another metric that is commonly used is the root mean squared error,  which is the square root of the mean squared error. The advantage of  using this square root of the MSE is that the units match the dependent  variable. For example, if the height is measured in inches, then the  MSE will be in square inches, but the RMSE is in inches, so the RMSE is  easier for us to interpret. 

    > [!NOTE]
    >
    > Because the errors are squared, the mean squared error and root means  squared error metrics emphasize the impact of outliers. These are good  metrics, but incorrect predictions can be very costly. If that is not  desired, a different metric called mean absolute error averages the  absolute values of the errors, so it doesn't emphasize the large errors.

- Mean Absolute Error (MAE)

  These are good metrics, but incorrect predictions can be very costly. If that is not desired, a different metric called mean absolute error  averages the absolute values of the errors, so it doesn't emphasize the  large errors.

### AI/ML Business Metrics

- Define business goals at the beginning

- Identify business metrics to improve

- Evaluate risk and cost of errors

- Measure actual improvement

- Measure actual costs

  > [!NOTE]
  >
  > Also, consider the actual cost of building and operating the model and  compare this cost with the initial cost benefit model. This way you'll  be able to calculate the return on investment. AWS allows you to define  cost allocation tags that are assigned to the resources that you create. For example, you can define a tag with the name of ML project and the  name of your project as the value. You add that tag to all the resources used in your pipeline. Then you can filter the cost reports in AWS Cost Explorer to determine the actual AWS charges incurred for the project.

- Compare with cost-benefit model

- Determine return on investiment (ROI)


----

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]
