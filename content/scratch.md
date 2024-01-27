---
draft: true
---
what is machine learning
machine learning is the intersection is stat and ai
machine learning is designing algorithms for inferring unknowns from knowns

ml applications





Here are some of the major supervised algorithms that I've run into and find useful to consider, with rough dates & scholars:

- 1970 - Ridge Regression (Hoerl & Kennard)
    
- 1980 - CHAID Decision Tree (Kass)
    
- 1983 - Classification and Regression Trees (Breiman, Friedman et al)
    
- 1986 - Generalized Additive Models (Hastie & Tibshirani)
    
- 1989 - Thin-Plate Splines (Bookstein)
    
- 1991 - Multivariate Adaptive Regression Splines (Friedman)
    
- 1992 - Support Vector Machines (Boser, Guyon, Vapnik)
    
- 1996 - Lasso (Tibshirani)
    
- 1996 - Bagging (Breiman)
    
- 2001 - Random Forest (Breiman)
    
- 2001 - Gradient Boosting (Friedman)
    
- 2005 - Elastic Net (Zou & Hastie)
    
- 2005 - RuleFit (Friedman)
    

Where you could find the specific papers via google scholar etc. or just get the flavor in major ML textbooks.

I haven't dug into neutral nets, unsupervised, or reinforcement learning much so leaving that line of research to someone else.



During the last decade numerous frameworks for machine learning appeared, but their open source implementations are seeming to be most promising due to several reasons: available source codes, big community of developers and end users, and, consequently, numerous applications, which demonstrate and validate the maturity of these frameworks.

**Deep Learning4j: Open source, distributed, deep learning library for the JVM**

- [Deep Learning4j](https://deeplearning4j.org/) (DL4J) is positioned as the open-source distributed deep-learning library written for Java and Scala that can be integrated with Hadoop and Spark. The libraries are completely open-source, Apache 2.0, and maintained by the developer community
- Deeplearning4j is written in Java and is compatible with any JVM language, such as Scala, Clojure or Kotlin. The underlying computations are written in C, C++ and Cuda. Keras will serve as the Python API.
- It is designed to be used on distributed GPUs and CPUs platforms, and provides the ability to work with arbitrary n-dimensional arrays (also called tensors), and usage of CPU and GPU resources.
- Unlike many other frameworks, DL4J splits the opti- 3 mization algorithm from the updater algorithm. This allows to be flexible while trying to find a combination that works best for data and problem.

**TensorFlow: flexible framework for large-scale machine learning**

- TensorFlow is an open source software library for machine learning and deep neural network research developed and released by the Google Brain Team within Google’s AI organization in 2015. TensorFlow is now widely used by several companies, including Dropbox, eBay, Intel, Twitter, and Uber.
- TensorFlow is available in Python, C++, Haskell, Java, Go, Rust, and most recently, JavaScript. One can also find third-party packages for other programming languages. The system is designed to facilitate research in machine learning, and to make it quick and easy to transition from research prototype to production system. The framework allows you to develop neural networks (and even other computational models) using flowgraphs.
- TensorFlow is quite flexible and can be used across various computation platforms (CPUs, GPUs, and TPUs) and devices, from desktops to clusters of servers to mobile and edge systems. It runs on Mac, Windows, and Linux.
- A significant feature of this library is that numerical computations are done with data flow graphs consisting of nodes and edges. Nodes represent mathematical operations, and edges are multidimensional data arrays or tensors, on which these operations are performed.

**TensorBoard: a good tool for model training visualization**

- TensorBoard is a suite of tools for graphical representation of different aspects and stages of machine learning in TensorFlow.
- TensorBoard reads TensorFlow event files containing summary data (observations about a model’s specific operations) being generated while TensorFlow is running.
- A model structure shown with graphs allows researchers to make sure model components are located where needed and are connected correctly.
- With the graph visualizer, users can explore different layers of model abstraction, zooming in and out of any part of the schema. Another important benefit of TensorBoard visualization is that nodes of the same types and similar structures are painted with the same colors. Users can also look at coloring by device (CPU, GPU, or a combination of both), highlight a specific node with the “trace inputs” feature, and visualize one or several charts at a time.This visualization approach makes TensorBoard a popular tool for model performance evaluation, especially for models of complex structures like deep neural networks.

**H2O - fast, scalable, open-source machine learning and deep learning**

- H2O software is built on Java, Python, and R with a purpose to optimize machine learning for Big Data. Big Data Friendly means that one can use all of their data in realtime for better predictions with H2O’s fast in-memory distributed parallel processing capabilities.
- For production deployment a developer need not worry about the variation in the development platform and production environment. Using in-memory compression, H2O handles billions of data rows in-memory, even with a small cluster. To make it easier for non-engineers to create complete analytic workflows.
- H2O’s platform includes interfaces for R, Python, Scala, Java, JSON, and CoffeeScript/JavaScript, as well as a built-in web interface, Flow. H2O is designed to run in standalone mode, on Hadoop, or within a Spark Cluster, and typically deploys within minutes
- H2O implements almost all common machine learning algorithms, such as generalized linear modeling (linear regression, logistic regression, etc.), Naive Bayes, principal components analysis, time series, k-means clustering, Random Forest, Gradient Boosting, and Deep Learning.
- H2O models are compiled into POJO (Plain Old Java Files) or a MOJO (Model Object Optimized) format which once created can be utilized and deployed like any Standard Java Object.

**PyTorch: easy to use tool for research**

- PyTorch is an open source machine learning framework for deep neural networks that supports and accelerates GPUs. Developed by Facebook’s team together with engineers from Twitter, SalesForce, NRIA, ENS, ParisTech, Nvidia, Digital Reasoning, and INRIA, the library was first released in October 2016. PyTorch is built on Torch framework, but unlike predecessor that’s written in Lua, it supports commonly used Python.
- PyTorch was developed with the idea of providing as fast and flexible a modeling experience as possible. It’s worth mentioning that workflow in PyTorch is similar to the one in NumPy, a Python-based scientific computing library.
- A dynamic computational graph is one of the features making this library popular. In most frameworks like TensorFlow, Theano, CNTK, and Caffe, the models are built in a static way. A data scientist must change the whole structure of the neural network — rebuild it from scratch — to change the way it behaves. PyTorch makes it easier and faster. The framework allows for changing the network behavior arbitrarily without lag or overhead.

**Keras: lightweight, easy-to-use library for fast prototyping**

- Keras is a Python deep learning library capable of running on top off Theano, TensorFlow, or CNTK. The Google Brain team member Francois Chollet developed it to give data scientists the ability to run machine learning experiments fast.
- Keras is known for its user-friendliness, modularity, and ease of extensibility. It is suitable if you need a machine learning library that allows for easy and fast prototyping, supports both convolutional and recurrent networks, and runs optimally on both CPUs (central processing units) and GPUs (graphics processing units).
- Fast prototyping is possible with the help of the library’s high-level, understandable interface, the division of networks into sequences of separate modules that are easy to create and add.

**Caffe2: deep learning library with mobile deployment support**

- [Caffe2](https://en.wikipedia.org/wiki/Theano_(software)), an improved version of [Caffe](https://en.wikipedia.org/wiki/Theano_(software)), is an open machine learning framework build by Facebook for the streamline and flexible deep learning of complex models and support for mobile deployment. Originally [Caffe,](https://github.com/BVLC/caffe) deep learning framework made with expression, speed, and modularity in mind and was developed by the Berkeley Vision and Learning Center ( [BVLC](https://en.wikipedia.org/wiki/Theano_(software))) and community contributors.
- Users have several options to organize computation with the library, which can be installed and run on a desktop, in the cloud, or at a datacenter.
- The library has native Python and C++ APIs that work alternately, allowing developers to prototype on the go and optimize later.
- Deployed models can run fast on mobile devices through the integration with Xcode, Visual Studio, and Android Studio IDEs. This framework also allows for quick scaling up or down without the need for design refactoring.

**Theano, a Python library with optimizing compiler mathematical expressions**

- [Theano](https://en.wikipedia.org/wiki/Theano_(software)) was written at the [LISA](https://mila.quebec/) lab to support rapid development of efficient machine learning algorithms. Theano is named after the [Greek mathematician](https://en.wikipedia.org/wiki/Theano_(philosopher)), who may have been Pythagoras’ wife. Theano is released under a BSD license.
- Theano allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. Using Theano it is possible to attain speeds rivalling hand-crafted C implementations for problems involving large amounts of data. It can also surpass C on a CPU by many orders of magnitude by taking advantage of recent GPUs.
- Theano combines aspects of a computer algebra system (CAS) with aspects of an optimizing compiler. It can also generate customized C code for many mathematical operations. This combination of CAS with optimizing compilation is particularly useful for tasks in which complicated mathematical expressions are evaluated repeatedly and evaluation speed is critical.

**NumPy: an extension package for scientific computing with Python**

- Previously mentioned NumPy is an extension package for performing numerical computing with Python that replaced NumArray and Numeric. It supports multidimensional arrays (tables) and matrices. ML data is represented in arrays. And a matrix is a two-dimensional array of numbers.
- [NumPy](https://numpy.org/) contains broadcasting functions as tools for integrating C/C++ and the Fortran code. Its functionality also includes the Fourier transform, linear algebra, and random number capabilities.
- Data science practitioners can use NumPy as an effective container for storage of multidimensional generic data. Through the ability to define arbitrary data types, NumPy easily and quickly integrates with numerous kinds of databases.

**Scikit-learn: easy-to-use machine learning framework for numerous industries**

- [scikit-learn](https://scikit-learn.org/stable/index.html) is an open source Python machine learning library build on top of SciPy (Scientific Python), NumPy, and matplotlib. Initially started in 2007 by David Cournapeau as a Google Summer of Code project, scikit-learn is currently maintained by volunteers.
- scikit-learn provides users with a number of well-established algorithms for supervised and unsupervised learning. As of today, more than thousand people have contributed to it. This library is designed for production use due to the simplicity, qualitative code, collaboration options, performance, and extensive documentation written in plain language contribute to its popularity among various specialists.

**NLTK: Python-based human language data processing platform**

- The Natural Language Toolkit, or more commonly [NLTK](https://www.nltk.org/) , is a suite of [libraries](https://en.wikipedia.org/wiki/Library_(computing)) and programs for symbolic and statistical natural language processing (NLP) for English written in the [Python programming language](https://en.wikipedia.org/wiki/Python_(programming_language)). It was developed by Steven Bird and Edward Loper in the Department of Computer and Information Science at the [University of Pennsylvania.](https://en.wikipedia.org/wiki/University_of_Pennsylvania)
- [NLTK](https://www.nltk.org/) is a platform for the development of Python programs to work with human language and provides a flexible framework for various AI/ML research projects and production deployments, from standard implementations of all the basic data structures and algorithms, interfaces to dozens of widely used datasets (corpora), and a flexible and extensible architecture to the most complex interdependent tools environments.

**Datasets and Machine Learning**

One of the hardest problems to solve in deep learning has nothing to do with neural nets: it’s the problem of getting the right data in the right format. Machine learning methods learn from examples. It is important to have good grasp of input data and the various terminology used when describing data. Getting the right data means gathering or identifying the data that correlates with the outcomes one wants to predict. Collecting and constructing the training dataset; a sizable body of known data takes time and domain-specific knowledge of where and how to gather relevant information.

The right end format for deep learning is generally a tensor, or a multi-dimensional array. The data pipelines built for deep learning will generally convert all data; be it images, video, sound, voice, text or time series etc. into vectors and tensors to which linear algebra operations can be applied. Deep learning, and machine learning more generally, needs a good training dataset to work properly and will have definite impact on the desired outcome.

The training set acts as the benchmark against which deep-learning nets are trained, to create a useful training set, first step is to understand the problem you’re solving; i.e. what one wants their deep-learning nets to pay attention to, which outcomes they want to predict and right tool..

**pandas: a Python data analysis library enhancing analytics and modeling**

- Python [pandas](https://pandas.pydata.org/), a free library with the cutest name. Data science devotee [Wes McKinney](https://github.com/wesm) developed this library to make data analysis and modeling convenient in Python.
- pandas simplifies analysis by converting CSV, JSON, and TSV data files or a SQL database into a data frame, a Python object looking like an Excel or an SPSS table with rows and columns.
- Even more, pandas is combined with the [IPython](https://ipython.org/) toolkit and other libraries to enhance performance and support collaborative work.

**matplotlib: a Python machine learning library for quality visualizations**

- [matplotlib](https://matplotlib.org/)is a Python 2D plotting library. Plotting is a visualization of machine learning data. matplotlib originates from MATLAB: Its developer John D. Hunter emulated plotting commands from Mathworks’ MATLAB software. While written mostly in Python, the library is extended with NumPy and other code, so it performs well even when used for large arrays.
- Allows for generating production-quality visualizations with a few lines of code. The library’s functionality can be extended with third-party visualization packages like [seaborn](https://seaborn.pydata.org/), [ggplot](http://ggplot.yhathq.com/), and [HoloViews](http://holoviews.org/). Specialists can also add extra features using [Basemap](https://matplotlib.org/basemap/) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/) projection and mapping toolkits.

**Quick Reference - Additional toolkits / projects ..**

(Disclaimer: ML ecosystem is vast, evolving and very dynamic, this attempt of simplified article is a quick dipstick and might not claim comprehensiveness, pls note)

**General-Purpose Machine Learning**

- [Darknet](https://github.com/pjreddie/darknet) - Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
- [Recommender](https://github.com/GHamrouni/Recommender) - A C library for product recommendations/suggestions using collaborative filtering (CF).
- [Hybrid Recommender System](https://github.com/SeniorSA/hybrid-rs-trainner) - A hybrid recommender system based upon scikit-learn algorithms.
- [mgl](https://github.com/melisgl/mgl/)-Neuralnetworks(boltzmannmachines,feed-forwardandrecurrentnets), Gaussian Processes and [mgl-gpr](https://github.com/melisgl/mgl-gpr/) - Evolutionary algorithms
- [cl-libsvm](https://github.com/melisgl/cl-libsvm/) - Wrapper for the libsvm support vector machine library

**Computer Vision**

- [CCV](https://github.com/liuliu/ccv) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library
- [VLFeat](http://www.vlfeat.org/) - VLFeat is an open and portable library of computer vision algorithms, which has Matlab toolbox
- [DLib](https://http//dlib.net/imaging.html) - DLib has C++ and Python interfaces for face detection and training general object detectors.
- [EBLearn](http://eblearn.sourceforge.net/) - Eblearn is an object-oriented C++ library that implements various machine learning models
- [OpenCV](https://opencv.org/) - OpenCV has C++, C, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS.
- [grt](https://github.com/nickgillian/grt) - The Gesture Recognition Toolkit. GRT is a cross-platform, open-source, C++ machine learning library designed for real-time gesture recognition.

**Speech Recognition**

- [Kaldi](https://github.com/kaldi-asr/kaldi)- Kaldi is a toolkit for speech recognition written in C++ and licensed under the Apache License v2.0. Kaldi is intended for use by speech recognition researchers.
- [HTK](http://htk.eng.cam.ac.uk/) -The Hidden Markov Model Toolkit. HTK is a portable toolkit for building and manipulating hidden Markov models.
- [ChatterBot](https://github.com/gunthercox/ChatterBot): machine learning for conversational dialog engine and creating chat bots

**Automated machine learning**

- [TPOT](https://github.com/EpistasisLab/tpot) - Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming
- [auto-sklearn](https://github.com/automl/auto-sklearn) - is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator
- [MLBox](https://github.com/AxeldeRomblay/MLBox) - a powerful Automated Machine Learning python library.

**Other toolkits ..**

- [Kubeflow](https://github.com/kubeflow/kubeflow): machine learning toolkit for Kubernetes
- [imgaug](https://github.com/aleju/imgaug): image augmentation for deep learning
- [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn): a python package under scikit learn specifically for tacking imbalanced datasets
- [mlflow](https://github.com/mlflow/mlflow): open source platform to manage the ML lifecycle, including experimentation, reproducibility and deployment.
- [AirSim](https://github.com/Microsoft/AirSim): simulator for autonomous vehicles built on Unreal Engine / Unity, from Microsoft AI & Research

**Ensemble methods**

Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would. This has been the case in a number of machine learning competitions, where the winning solutions used ensemble methods.

- [ML-Ensemble](http://ml-ensemble.com/) - high performance ensemble learning
- [brew](https://github.com/ikki407/stacking)- Python Ensemble Learning API
- [Stacking](https://github.com/ikki407/stacking) - Simple and useful stacking library, written in Python.
- [stacked_generalization](https://github.com/fukatani/stacked_generalization) - library for machine learning stacking generalization.
- [vecstack](https://https//github.com/vecxoz/vecstack) - Python package for stacking (machine learning technique)