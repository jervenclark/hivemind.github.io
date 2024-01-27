


















Contents
--------


move to sidebar
hide

* (Top)
* 1Tasks
* 2Machine learning approaches



Toggle Machine learning approaches subsection

	+ 2.1Artificial neural networks
	+ 2.2Feature engineering
	+ 2.3Classification
	
		- 2.3.1Hidden Markov models
		- 2.3.2Convolutional neural networks
		- 2.3.3Self-supervised learning
		- 2.3.4Random forest
	+ 2.4Clustering
	
		- 2.4.1Clustering algorithms used in bioinformatics
	+ 2.5Workflow
	+ 2.6Data errors
* 3Applications



Toggle Applications subsection

	+ 3.1Precision/personalized medicine
	+ 3.2Genomics
	+ 3.3Proteomics
	+ 3.4Metagenomics
	+ 3.5Microarrays
	+ 3.6Systems biology
	+ 3.7Evolution
	+ 3.8Stroke diagnosis
	+ 3.9Text mining
	+ 3.10Clustering and abundance profiling of biosynthetic gene clusters
	+ 3.11Decodification of RiPPs chemical structures
	+ 3.12Mass spectral similarity scoring
* 4Databases



Toggle Databases subsection

	+ 4.1General databases by bioinformatics
	
		- 4.1.1National Center for Biotechnology Information
	+ 4.2Bioinformatics analysis for biosynthetic gene clusters
	
		- 4.2.1antiSMASH
		- 4.2.2gutSMASH
		- 4.2.3MIBiG
		- 4.2.4SILVA
		- 4.2.5Greengenes
		- 4.2.6Open Tree of Life Taxonomy
		- 4.2.7Ribosomal Database Project
* 5References














Toggle the table of contents







Machine learning in bioinformatics
==================================





3 languages



* العربية
* Español
* فارسی


Edit links










* Article
* Talk






English
















* Read
* Edit
* View history








Tools





Tools
move to sidebar
hide



 Actions
 

* Read
* Edit
* View history






 General
 

* What links here
* Related changes
* Upload file
* Special pages
* Permanent link
* Page information
* Cite this page
* Get shortened URL
* Wikidata item






 Print/export
 

* Download as PDF
* Printable version


























From Wikipedia, the free encyclopedia








|  |
| --- |
| Part of a series on |
| Artificial intelligence |
| ![](//upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Anatomy-1751201_1280.png/100px-Anatomy-1751201_1280.png) |
| Major goals
* Artificial general intelligence
* Planning
* Computer vision
* General game playing
* Knowledge reasoning
* Machine learning
* Natural language processing
* Robotics
* AI safety
 |
| Approaches
* Symbolic
* Deep learning
* Bayesian networks
* Evolutionary algorithms
* Situated approach
* Hybrid intelligent systems
* Systems integration
 |
| Applications
* Projects
* Deepfake
* Machine translation
* Art
* Healthcare
	+ Mental health
* Government
* Music
* Industry
* Earth sciences
* Bioinformatics
* Physics
 |
| Philosophy
* Chinese room
* Friendly AI
* Control problem/Takeover
* Ethics
* Existential risk
* Turing test
* Regulation
 |
| History
* Timeline
* Progress
* AI winter
* AI boom
 |
| Technology
* Programming languages
 |
| Glossary
* Glossary
 |
| * v
* t
* e
 |


**Machine learning in bioinformatics** is the application of machine learning algorithms to bioinformatics,[1] including genomics, proteomics, microarrays, systems biology, evolution, and text mining.[2][3]


Prior to the emergence of machine learning, bioinformatics algorithms had to be programmed by hand; for problems such as protein structure prediction, this proved difficult.[4] Machine learning techniques, such as deep learning can learn features of data sets, instead of requiring the programmer to define them individually. The algorithm can further learn how to combine low-level features into more abstract features, and so on. This multi-layered approach allows such systems to make sophisticated predictions when appropriately trained. These methods contrast with other computational biology approaches which, while exploiting existing datasets, do not allow the data to be interpreted and analyzed in unanticipated ways. 




Tasks[edit]
-----------


Machine learning algorithms in bioinformatics can be used for prediction, classification, and feature selection. Methods to achieve this task are varied and span many disciplines; most well known among them are machine learning and statistics. Classification and prediction tasks aim at building models that describe and distinguish classes or concepts for future prediction. The differences between them are the following:



* Classification/recognition outputs a categorical class, while prediction outputs a numerical valued feature.
* The type of algorithm, or process used to build the predictive models from data using analogies, rules, neural networks, probabilities, and/or statistics.


Due to the exponential growth of information technologies and applicable models, including artificial intelligence and data mining, in addition to the access ever-more comprehensive data sets, new and better information analysis techniques have been created, based on their ability to learn. Such models allow reach beyond description and provide insights in the form of testable models.



Machine learning approaches[edit]
---------------------------------


### Artificial neural networks[edit]


Artificial neural networks in bioinformatics have been used for:[5]



* Comparing and aligning RNA, protein, and DNA sequences.
* Identification of promoters and finding genes from sequences related to DNA.
* Interpreting the expression-gene and micro-array data.
* Identifying the network (regulatory) of genes.
* Learning evolutionary relationships by constructing phylogenetic trees.
* Classifying and predicting protein structure.
* Molecular design and docking.


### Feature engineering[edit]


The way that features, often vectors in a many-dimensional space, are extracted from the domain data is an important component of learning systems.[6] In genomics, a typical representation of a sequence is a vector of k-mers frequencies, which is a vector of dimension 




4

k




{\displaystyle 4^{k}}

![{\displaystyle 4^{k}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/563b380d4a8d0bb311093679b5696422a1c6f66d) whose entries count the appearance of each subsequence of length 



k


{\displaystyle k}

![k](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3c9a2c7b599b37105512c5d570edc034056dd40) in a given sequence. Since for a value as small as 



k
=
12


{\displaystyle k=12}

![{\displaystyle k=12}](https://wikimedia.org/api/rest_v1/media/math/render/svg/ce3998d99da56b6042d806b9b81836ad78965107) the dimensionality of these vectors is huge (e.g. in this case the dimension is 




4

12


≈
16
×

10

6




{\displaystyle 4^{12}\approx 16\times 10^{6}}

![{\displaystyle 4^{12}\approx 16\times 10^{6}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/9b1bf750825a2f8ec039d81f1d58d3cc8ccf2c75)), techniques such as principal component analysis are used to project the data to a lower dimensional space, thus selecting a smaller set of features from the sequences.[6][*additional citation(s) needed*]



### Classification[edit]


In this type of machine learning task, the output is a discrete variable. One example of this type of task in bioinformatics is labeling new genomic data (such as genomes of unculturable bacteria) based on a model of already labeled data.[6]



#### Hidden Markov models[edit]


Hidden Markov models (HMMs) are a class of statistical models for sequential data (often related to systems evolving over time). An HMM is composed of two mathematical objects: an observed state‐dependent process 




X

1


,

X

2


,
…
,

X

M




{\displaystyle X\_{1},X\_{2},\ldots ,X\_{M}}

![{\displaystyle X_{1},X_{2},\ldots ,X_{M}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/453e34b952c09e9cf939f72e3e82a392311dba34), and an unobserved (hidden) state process 




S

1


,

S

2


,
…
,

S

T




{\displaystyle S\_{1},S\_{2},\ldots ,S\_{T}}

![{\displaystyle S_{1},S_{2},\ldots ,S_{T}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/44233318e4d149e3e80fd60271d6aa01b5127fa4). In an HMM, the state process is not directly observed – it is a 'hidden' (or 'latent') variable – but observations are made of a state‐dependent process (or observation process) that is driven by the underlying state process (and which can thus be regarded as a noisy measurement of the system states of interest).[7] HMMs can be formulated in continuous time.[8][9]


HMMs can be used to profile and convert a multiple sequence alignment into a position-specific scoring system suitable for searching databases for homologous sequences remotely.[10] Additionally, ecological phenomena can be described by HMMs.[11]



#### Convolutional neural networks[edit]


Convolutional neural networks (CNN) are a class of deep neural network whose architecture is based on shared weights of convolution kernels or filters that slide along input features, providing translation-equivariant responses known as feature maps.[12][13] CNNs take advantage of the hierarchical pattern in data and assemble patterns of increasing complexity using smaller and simpler patterns discovered via their filters.[14]


Convolutional networks were inspired by biological processes[15][16][17][18] in that the connectivity pattern between neurons resembles the organization of the animal visual cortex. Individual cortical neurons respond to stimuli only in a restricted region of the visual field known as the receptive field. The receptive fields of different neurons partially overlap such that they cover the entire visual field.


CNN uses relatively little pre-processing compared to other image classification algorithms. This means that the network learns to optimize the filters (or kernels) through automated learning, whereas in traditional algorithms these filters are hand-engineered. This reduced reliance on prior knowledge of the analyst and on human intervention in manual feature extraction makes CNNs a desirable model.[14]


A phylogenetic convolutional neural network (Ph-CNN) is a convolutional neural network architecture proposed by Fioranti et al. in 2018 to classify metagenomics data.[19] In this approach, phylogenetic data is endowed with patristic distance (the sum of the lengths of all branches connecting two operational taxonomic units [OTU]) to select k-neighborhoods for each OTU, and each OTU and its neighbors are processed with convolutional filters.



#### Self-supervised learning[edit]


Unlike supervised methods, self-supervised learning methods learn representations without relying on annotated data. That is well-suited for genomics, where high throughput sequencing techniques can create potentially large amounts of unlabeled data. Some examples of self-supervised learning methods applied on genomics include DNABERT and Self-GenomeNet.[20][21]



#### Random forest[edit]


![](//upload.wikimedia.org/wikipedia/commons/thumb/8/87/Some_bioinformatic_applications_of_Random_Forest.jpg/220px-Some_bioinformatic_applications_of_Random_Forest.jpg)Some bioinformatic applications[*which?*] of Random Forest.
Random forests (RF) classify by constructing an ensemble of decision trees, and outputting the average prediction of the individual trees.[22] This is a modification of bootstrap aggregating (which aggregates a large collection of decision trees) and can be used for classification or regression.[23][24]


As random forests give an internal estimate of generalization error, cross-validation is unnecessary. In addition, they produce proximities, which can be used to impute missing values, and which enable novel data visualizations.[25]


Computationally, random forests are appealing because they naturally handle both regression and (multiclass) classification, are relatively fast to train and to predict, depend only on one or two tuning parameters, have a built-in estimate of the generalization error, can be used directly for high-dimensional problems, and can easily be implemented in parallel. Statistically, random forests are appealing for additional features, such as measures of variable importance, differential class weighting, missing value imputation, visualization, outlier detection, and unsupervised learning.[25]



### Clustering[edit]


Clustering - the partitioning of a data set into disjoint subsets, so that the data in each subset are as close as possible to each other and as distant as possible from data in any other subset, according to some defined distance or similarity function - is a common technique for statistical data analysis.


Clustering is central to much data-driven bioinformatics research and serves as a powerful computational method whereby means of hierarchical, centroid-based, distribution-based, density-based, and self-organizing maps classification, has long been studied and used in classical machine learning settings. Particularly, clustering helps to analyze unstructured and high-dimensional data in the form of sequences, expressions, texts, images, and so on. Clustering is also used to gain insights into biological processes at the genomic level, e.g. gene functions, cellular processes, subtypes of cells, gene regulation, and metabolic processes.[26]



#### Clustering algorithms used in bioinformatics[edit]


Data clustering algorithms can be hierarchical or partitional. Hierarchical algorithms find successive clusters using previously established clusters, whereas partitional algorithms determine all clusters at once. Hierarchical algorithms can be agglomerative (bottom-up) or divisive (top-down).


Agglomerative algorithms begin with each element as a separate cluster and merge them in successively larger clusters. Divisive algorithms begin with the whole set and proceed to divide it into successively smaller clusters. Hierarchical clustering is calculated using metrics on Euclidean spaces, the most commonly used is the Euclidean distance computed by finding the square of the difference between each variable, adding all the squares, and finding the square root of the said sum. An example of a hierarchical clustering algorithm is BIRCH, which is particularly good on bioinformatics for its nearly linear time complexity given generally large datasets.[27] Partitioning algorithms are based on specifying an initial number of groups, and iteratively reallocating objects among groups to convergence. This algorithm typically determines all clusters at once. Most applications adopt one of two popular heuristic methods: k-means algorithm or k-medoids. Other algorithms do not require an initial number of groups, such as affinity propagation. In a genomic setting this algorithm has been used both to cluster biosynthetic gene clusters in gene cluster families(GCF) and to cluster said GCFs.[28]



### Workflow[edit]


Typically, a workflow for applying machine learning to biological data goes through four steps:[2]



* *Recording,* including capture and storage. In this step, different information sources may be merged into a single set.
* *Preprocessing,* including cleaning and restructuring into a ready-to-analyze form. In this step, uncorrected data are eliminated or corrected, while missing data maybe imputed and relevant variables chosen.
* *Analysis,* evaluating data using either supervised or unsupervised algorithms. The algorithm is typically trained on a subset of data, optimizing parameters, and evaluated on a separate test subset.
* *Visualization and interpretation,* where knowledge is represented effectively using different methods to assess the significance and importance of the findings.


### Data errors[edit]


* Duplicate data is a significant issue in bioinformatics. Publicly available data may be of uncertain quality.[29]
* Errors during experimentation.[29]
* Erroneous interpretation.[29]
* Typing mistakes.[29]
* Non-standardized methods (3D structure in PDB from multiple sources, X-ray diffraction, theoretical modeling, nuclear magnetic resonance, etc.) are used in experiments.[29]


Applications[edit]
------------------


In general, a machine learning system can usually be trained to recognize elements of a certain class given sufficient samples.[30] For example, machine learning methods can be trained to identify specific visual features such as splice sites.[31]


Support vector machines have been extensively used in cancer genomic studies.[32] In addition, deep learning has been incorporated into bioinformatic algorithms. Deep learning applications have been used for regulatory genomics and cellular imaging.[33] Other applications include medical image classification, genomic sequence analysis, as well as protein structure classification and prediction.[34] Deep learning has been applied to regulatory genomics, variant calling and pathogenicity scores.[35] Natural language processing and text mining have helped to understand phenomena including protein-protein interaction, gene-disease relation as well as predicting biomolecule structures and functions.[36]



### Precision/personalized medicine[edit]


Natural language processing algorithms personalized medicine for patients who suffer genetic diseases, by combining the extraction of clinical information and genomic data available from the patients. Institutes such as Health-funded Pharmacogenomics Research Network focus on finding breast cancer treatments.[37]


Precision medicine considers individual genomic variability, enabled by large-scale biological databases. Machine learning can be applied to perform the matching function between (groups of patients) and specific treatment modalities.[38]


Computational techniques are used to solve other problems, such as efficient primer design for PCR, biological-image analysis and back translation of proteins (which is, given the degeneration of the genetic code, a complex combinatorial problem).[2]



### Genomics[edit]


![](//upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Growth_of_GenBank.png/220px-Growth_of_GenBank.png)The growth of GenBank, a genomic sequence database provided by the National center for Biotechnology Information (NCBI)
While genomic sequence data has historically been sparse due to the technical difficulty of sequencing a piece of DNA, the number of available sequences is growing. On average, the number of bases available in the GenBank public repository has doubled every 18 months since 1982.[39] However, while raw data was becoming increasingly available and accessible, As of 2002[update], biological interpretation of this data was occurring at a much slower pace.[40] This made for an increasing need for developing computational genomics tools, including machine learning systems, that can automatically determine the location of protein-encoding genes within a given DNA sequence (i.e. gene prediction).[40]


Gene prediction is commonly performed through both *extrinsic searches* and *intrinsic searches*.[40] For the extrinsic search, the input DNA sequence is run through a large database of sequences whose genes have been previously discovered and their locations annotated and identifying the target sequence's genes by determining which strings of bases within the sequence are homologous to known gene sequences. However, not all the genes in a given input sequence can be identified through homology alone, due to limits in the size of the database of known and annotated gene sequences. Therefore, an intrinsic search is needed where a gene prediction program attempts to identify the remaining genes from the DNA sequence alone.[40]


Machine learning has also been used for the problem of multiple sequence alignment which involves aligning many DNA or amino acid sequences in order to determine regions of similarity that could indicate a shared evolutionary history.[2] It can also be used to detect and visualize genome rearrangements.[41]



### Proteomics[edit]


![](//upload.wikimedia.org/wikipedia/commons/thumb/4/46/C16orf95_protein_secondary_structure_prediction.png/220px-C16orf95_protein_secondary_structure_prediction.png)A protein's amino acid sequence annotated with the protein secondary structure. Each amino acid is labeled as either an alpha helix, a beta-sheet, or a coil.
Proteins, strings of amino acids, gain much of their function from protein folding, where they conform into a three-dimensional structure, including the primary structure, the secondary structure (alpha helices and beta sheets), the tertiary structure, and the quaternary structure.


Protein secondary structure prediction is a main focus of this subfield as tertiary and quaternary structures are determined based on the secondary structure.[4] Solving the true structure of a protein is expensive and time-intensive, furthering the need for systems that can accurately predict the structure of a protein by analyzing the amino acid sequence directly.[4][2] Prior to machine learning, researchers needed to conduct this prediction manually. This trend began in 1951 when Pauling and Corey released their work on predicting the hydrogen bond configurations of a protein from a polypeptide chain.[42] Automatic feature learning reaches an accuracy of 82-84%.[4][43] The current state-of-the-art in secondary structure prediction uses a system called DeepCNF (deep convolutional neural fields) which relies on the machine learning model of artificial neural networks to achieve an accuracy of approximately 84% when tasked to classify the amino acids of a protein sequence into one of three structural classes (helix, sheet, or coil).[43] [*needs update*] The theoretical limit for three-state protein secondary structure is 88–90%.[4]


Machine learning has also been applied to proteomics problems such as protein side-chain prediction, protein loop modeling, and protein contact map prediction.[2]



### Metagenomics[edit]


Metagenomics is the study of microbial communities from environmental DNA samples.[44] Currently, limitations and challenges predominate in the implementation of machine learning tools due to the amount of data in environmental samples.[45] Supercomputers and web servers have made access to these tools easier.[46] The high dimensionality of microbiome datasets is a major challenge in studying the microbiome; this significantly limits the power of current approaches for identifying true differences and increases the chance of false discoveries.[47][*better source needed*]


Despite their importance, machine learning tools related to metagenomics have focused on the study of gut microbiota and the relationship with digestive diseases, such as inflammatory bowel disease (IBD), *Clostridioides difficile* infection (CDI), colorectal cancer and diabetes, seeking better diagnosis and treatments.[46] Many algorithms were developed to classify microbial communities according to the health condition of the host, regardless of the type of sequence data, e.g. 16S rRNA or whole-genome sequencing (WGS), using methods such as least absolute shrinkage and selection operator classifier, random forest, supervised classification model, and gradient boosted tree model. Neural networks, such as recurrent neural networks (RNN), convolutional neural networks (CNN), and Hopfield neural networks have been added.[46] For example, in 2018, Fioravanti et al. developed an algorithm called Ph-CNN to classify data samples from healthy patients and patients with IBD symptoms (to distinguish healthy and sick patients) by using phylogenetic trees and convolutional neural networks.[48]


In addition, random forest (RF) methods and implemented importance measures help in the identification of microbiome species that can be used to distinguish diseased and non-diseased samples. However, the performance of a decision tree and the diversity of decision trees in the ensemble significantly influence the performance of RF algorithms. The generalization error for RF measures how accurate the individual classifiers are and their interdependence. Therefore, the high dimensionality problems of microbiome datasets pose challenges. Effective approaches require many possible variable combinations, which exponentially increases the computational burden as the number of features increases.[47]


For microbiome analysis in 2020 Dang & Kishino[47] developed a novel analysis pipeline. The core of the pipeline is an RF classifier coupled with forwarding variable selection (RF-FVS), which selects a minimum-size core set of microbial species or functional signatures that maximize the predictive classifier performance. The framework combines:



* identifying a few significant features by a massively parallel forward variable selection procedure
* mapping the selected species on a phylogenetic tree, and
* predicting functional profiles by functional gene enrichment analysis from metagenomic 16S rRNA data.


They demonstrated performance by analyzing two published datasets from large-scale case-control studies:



* 16S rRNA gene amplicon data for *C. difficile* infection (CDI) and
* shotgun metagenomics data for human colorectal cancer (CRC).


The proposed approach improved the accuracy from 81% to 99.01% for CDI and from 75.14% to 90.17% for CRC.


The use of machine learning in environmental samples has been less explored, maybe because of data complexity, especially from WGS. Some works show that it is possible to apply these tools in environmental samples. In 2021 Dhungel et al.,[49] designed an R package called MegaR. This package allows working with 16S rRNA and whole metagenomic sequences to make taxonomic profiles and classification models by machine learning models. MegaR includes a comfortable visualization environment to improve the user experience. Machine learning in environmental metagenomics can help to answer questions related to the interactions between microbial communities and ecosystems, e.g. the work of Xun et al., in 2021[50] where the use of different machine learning methods offered insights on the relationship among the soil, microbiome biodiversity, and ecosystem stability.



### Microarrays[edit]


Microarrays, a type of lab-on-a-chip, are used for automatically collecting data about large amounts of biological material. Machine learning can aid in analysis, and has been applied to expression pattern identification, classification, and genetic network induction.[2]



![](//upload.wikimedia.org/wikipedia/commons/thumb/e/e2/DNA-microarray_analysis.jpg/220px-DNA-microarray_analysis.jpg)A DNA-microarray analysis of Burkitt's lymphoma and diffuse large B-cell lymphoma (DLBCL), which differences in gene expression patterns
This technology is especially useful for monitoring gene expression, aiding in diagnosing cancer by examining which genes are expressed.[51] One of the main tasks is identifying which genes are expressed based on the collected data.[2] In addition, due to the huge number of genes on which data is collected by the microarray, winnowing the large amount of irrelevant data to the task of expressed gene identification is challenging. Machine learning presents a potential solution as various classification methods can be used to perform this identification. The most commonly used methods are radial basis function networks, deep learning, Bayesian classification, decision trees, and random forest.[51]



### Systems biology[edit]


Systems biology focuses on the study of emergent behaviors from complex interactions of simple biological components in a system. Such components can include DNA, RNA, proteins, and metabolites.[52]


Machine learning has been used to aid in modeling these interactions in domains such as genetic networks, signal transduction networks, and metabolic pathways.[2] Probabilistic graphical models, a machine learning technique for determining the relationship between different variables, are one of the most commonly used methods for modeling genetic networks.[2] In addition, machine learning has been applied to systems biology problems such as identifying transcription factor binding sites using Markov chain optimization.[2] Genetic algorithms, machine learning techniques which are based on the natural process of evolution, have been used to model genetic networks and regulatory structures.[2]


Other systems biology applications of machine learning include the task of enzyme function prediction, high throughput microarray data analysis, analysis of genome-wide association studies to better understand markers of disease, protein function prediction.[53]



### Evolution[edit]


This domain, particularly phylogenetic tree reconstruction, uses the features of machine learning techniques. Phylogenetic trees are schematic representations of the evolution of organisms. Initially, they were constructed using features such as morphological and metabolic features. Later, due to the availability of genome sequences, the construction of the phylogenetic tree algorithm used the concept based on genome comparison. With the help of optimization techniques, a comparison was done by means of multiple sequence alignment.[54]



### Stroke diagnosis[edit]


Machine learning methods for the analysis of neuroimaging data are used to help diagnose stroke. Historically multiple approaches to this problem involved neural networks.[55][56]


Multiple approaches to detect strokes used machine learning. As proposed by Mirtskhulava,[57] feed-forward networks were tested to detect strokes using neural imaging. As proposed by Titano[58] 3D-CNN techniques were tested in supervised classification to screen head CT images for acute neurologic events. Three-dimensional CNN and SVM methods are often used.[56]



### Text mining[edit]


The increase in biological publications increased the difficulty in searching and compiling relevant available information on a given topic. This task is known as knowledge extraction. It is necessary for biological data collection which can then in turn be fed into machine learning algorithms to generate new biological knowledge.[2][59] Machine learning can be used for this knowledge extraction task using techniques such as natural language processing to extract the useful information from human-generated reports in a database. Text Nailing, an alternative approach to machine learning, capable of extracting features from clinical narrative notes was introduced in 2017.


This technique has been applied to the search for novel drug targets, as this task requires the examination of information stored in biological databases and journals.[59] Annotations of proteins in protein databases often do not reflect the complete known set of knowledge of each protein, so additional information must be extracted from biomedical literature. Machine learning has been applied to the automatic annotation of gene and protein function, determination of the protein subcellular localization, DNA-expression array analysis, large-scale protein interaction analysis, and molecule interaction analysis.[59]


Another application of text mining is the detection and visualization of distinct DNA regions given sufficient reference data.[60]



### Clustering and abundance profiling of biosynthetic gene clusters[edit]


Microbial communities are complex assembles of diverse microorganisms,[61] where symbiont partners constantly produce diverse metabolites derived from the primary and secondary (specialized) metabolism, from which metabolism plays an important role in microbial interaction.[62] Metagenomic and metatranscriptomic data are an important source for deciphering communications signals.


Molecular mechanisms produce specialized metabolites in various ways. Biosynthetic Gene Clusters (BGCs) attract attention, since several metabolites are clinically valuable, anti-microbial, anti-fungal, anti-parasitic, anti-tumor and immunosuppressive agents produced by the modular action of multi-enzymatic, multi-domains gene clusters, such as Nonribosomal peptide synthetases (NRPSs) and polyketide synthases (PKSs).[63] Diverse studies[64][65][66][67][68][69][70][71] show that grouping BGCs that share homologous core genes into gene cluster families (GCFs) can yield useful insights into the chemical diversity of the analyzed strains, and can support linking BGCs to their secondary metabolites.[65][67] GCFs have been used as functional markers in human health studies[72][73] and to study the ability of soil to suppress fungal pathogens.[74] Given their direct relationship to catalytic enzymes, and compounds produced from their encoded pathways, BGCs/GCFs can serve as a proxy to explore the chemical space of microbial secondary metabolism. Cataloging GCFs in sequenced microbial genomes yields an overview of the existing chemical diversity and offers insights into future priorities.[64][66] Tools such as BiG-SLiCE and BIG-MAP[75] have emerged with the sole purpose of unveiling the importance of BGCs in natural environments.



### Decodification of RiPPs chemical structures[edit]


The increase of experimentally characterized ribosomally synthesized and post-translationally modified peptides (RiPPs), together with the availability of information on their sequence and chemical structure, selected from databases such as BAGEL, BACTIBASE, MIBIG, and THIOBASE, provide the opportunity to develop machine learning tools to decode the chemical structure and classify them.


In 2017, researchers at the National Institute of Immunology of New Delhi, India, developed RiPPMiner[76] software, a bioinformatics resource for decoding RiPP chemical structures by genome mining. The RiPPMiner web server consists of a query interface and the RiPPDB database. RiPPMiner defines 12 subclasses of RiPPs, predicting the cleavage site of the leader peptide and the final cross-link of the RiPP chemical structure.



### Mass spectral similarity scoring[edit]


Many tandem mass spectrometry **(**MS/MS**)** based metabolomics studies, such as library matching and molecular networking, use spectral similarity as a proxy for structural similarity. Spec2vec[77] algorithm provides a new way of spectral similarity score, based on Word2Vec. Spec2Vec learns fragmental relationships within a large set of spectral data, in order to assess spectral similarities between molecules and to classify unknown molecules through these comparisons.


For systemic annotation, some metabolomics studies rely on fitting measured fragmentation mass spectra to library spectra or contrasting spectra via network analysis. Scoring functions are used to determine the similarity between pairs of fragment spectra as part of these processes. So far, no research has suggested scores that are significantly different from the commonly utilized cosine-based similarity.[78]



Databases[edit]
---------------


An important part of bioinformatics is the management of big datasets, known as databases of reference. Databases exist for each type of biological data, for example for biosynthetic gene clusters and metagenomes.



### General databases by bioinformatics[edit]


#### National Center for Biotechnology Information[edit]


The National Center for Biotechnology Information (NCBI)[79] provides a large suite of online resources for biological information and data, including the GenBank nucleic acid sequence database and the PubMed database of citations and abstracts for published life science journals. Augmenting many of the Web applications are custom implementations of the BLAST program optimized to search specialized data sets. Resources include PubMed Data Management, RefSeq Functional Elements, genome data download, variation services API, Magic-BLAST, QuickBLASTp, and Identical Protein Groups. All of these resources can be accessed through NCBI.[80]



### Bioinformatics analysis for biosynthetic gene clusters[edit]


#### antiSMASH[edit]


antiSMASH allows the rapid genome-wide identification, annotation and analysis of secondary metabolite biosynthesis gene clusters in bacterial and fungal genomes. It integrates and cross-links with a large number of in silico secondary metabolite analysis tools.[81]



#### gutSMASH[edit]


gutSMASH is a tool that systematically evaluates bacterial metabolic potential by predicting both known and novel anaerobic metabolic gene clusters (MGCs) from the gut microbiome.



#### MIBiG[edit]


MIBiG,[82] the minimum information about a biosynthetic gene cluster specification, provides a standard for annotations and metadata on biosynthetic gene clusters and their molecular products. MIBiG is a Genomic Standards Consortium project that builds on the minimum information about any sequence (MIxS) framework.[83]


MIBiG facilitates the standardized deposition and retrieval of biosynthetic gene cluster data as well as the development of comprehensive comparative analysis tools. It empowers next-generation research on the biosynthesis, chemistry and ecology of broad classes of societally relevant bioactive secondary metabolites, guided by robust experimental evidence and rich metadata components.[84]



#### SILVA[edit]


SILVA[85] is an interdisciplinary project among biologists and computers scientists assembling a complete database of RNA ribosomal (rRNA) sequences of genes, both small (16S, 18S, SSU) and large (23S, 28S, LSU) subunits, which belong to the bacteria, archaea and eukarya domains. These data are freely available for academic and commercial use.[86]



#### Greengenes[edit]


Greengenes[87] is a full-length 16S rRNA gene database that provides chimera screening, standard alignment and a curated taxonomy based on de novo tree inference.[88][89] Overview: 



* 1,012,863 RNA sequences from 92,684 organisms contributed to RNAcentral.
* The shortest sequence has 1,253 nucleotides, the longest 2,368.
* The average length is 1,402 nucleotides.
* Database version: 13.5.


#### Open Tree of Life Taxonomy[edit]


Open Tree of Life Taxonomy (OTT)[90] aims to build a complete, dynamic, and digitally available Tree of Life by synthesizing published phylogenetic trees along with taxonomic data. Phylogenetic trees have been classified, aligned, and merged. Taxonomies have been used to fill in sparse regions and gaps left by phylogenies. OTT is a base that has been little used for sequencing analyzes of the 16S region, however, it has a greater number of sequences classified taxonomically down to the genus level compared to SILVA and Greengenes. However, in terms of classification at the edge level, it contains a lesser amount of information[91]



#### Ribosomal Database Project[edit]


Ribosomal Database Project (RDP)[92] is a database that provides RNA ribosomal (rRNA) sequences of small subunits of domain bacterial and archaeal (16S); and fungal rRNA sequences of large subunits (28S).[93]



References[edit]
----------------



1. **^** Chicco D (December 2017). "Ten quick tips for machine learning in computational biology". *BioData Mining*. **10** (35): 35. doi:10.1186/s13040-017-0155-3. PMC 5721660. PMID 29234465.
2. ^ ***a*** ***b*** ***c*** ***d*** ***e*** ***f*** ***g*** ***h*** ***i*** ***j*** ***k*** ***l*** ***m*** Larrañaga P, Calvo B, Santana R, Bielza C, Galdiano J, Inza I, et al. (March 2006). "Machine learning in bioinformatics". *Briefings in Bioinformatics*. **7** (1): 86–112. doi:10.1093/bib/bbk007. PMID 16761367.
3. **^** Pérez-Wohlfeil E, Torrenoa O, Bellis LJ, Fernandes PL, Leskosek B, Trellesa O (December 2018). "Training bioinformaticians in High Performance Computing". *Heliyon*. **4** (12): e01057. Bibcode:2018Heliy...401057P. doi:10.1016/j.heliyon.2018.e01057. PMC 6299036. PMID 30582061.
4. ^ ***a*** ***b*** ***c*** ***d*** ***e*** Yang Y, Gao J, Wang J, Heffernan R, Hanson J, Paliwal K, Zhou Y (May 2018). "Sixty-five years of the long march in protein secondary structure prediction: the final stretch?". *Briefings in Bioinformatics*. **19** (3): 482–494. doi:10.1093/bib/bbw129. PMC 5952956. PMID 28040746.
5. **^** Shastry KA, Sanjay HA (2020). "Machine Learning for Bioinformatics". In Srinivasa K, Siddesh G, Manisekhar S (eds.). *Statistical Modelling and Machine Learning Principles for Bioinformatics Techniques, Tools, and Applications*. Algorithms for Intelligent Systems. Singapore: Springer. pp. 25–39. doi:10.1007/978-981-15-2445-5\_3. ISBN 978-981-15-2445-5. S2CID 214350490. Retrieved June 28, 2021.
6. ^ ***a*** ***b*** ***c*** Soueidan H, Nikolski M (2019). "Machine learning for metagenomics: methods and tools". *Metagenomics*. **1**. arXiv:1510.06621. doi:10.1515/metgen-2016-0001. ISSN 2449-7657. S2CID 17418188.
7. **^** Rabiner L, Juang B (January 1986). "An introduction to hidden Markov models". *IEEE ASSP Magazine*. **3** (1): 4–16. doi:10.1109/MASSP.1986.1165342. ISSN 1558-1284. S2CID 11358505.
8. **^** Jackson CH, Sharples LD, Thompson SG, Duffy SW, Couto E (July 2003). "Multistate Markov models for disease progression with classification error". *Journal of the Royal Statistical Society, Series D (The Statistician)*. **52** (2): 193–209. doi:10.1111/1467-9884.00351. S2CID 9824404.
9. **^** Amoros R, King R, Toyoda H, Kumada T, Johnson PJ, Bird TG (May 30, 2019). "A continuous-time hidden Markov model for cancer surveillance using serum biomarkers with application to hepatocellular carcinoma". *Metron*. **77** (2): 67–86. doi:10.1007/s40300-019-00151-8. PMC 6820468. PMID 31708595.
10. **^** Eddy SR (October 1, 1998). "Profile hidden Markov models". *Bioinformatics*. **14** (9): 755–63. doi:10.1093/bioinformatics/14.9.755. PMID 9918945.
11. **^** McClintock BT, Langrock R, Gimenez O, Cam E, Borchers DL, Glennie R, Patterson TA (December 2020). "Uncovering ecological state dynamics with hidden Markov models". *Ecology Letters*. **23** (12): 1878–1903. arXiv:2002.10497. Bibcode:2020EcolL..23.1878M. doi:10.1111/ele.13610. PMC 7702077. PMID 33073921.
12. **^** Zhang W (1988). "Shift-invariant pattern recognition neural network and its optical architecture". *Proceedings of Annual Conference of the Japan Society of Applied Physics*.
13. **^** Zhang W, Itoh K, Tanida J, Ichioka Y (November 1990). "Parallel distributed processing model with local space-invariant interconnections and its optical architecture". *Applied Optics*. **29** (32): 4790–7. Bibcode:1990ApOpt..29.4790Z. doi:10.1364/AO.29.004790. PMID 20577468.
14. ^ ***a*** ***b*** Bishop, Christopher M. (August 17, 2006). *Pattern Recognition and Machine Learning*. New York: Springer. ISBN 978-0-387-31073-2.
15. **^** Fukushima K (2007). "Neocognitron". *Scholarpedia*. **2** (1): 1717. Bibcode:2007SchpJ...2.1717F. doi:10.4249/scholarpedia.1717.
16. **^** Hubel DH, Wiesel TN (March 1968). "Receptive fields and functional architecture of monkey striate cortex". *The Journal of Physiology*. **195** (1): 215–43. doi:10.1113/jphysiol.1968.sp008455. PMC 1557912. PMID 4966457.
17. **^** Fukushima K (1980). "Neocognitron: a self organizing neural network model for a mechanism of pattern recognition unaffected by shift in position". *Biological Cybernetics*. **36** (4): 193–202. doi:10.1007/BF00344251. PMID 7370364. S2CID 206775608.
18. **^** Matsugu M, Mori K, Mitari Y, Kaneda Y (2003). "Subject independent facial expression recognition with robust face detection using a convolutional neural network". *Neural Networks*. **16** (5–6): 555–9. doi:10.1016/S0893-6080(03)00115-1. PMID 12850007.
19. **^** Fioravanti D, Giarratano Y, Maggio V, Agostinelli C, Chierici M, Jurman G, Furlanello C (March 2018). "Phylogenetic convolutional neural networks in metagenomics". *BMC Bioinformatics*. **19** (Suppl 2): 49. doi:10.1186/s12859-018-2033-5. PMC 5850953. PMID 29536822.
20. **^** Ji, Yanrong; Zhou, Zhihan; Liu, Han; Davuluri, Ramana V (August 9, 2021). Kelso, Janet (ed.). "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome". *Bioinformatics*. **37** (15): 2112–2120. doi:10.1093/bioinformatics/btab083. ISSN 1367-4803. PMID 33538820.
21. **^** Gündüz, Hüseyin Anil; Binder, Martin; To, Xiao-Yin; Mreches, René; Bischl, Bernd; McHardy, Alice C.; Münch, Philipp C.; Rezaei, Mina (September 11, 2023). "A self-supervised deep learning method for data-efficient training in genomics". *Communications Biology*. **6** (1): 1–12. doi:10.1038/s42003-023-05310-2. ISSN 2399-3642. PMID 37696966.
22. **^** Ho TK (1995). *Random Decision Forests*. Proceedings of the 3rd International Conference on Document Analysis and Recognition, Montreal, QC, 14–16 August 1995. pp. 278–282.
23. **^** Dietterich T (2000). *An Experimental Comparison of Three Methodsfor Constructing Ensembles of Decision Trees:Bagging, Boosting, and Randomization*. Kluwer Academic Publishers. pp. 139–157.
24. **^** Breiman, Leo (2001). "Radom Forests". *Machine Learning*. **45** (1): 5–32. Bibcode:2001MachL..45....5B. doi:10.1023/A:1010933404324. S2CID 89141.
25. ^ ***a*** ***b*** Zhang C, Ma Y (2012). *Ensemble machine learning: methods and applications*. New York: Springer New York Dordrecht Heidelberg London. pp. 157–175. ISBN 978-1-4419-9325-0.
26. **^** Karim MR, Beyan O, Zappa A, Costa IG, Rebholz-Schuhmann D, Cochez M, Decker S (January 2021). "Deep learning-based clustering approaches for bioinformatics". *Briefings in Bioinformatics*. **22** (1): 393–415. doi:10.1093/bib/bbz170. PMC 7820885. PMID 32008043.
27. **^** Lorbeer B, Kosareva A, Deva B, Softić D, Ruppel P, Küpper A (March 1, 2018). "Variations on the Clustering Algorithm BIRCH". *Big Data Research*. **11**: 44–53. doi:10.1016/j.bdr.2017.09.002.
28. **^** Navarro-Muñoz JC, Selem-Mojica N, Mullowney MW, Kautsar SA, Tryon JH, Parkinson EI, et al. (January 2020). "A computational framework to explore large-scale biosynthetic diversity". *Nature Chemical Biology*. **16** (1): 60–68. doi:10.1038/s41589-019-0400-9. PMC 6917865. PMID 31768033.
29. ^ ***a*** ***b*** ***c*** ***d*** ***e*** Shastry KA, Sanjay HA (2020). "Machine Learning for Bioinformatics". *Statistical Modelling and Machine Learning Principles for Bioinformatics Techniques, Tools, and Applications*. Algorithms for Intelligent Systems. Springer Singapore. pp. 25–39. doi:10.1007/978-981-15-2445-5\_3. ISBN 978-981-15-2444-8. S2CID 214350490.
30. **^** Libbrecht MW, Noble WS (June 2015). "Machine learning applications in genetics and genomics". *Nature Reviews. Genetics*. **16** (6): 321–32. doi:10.1038/nrg3920. PMC 5204302. PMID 25948244.
31. **^** Degroeve S, De Baets B, Van de Peer Y, Rouzé P (2002). "Feature subset selection for splice site prediction". *Bioinformatics*. **18** (Suppl 2): S75-83. doi:10.1093/bioinformatics/18.suppl\_2.s75. PMID 12385987.
32. **^** Huang S, Cai N, Pacheco PP, Narrandes S, Wang Y, Xu W (January 2018). "Applications of Support Vector Machine (SVM) Learning in Cancer Genomics". *Cancer Genomics & Proteomics*. **15** (1): 41–51. doi:10.21873/cgp.20063. PMC 5822181. PMID 29275361.
33. **^** Angermueller C, Pärnamaa T, Parts L, Stegle O (July 2016). "Deep learning for computational biology". *Molecular Systems Biology*. **12** (7): 878. doi:10.15252/msb.20156651. PMC 4965871. PMID 27474269.
34. **^** Cao C, Liu F, Tan H, Song D, Shu W, Li W, et al. (February 2018). "Deep Learning and Its Applications in Biomedicine". *Genomics, Proteomics & Bioinformatics*. **16** (1): 17–32. doi:10.1016/j.gpb.2017.07.003. PMC 6000200. PMID 29522900.
35. **^** Zou J, Huss M, Abid A, Mohammadi P, Torkamani A, Telenti A (January 2019). "A primer on deep learning in genomics". *Nature Genetics*. **51** (1): 12–18. doi:10.1038/s41588-018-0295-5. PMID 30478442. S2CID 205572042.
36. **^** Zeng Z, Shi H, Wu Y, Hong Z (2015). "Survey of Natural Language Processing Techniques in Bioinformatics". *Computational and Mathematical Methods in Medicine*. **2015** (D1): 674296. doi:10.1155/2015/674296. PMC 4615216. PMID 26525745.
37. **^** Zeng Z, Shi H, Wu Y, Hong Z (2012). "Survey of Natural Language Processing Techniques in Bioinformatics". *Computational and Mathematical Methods in Medicine*. **2015** (D1): 674296. doi:10.1016/B978-0-12-385467-4.00006-3. PMC 4615216. PMID 26525745.
38. **^** Zeng Z, Shi H, Wu Y, Hong Z (2017). "Survey of Natural Language Processing Techniques in Bioinformatics". *Computational and Mathematical Methods in Medicine*. **2015** (D1): 674296. doi:10.1155/2015/674296. PMC 4615216. PMID 26525745.
39. **^** "GenBank and WGS Statistics". *www.ncbi.nlm.nih.gov*. Retrieved November 25, 2023.
40. ^ ***a*** ***b*** ***c*** ***d*** Mathé C, Sagot MF, Schiex T, Rouzé P (October 2002). "Current methods of gene prediction, their strengths and weaknesses". *Nucleic Acids Research*. **30** (19): 4103–17. doi:10.1093/nar/gkf543. PMC 140543. PMID 12364589.
41. **^** Pratas D, Silva RM, Pinho AJ, Ferreira PJ (May 2015). "An alignment-free method to find and visualise rearrangements between pairs of DNA sequences". *Scientific Reports*. **5** (10203): 10203. Bibcode:2015NatSR...510203P. doi:10.1038/srep10203. PMC 4434998. PMID 25984837.
42. **^** Pauling L, Corey RB, Branson HR (April 1951). "The structure of proteins; two hydrogen-bonded helical configurations of the polypeptide chain". *Proceedings of the National Academy of Sciences of the United States of America*. **37** (4): 205–11. Bibcode:1951PNAS...37..205P. doi:10.1073/pnas.37.4.205. PMC 1063337. PMID 14816373.
43. ^ ***a*** ***b*** Wang S, Peng J, Ma J, Xu J (January 2016). "Protein Secondary Structure Prediction Using Deep Convolutional Neural Fields". *Scientific Reports*. **6**: 18962. arXiv:1512.00843. Bibcode:2016NatSR...618962W. doi:10.1038/srep18962. PMC 4707437. PMID 26752681.
44. **^** Riesenfeld CS, Schloss PD, Handelsman J (2004). "Metagenomics: genomic analysis of microbial communities". *Annual Review of Genetics*. **38** (1): 525–52. doi:10.1146/annurev.genet.38.072902.091216. PMID 15568985.
45. **^** Soueidan, Hayssam; Nikolski, Macha (January 1, 2017). "Machine learning for metagenomics: methods and tools". *Metagenomics*. **1** (1). doi:10.1515/metgen-2016-0001. ISSN 2449-7657. S2CID 17418188.
46. ^ ***a*** ***b*** ***c*** Lin Y, Wang G, Yu J, Sung JJ (April 2021). "Artificial intelligence and metagenomics in intestinal diseases". *Journal of Gastroenterology and Hepatology*. **36** (4): 841–847. doi:10.1111/jgh.15501. PMID 33880764. S2CID 233312307.
47. ^ ***a*** ***b*** ***c*** Dang T, Kishino H (January 2020). "Detecting significant components of microbiomes by random forest with forward variable selection and phylogenetics". bioRxiv 10.1101/2020.10.29.361360.
48. **^** Fioravanti D, Giarratano Y, Maggio V, Agostinelli C, Chierici M, Jurman G, Furlanello C (March 2018). "Phylogenetic convolutional neural networks in metagenomics". *BMC Bioinformatics*. **19** (Suppl 2): 49. doi:10.1186/s12859-018-2033-5. PMC 5850953. PMID 29536822.
49. **^** Dhungel E, Mreyoud Y, Gwak HJ, Rajeh A, Rho M, Ahn TH (January 2021). "MegaR: an interactive R package for rapid sample classification and phenotype prediction using metagenome profiles and machine learning". *BMC Bioinformatics*. **22** (1): 25. doi:10.1186/s12859-020-03933-4. PMC 7814621. PMID 33461494.
50. **^** Xun W, Liu Y, Li W, Ren Y, Xiong W, Xu Z, et al. (January 2021). "Specialized metabolic functions of keystone taxa sustain soil microbiome stability". *Microbiome*. **9** (1): 35. doi:10.1186/s40168-020-00985-9. PMC 7849160. PMID 33517892.
51. ^ ***a*** ***b*** Pirooznia M, Yang JY, Yang MQ, Deng Y (2008). "A comparative study of different machine learning methods on microarray gene expression data". *BMC Genomics*. 9 Suppl 1 (1): S13. doi:10.1186/1471-2164-9-S1-S13. PMC 2386055. PMID 18366602.
52. **^** "Machine Learning in Molecular Systems Biology". *Frontiers*. Retrieved June 9, 2017.
53. **^** d'Alché-Buc F, Wehenkel L (December 2008). "Machine learning in systems biology". *BMC Proceedings*. 2 Suppl 4 (4): S1. doi:10.1186/1753-6561-2-S4-S1. PMC 2654969. PMID 19091048.
54. **^** Bhattacharya M (2020). "Unsupervised Techniques in Genomics". In Srinivasa MG, Siddesh GM, MAnisekhar SR (eds.). *Statistical Modelling and Machine Learning Principles for Bioinformatics Techniques, Tools, and Applications*. Springer Singapore. pp. 164–188. ISBN 978-981-15-2445-5.
55. **^** Topol EJ (January 2019). "High-performance medicine: the convergence of human and artificial intelligence". *Nature Medicine*. **25** (1): 44–56. doi:10.1038/s41591-018-0300-7. PMID 30617339. S2CID 57574615.
56. ^ ***a*** ***b*** Jiang F, Jiang Y, Zhi H, Dong Y, Li H, Ma S, et al. (December 2017). "Artificial intelligence in healthcare: past, present and future". *Stroke and Vascular Neurology*. **2** (4): 230–243. doi:10.1136/svn-2017-000101. PMC 5829945. PMID 29507784.
57. **^** Mirtskhulava L, Wong J, Al-Majeed S, Pearce G (March 2015). "Artificial Neural Network Model in Stroke Diagnosis" (PDF). *2015 17th UKSim-AMSS International Conference on Modelling and Simulation (UKSim)*. pp. 50–53. doi:10.1109/UKSim.2015.33. ISBN 978-1-4799-8713-9. S2CID 6391733.
58. **^** Titano JJ, Badgeley M, Schefflein J, Pain M, Su A, Cai M, et al. (September 2018). "Automated deep-neural-network surveillance of cranial images for acute neurologic events". *Nature Medicine*. **24** (9): 1337–1341. doi:10.1038/s41591-018-0147-y. PMID 30104767. S2CID 51976344.
59. ^ ***a*** ***b*** ***c*** Krallinger M, Erhardt RA, Valencia A (March 2005). "Text-mining approaches in molecular biology and biomedicine". *Drug Discovery Today*. **10** (6): 439–45. doi:10.1016/S1359-6446(05)03376-3. PMID 15808823.
60. **^** Pratas D, Hosseini M, Silva R, Pinho A, Ferreira P (June 20–23, 2017). "Visualization of Distinct DNA Regions of the Modern Human Relatively to a Neanderthal Genome". *Pattern Recognition and Image Analysis*. Lecture Notes in Computer Science. Vol. 10255. pp. 235–242. doi:10.1007/978-3-319-58838-4\_26. ISBN 978-3-319-58837-7.
61. **^** Bardgett RD, Caruso T (March 2020). "Soil microbial community responses to climate extremes: resistance, resilience and transitions to alternative states". *Philosophical Transactions of the Royal Society of London. Series B, Biological Sciences*. **375** (1794): 20190112. doi:10.1098/rstb.2019.0112. PMC 7017770. PMID 31983338.
62. **^** Deveau A, Bonito G, Uehling J, Paoletti M, Becker M, Bindschedler S, et al. (May 2018). "Bacterial-fungal interactions: ecology, mechanisms and challenges". *FEMS Microbiology Reviews*. **42** (3): 335–352. doi:10.1093/femsre/fuy008. PMID 29471481.
63. **^** Ansari MZ, Yadav G, Gokhale RS, Mohanty D (July 2004). "NRPS-PKS: a knowledge-based resource for analysis of NRPS/PKS megasynthases". *Nucleic Acids Research*. **32** (Web Server issue): W405-13. doi:10.1093/nar/gkh359. PMC 441497. PMID 15215420.
64. ^ ***a*** ***b*** Navarro-Muñoz JC, Selem-Mojica N, Mullowney MW, Kautsar SA, Tryon JH, Parkinson EI, et al. (January 2020). "A computational framework to explore large-scale biosynthetic diversity". *Nature Chemical Biology*. **16** (1): 60–68. doi:10.1038/s41589-019-0400-9. PMC 6917865. PMID 31768033.
65. ^ ***a*** ***b*** Doroghazi JR, Albright JC, Goering AW, Ju KS, Haines RR, Tchalukov KA, et al. (November 2014). "A roadmap for natural product discovery based on large-scale genomics and metabolomics". *Nature Chemical Biology*. **10** (11): 963–8. doi:10.1038/nchembio.1659. PMC 4201863. PMID 25262415.
66. ^ ***a*** ***b*** Cimermancic P, Medema MH, Claesen J, Kurita K, Wieland Brown LC, Mavrommatis K, et al. (July 2014). "Insights into secondary metabolism from a global analysis of prokaryotic biosynthetic gene clusters". *Cell*. **158** (2): 412–421. doi:10.1016/j.cell.2014.06.034. PMC 4123684. PMID 25036635.
67. ^ ***a*** ***b*** Goering AW, McClure RA, Doroghazi JR, Albright JC, Haverland NA, Zhang Y, et al. (February 2016). "Metabologenomics: Correlation of Microbial Gene Clusters with Metabolites Drives Discovery of a Nonribosomal Peptide with an Unusual Amino Acid Monomer". *ACS Central Science*. **2** (2): 99–108. doi:10.1021/acscentsci.5b00331. PMC 4827660. PMID 27163034.
68. **^** Amiri Moghaddam J, Crüsemann M, Alanjary M, Harms H, Dávila-Céspedes A, Blom J, et al. (November 2018). "Analysis of the Genome and Metabolome of Marine Myxobacteria Reveals High Potential for Biosynthesis of Novel Specialized Metabolites". *Scientific Reports*. **8** (1): 16600. Bibcode:2018NatSR...816600A. doi:10.1038/s41598-018-34954-y. PMC 6226438. PMID 30413766.
69. **^** Duncan KR, Crüsemann M, Lechner A, Sarkar A, Li J, Ziemert N, et al. (April 2015). "Molecular networking and pattern-based genome mining improves discovery of biosynthetic gene clusters and their products from Salinispora species". *Chemistry & Biology*. **22** (4): 460–471. doi:10.1016/j.chembiol.2015.03.010. PMC 4409930. PMID 25865308.
70. **^** Nielsen JC, Grijseels S, Prigent S, Ji B, Dainat J, Nielsen KF, et al. (April 2017). "Global analysis of biosynthetic gene clusters reveals vast potential of secondary metabolite production in Penicillium species". *Nature Microbiology*. **2** (6): 17044. doi:10.1038/nmicrobiol.2017.44. PMID 28368369. S2CID 22699928.
71. **^** McClure RA, Goering AW, Ju KS, Baccile JA, Schroeder FC, Metcalf WW, et al. (December 2016). "Elucidating the Rimosamide-Detoxin Natural Product Families and Their Biosynthesis Using Metabolite/Gene Cluster Correlations". *ACS Chemical Biology*. **11** (12): 3452–3460. doi:10.1021/acschembio.6b00779. PMC 5295535. PMID 27809474.
72. **^** Cao L, Shcherbin E, Mohimani H (August 2019). "A Metabolome- and Metagenome-Wide Association Network Reveals Microbial Natural Products and Microbial Biotransformation Products from the Human Microbiota". *mSystems*. **4** (4): e00387–19, /msystems/4/4/msys.00387–19.atom. doi:10.1128/mSystems.00387-19. PMC 6712304. PMID 31455639.
73. **^** Olm MR, Bhattacharya N, Crits-Christoph A, Firek BA, Baker R, Song YS, et al. (December 2019). "Necrotizing enterocolitis is preceded by increased gut bacterial replication, *Klebsiella*, and fimbriae-encoding bacteria". *Science Advances*. **5** (12): eaax5727. Bibcode:2019SciA....5.5727O. doi:10.1126/sciadv.aax5727. PMC 6905865. PMID 31844663.
74. **^** Carrión VJ, Perez-Jaramillo J, Cordovez V, Tracanna V, de Hollander M, Ruiz-Buck D, et al. (November 2019). "Pathogen-induced activation of disease-suppressive functions in the endophytic root microbiome". *Science*. **366** (6465): 606–612. Bibcode:2019Sci...366..606C. doi:10.1126/science.aaw9285. PMID 31672892. S2CID 207814746.
75. **^** Pascal Andreu, Victória; Augustijn, Hannah E.; van den Berg, Koen; van der Hooft, Justin J. J.; Fischbach, Michael A.; Medema, Marnix H. (October 26, 2021). Shank, Elizabeth Anne (ed.). "BiG-MAP: an Automated Pipeline To Profile Metabolic Gene Cluster Abundance and Expression in Microbiomes". *mSystems*. **6** (5): e0093721. doi:10.1128/mSystems.00937-21. ISSN 2379-5077. PMC 8547482. PMID 34581602.
76. **^** Agrawal P, Khater S, Gupta M, Sain N, Mohanty D (July 2017). "RiPPMiner: a bioinformatics resource for deciphering chemical structures of RiPPs based on prediction of cleavage and cross-links". *Nucleic Acids Research*. **45** (W1): W80–W88. doi:10.1093/nar/gkx408. PMC 5570163. PMID 28499008.
77. **^** Huber F, Ridder L, Verhoeven S, Spaaks JH, Diblen F, Rogers S, van der Hooft JJ (February 2021). "Spec2Vec: Improved mass spectral similarity scoring through learning of structural relationships". *PLOS Computational Biology*. **17** (2): e1008724. Bibcode:2021PLSCB..17E8724H. doi:10.1371/journal.pcbi.1008724. PMC 7909622. PMID 33591968.
78. **^** Huber F, Ridder L, Verhoeven S, Spaaks JH, Diblen F, Rogers S, van der Hooft JJ (February 2021). "Spec2Vec: Improved mass spectral similarity scoring through learning of structural relationships". *PLOS Computational Biology*. **17** (2): e1008724. Bibcode:2021PLSCB..17E8724H. doi:10.1371/journal.pcbi.1008724. PMC 7909622. PMID 33591968.
79. **^** National Center for Biotechnology Information; U.S. National Library of Medicine. "National Center for Biotechnology Information". *ncbi.nlm.nih.gov*. Retrieved July 30, 2021.
80. **^** Agarwala R, Barrett T, Beck J, Benson DA, Bollin C, Bolton E, et al. (NCBI Resource Coordinators) (January 2018). "Database resources of the National Center for Biotechnology Information". *Nucleic Acids Research*. **46** (D1): D8–D13. doi:10.1093/nar/gkx1095. PMC 5753372. PMID 29140470.
81. **^** "antiSMASH database". *antismash-db.secondarymetabolites.org*.
82. **^** "MIBiG: Minimum Information about a Biosynthetic Gene cluster". *mibig.secondarymetabolites.org*. Retrieved July 30, 2021.
83. **^** MiBiG
84. **^** Kautsar SA, Blin K, Shaw S, Navarro-Muñoz JC, Terlouw BR, van der Hooft JJ, et al. (January 2020). "MIBiG 2.0: a repository for biosynthetic gene clusters of known function". *Nucleic Acids Research*. **48** (D1): D454–D458. doi:10.1093/nar/gkz882. PMC 7145714. PMID 31612915.
85. **^** "Silva". *arb-silva.de*. Retrieved July 30, 2021.
86. **^** Quast C, Pruesse E, Yilmaz P, Gerken J, Schweer T, Yarza P, et al. (January 2013). "The SILVA ribosomal RNA gene database project: improved data processing and web-based tools". *Nucleic Acids Research*. **41** (Database issue): D590-6. doi:10.1093/nar/gks1219. PMC 3531112. PMID 23193283.
87. **^** "greengenes.secondgenome.com". *greengenes.secondgenome.com*. Retrieved July 30, 2021.
88. **^** DeSantis TZ, Hugenholtz P, Larsen N, Rojas M, Brodie EL, Keller K, et al. (July 2006). "Greengenes, a chimera-checked 16S rRNA gene database and workbench compatible with ARB". *Applied and Environmental Microbiology*. **72** (7): 5069–72. Bibcode:2006ApEnM..72.5069D. doi:10.1128/AEM.03006-05. PMC 1489311. PMID 16820507.
89. **^** McDonald D, Price MN, Goodrich J, Nawrocki EP, DeSantis TZ, Probst A, et al. (March 2012). "An improved Greengenes taxonomy with explicit ranks for ecological and evolutionary analyses of bacteria and archaea". *The ISME Journal*. **6** (3): 610–8. doi:10.1038/ismej.2011.139. PMC 3280142. PMID 22134646.
90. **^** "opentree". *tree.opentreeoflife.org*. Retrieved July 30, 2021.
91. **^** Hinchliff CE, Smith SA, Allman JF, Burleigh JG, Chaudhary R, Coghill LM, et al. (October 2015). "Synthesis of phylogeny and taxonomy into a comprehensive tree of life". *Proceedings of the National Academy of Sciences of the United States of America*. **112** (41): 12764–9. Bibcode:2015PNAS..11212764H. doi:10.1073/pnas.1423041112. PMC 4611642. PMID 26385966.
92. **^** "RDP Release 11 -- Sequence Analysis Tools". *rdp.cme.msu.edu*. Retrieved July 30, 2021.
93. **^** Cole JR, Wang Q, Fish JA, Chai B, McGarrell DM, Sun Y, et al. (January 2014). "Ribosomal Database Project: data and tools for high throughput rRNA analysis". *Nucleic Acids Research*. **42** (Database issue): D633-42. doi:10.1093/nar/gkt1244. PMC 3965039. PMID 24288368.

  






| * v
* t
* e
Differentiable computing |
| --- |
| General | 
* **Differentiable programming**
* Information geometry
* Statistical manifold
* Automatic differentiation
* Neuromorphic engineering
* Pattern recognition
* Tensor calculus
* Computational learning theory
* Inductive bias


 |
| Concepts | 
* Gradient descent
	+ SGD
* Clustering
* Regression
	+ Overfitting
* Hallucination
* Adversary
* Attention
* Convolution
* Loss functions
* Backpropagation
* Batchnorm
* Activation
	+ Softmax
	+ Sigmoid
	+ Rectifier
* Regularization
* Datasets
	+ Augmentation
* Diffusion
* Autoregression


 |
| Applications | 
* Machine learning
	+ In-context learning
* Artificial neural network
	+ Deep learning
* Scientific computing
* Artificial Intelligence
* Language model
	+ Large language model


 |
| Hardware | 
* IPU
* TPU
* VPU
* Memristor
* SpiNNaker


 |
| Software libraries | 
* TensorFlow
* PyTorch
* Keras
* Theano
* JAX
* Flux.jl


 |
| Implementations | 

|  |  |
| --- | --- |
| Audio–visual | 
* AlexNet
* WaveNet
* Human image synthesis
* HWR
* OCR
* Speech synthesis
* Speech recognition
* Facial recognition
* AlphaFold
* DALL-E
* Midjourney
* Stable Diffusion
* Whisper


 |
| Verbal | 
* Word2vec
* Seq2seq
* BERT
* Gemini
* LaMDA
	+ Bard
* NMT
* Project Debater
* IBM Watson
* GPT-1
* GPT-2
* GPT-3
* GPT-4
* ChatGPT
* GPT-J
* Chinchilla AI
* PaLM
* BLOOM
* LLaMA


 |
| Decisional | 
* AlphaGo
* AlphaZero
* Q-learning
* SARSA
* OpenAI Five
* Self-driving car
* MuZero
* Action selection
	+ Auto-GPT
* Robot control


 |

 |
| People | 
* Yoshua Bengio
* Alex Graves
* Ian Goodfellow
* Stephen Grossberg
* Demis Hassabis
* Geoffrey Hinton
* Yann LeCun
* Fei-Fei Li
* Andrew Ng
* Jürgen Schmidhuber
* David Silver
* Ilya Sutskever


 |
| Organizations | 
* Anthropic
* EleutherAI
* Google DeepMind
* Hugging Face
* OpenAI
* Meta AI
* Mila
* MIT CSAIL


 |
| Architectures | 
* Neural Turing machine
* Differentiable neural computer
* Transformer
* Recurrent neural network (RNN)
* Long short-term memory (LSTM)
* Gated recurrent unit (GRU)
* Echo state network
* Multilayer perceptron (MLP)
* Convolutional neural network
* Residual neural network
* Autoencoder
* Variational autoencoder (VAE)
* Generative adversarial network (GAN)
* Graph neural network


 |
| 
* ![](//upload.wikimedia.org/wikipedia/en/thumb/e/e2/Symbol_portal_class.svg/16px-Symbol_portal_class.svg.png) Portals
	+ Computer programming
	+ Technology
* Categories
	+ Artificial neural networks
	+ Machine learning


 |





![](https://login.wikimedia.org/wiki/Special:CentralAutoLogin/start?type=1x1)
Retrieved from "https://en.wikipedia.org/w/index.php?title=Machine\_learning\_in\_bioinformatics&oldid=1189937179"
Categories: * Machine learning
* Bioinformatics
Hidden categories: * CS1: long volume value
* Use mdy dates from August 2022
* All articles needing additional references
* Articles needing additional references from June 2021
* All articles with specifically marked weasel-worded phrases
* Articles with specifically marked weasel-worded phrases from June 2021
* Articles containing potentially dated statements from 2002
* All articles containing potentially dated statements
* Wikipedia articles in need of updating from June 2021
* All Wikipedia articles in need of updating
* All articles lacking reliable references
* Articles lacking reliable references from November 2023






* This page was last edited on 14 December 2023, at 23:27 (UTC).
* Text is available under the Creative Commons Attribution-ShareAlike License 4.0;
additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.


* Privacy policy
* About Wikipedia
* Disclaimers
* Contact Wikipedia
* Code of Conduct
* Developers
* Statistics
* Cookie statement
* Mobile view


* ![Wikimedia Foundation](/static/images/footer/wikimedia-button.png)
* ![Powered by MediaWiki](/static/images/footer/poweredby_mediawiki_88x31.png)





