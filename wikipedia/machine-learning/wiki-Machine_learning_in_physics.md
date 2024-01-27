


















Contents
--------


move to sidebar
hide

* (Top)
* 1Applications of machine learning to physics



Toggle Applications of machine learning to physics subsection

	+ 1.1Noisy data
	+ 1.2Calculated and noise-free data
	+ 1.3Variational circuits
	+ 1.4Sign problem
	+ 1.5Fluid dynamics
	+ 1.6Physics discovery and prediction
* 2See also
* 3References














Toggle the table of contents







Machine learning in physics
===========================





6 languages



* বাংলা
* Català
* Ελληνικά
* Español
* فارسی
* Українська


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


Applications of machine learning to quantum physics
This article is about classical machine learning of quantum systems. For machine learning enhanced by quantum computation, see quantum machine learning.


|  |
| --- |
| Part of a series of articles about |
| Quantum mechanics |
| 



i
ℏ


∂

∂
t




|

ψ
(
t
)
⟩
=



H
^




|

ψ
(
t
)
⟩


{\displaystyle i\hbar {\frac {\partial }{\partial t}}|\psi (t)\rangle ={\hat {H}}|\psi (t)\rangle }

{\displaystyle i\hbar {\frac {\partial }{\partial t}}|\psi (t)\rangle ={\hat {H}}|\psi (t)\rangle }Schrödinger equation |
| * Introduction
* Glossary
* History
 |
| Background
* Classical mechanics
* Old quantum theory
* Bra–ket notation



* Hamiltonian
* Interference


 |
| Fundamentals
* Complementarity
* Decoherence
* Entanglement
* Energy level
* Measurement
* Nonlocality
* Quantum number
* State
* Superposition
* Symmetry
* Tunnelling
* Uncertainty
* Wave function
	+ Collapse


 |
| Experiments
* Bell's inequality
* Davisson–Germer
* Double-slit
* Elitzur–Vaidman
* Franck–Hertz
* Leggett–Garg inequality
* Mach–Zehnder
* Popper



* Quantum eraser
	+ Delayed-choice



* Schrödinger's cat
* Stern–Gerlach
* Wheeler's delayed-choice


 |
| Formulations
* Overview



* Heisenberg
* Interaction
* Matrix
* Phase-space
* Schrödinger
* Sum-over-histories (path integral)


 |
| Equations
* Dirac
* Klein–Gordon
* Pauli
* Rydberg
* Schrödinger


 |
| Interpretations
* Bayesian
* Consistent histories
* Copenhagen
* de Broglie–Bohm
* Ensemble
* Hidden-variable
	+ Local
		- Superdeterminism
* Many-worlds
* Objective collapse
* Quantum logic
* Relational
* Transactional
* Von Neumann–Wigner


 |
| Advanced topics
* Relativistic quantum mechanics
* Quantum field theory
* Quantum information science
* Quantum computing
* Quantum chaos
* EPR paradox
* Density matrix
* Scattering theory
* Quantum statistical mechanics
* Quantum machine learning
 |
| Scientists
* Aharonov
* Bell
* Bethe
* Blackett
* Bloch
* Bohm
* Bohr
* Born
* Bose
* de Broglie
* Compton
* Dirac
* Davisson
* Debye
* Ehrenfest
* Einstein
* Everett
* Fock
* Fermi
* Feynman
* Glauber
* Gutzwiller
* Heisenberg
* Hilbert
* Jordan
* Kramers
* Pauli
* Lamb
* Landau
* Laue
* Moseley
* Millikan
* Onnes
* Planck
* Rabi
* Raman
* Rydberg
* Schrödinger
* Simmons
* Sommerfeld
* von Neumann
* Weyl
* Wien
* Wigner
* Zeeman
* Zeilinger


 |
| * v
* t
* e
 |


Applying classical methods of machine learning to the study of quantum systems is the focus of an emergent area of physics research. A basic example of this is quantum state tomography, where a quantum state is learned from measurement.[1] Other examples include learning Hamiltonians,[2][3] learning quantum phase transitions,[4][5] and automatically generating new quantum experiments.[6][7][8][9] Classical machine learning is effective at processing large amounts of experimental or calculated data in order to characterize an unknown quantum system, making its application useful in contexts including quantum information theory, quantum technologies development, and computational materials design. In this context, it can be used for example as a tool to interpolate pre-calculated interatomic potentials[10] or directly solving the Schrödinger equation with a variational method.[11]




Applications of machine learning to physics[edit]
-------------------------------------------------


### Noisy data[edit]


The ability to experimentally control and prepare increasingly complex quantum systems brings with it a growing need to turn large and noisy data sets into meaningful information. This is a problem that has already been studied extensively in the classical setting, and consequently, many existing machine learning techniques can be naturally adapted to more efficiently address experimentally relevant problems. For example, Bayesian methods and concepts of algorithmic learning can be fruitfully applied to tackle quantum state classification,[12] Hamiltonian learning,[13] and the characterization of an unknown unitary transformation.[14][15] Other problems that have been addressed with this approach are given in the following list:



* Identifying an accurate model for the dynamics of a quantum system, through the reconstruction of the Hamiltonian;[16][17][18]
* Extracting information on unknown states;[19][20][21][12][22][1]
* Learning unknown unitary transformations and measurements;[14][15]
* Engineering of quantum gates from qubit networks with pairwise interactions, using time dependent[23] or independent[24] Hamiltonians.
* Improving the extraction accuracy of physical observables from absorption images of ultracold atoms (degenerate Fermi gas), by the generation of an ideal reference frame.[25]


### Calculated and noise-free data[edit]


Quantum machine learning can also be applied to dramatically accelerate the prediction of quantum properties of molecules and materials.[26] This can be helpful for the computational design of new molecules or materials. Some examples include



* Interpolating interatomic potentials;[27]
* Inferring molecular atomization energies throughout chemical compound space;[28]
* Accurate potential energy surfaces with restricted Boltzmann machines;[29]
* Automatic generation of new quantum experiments;[6][7]
* Solving the many-body, static and time-dependent Schrödinger equation;[11]
* Identifying phase transitions from entanglement spectra;[30]
* Generating adaptive feedback schemes for quantum metrology and quantum tomography.[31][32]


### Variational circuits[edit]


Variational circuits are a family of algorithms which utilize training based on circuit parameters and an objective function.[33] Variational circuits are generally composed of a classical device communicating input parameters (random or pre-trained parameters) into a quantum device, along with a classical Mathematical optimization function. These circuits are very heavily dependent on the architecture of the proposed quantum device because parameter adjustments are adjusted based solely on the classical components within the device.[34] Though the application is considerably infantile in the field of quantum machine learning, it has incredibly high promise for more efficiently generating efficient optimization functions.



### Sign problem[edit]


Machine learning techniques can be used to find a better manifold of integration for path integrals in order to avoid the sign problem.[35]



### Fluid dynamics[edit]


This section is an excerpt from Deep learning § Partial differential equations.[edit]
Physics informed neural networks have been used to solve partial differential equations in both forward and inverse problems in a data driven manner.[36] One example is the reconstructing fluid flow governed by the Navier-Stokes equations. Using physics informed neural networks does not require the often expensive mesh generation that conventional CFD methods relies on.[37][38]
### Physics discovery and prediction[edit]


See also: Laboratory robotics
![](//upload.wikimedia.org/wikipedia/commons/thumb/2/24/An_AI_learns_basic_physical_principles.webp/220px-An_AI_learns_basic_physical_principles.webp.png)Illustration of how an AI learns the basic fundamental physical concept of 'unchangeableness'[39]
A deep learning system was reported to learn intuitive physics from visual data (of virtual 3D environments) based on an unpublished approach inspired by studies of visual cognition in infants.[40][39] Other researchers have developed a machine learning algorithm that could discover sets of basic variables of various physical systems and predict the systems' future dynamics from video recordings of their behavior.[41][42] In the future, it may be possible that such can be used to automate the discovery of physical laws of complex systems.[41] Beyond discovery and prediction, "blank slate"-type of learning of fundamental aspects of the physical world may have further applications such as improving adaptive and broad artificial general intelligence.[*additional citation(s) needed*] In specific, prior machine learning models were "highly specialised and lack a general understanding of the world".[40]



See also[edit]
--------------


* Quantum computing
* Quantum machine learning
* Quantum algorithm for linear systems of equations
* Quantum annealing
* Quantum neural network


References[edit]
----------------



1. ^ ***a*** ***b*** Torlai, Giacomo; Mazzola, Guglielmo; Carrasquilla, Juan; Troyer, Matthias; Melko, Roger; Carleo, Giuseppe (May 2018). "Neural-network quantum state tomography". *Nature Physics*. **14** (5): 447–450. arXiv:1703.05334. Bibcode:2018NatPh..14..447T. doi:10.1038/s41567-018-0048-5. ISSN 1745-2481. S2CID 125415859.
2. **^** Cory, D. G.; Wiebe, Nathan; Ferrie, Christopher; Granade, Christopher E. (2012-07-06). "Robust Online Hamiltonian Learning". *New Journal of Physics*. **14** (10): 103013. arXiv:1207.1655. Bibcode:2012NJPh...14j3013G. doi:10.1088/1367-2630/14/10/103013. S2CID 9928389.
3. **^** Cao, Chenfeng; Hou, Shi-Yao; Cao, Ningping; Zeng, Bei (2020-02-10). "Supervised learning in Hamiltonian reconstruction from local measurements on eigenstates". *Journal of Physics: Condensed Matter*. **33** (6): 064002. arXiv:2007.05962. doi:10.1088/1361-648x/abc4cf. ISSN 0953-8984. PMID 33105109. S2CID 220496757.
4. **^** Broecker, Peter; Assaad, Fakher F.; Trebst, Simon (2017-07-03). "Quantum phase recognition via unsupervised machine learning". arXiv:1707.00663 [cond-mat.str-el].
5. **^** Huembeli, Patrick; Dauphin, Alexandre; Wittek, Peter (2018). "Identifying Quantum Phase Transitions with Adversarial Neural Networks". *Physical Review B*. **97** (13): 134109. arXiv:1710.08382. Bibcode:2018PhRvB..97m4109H. doi:10.1103/PhysRevB.97.134109. ISSN 2469-9950. S2CID 125593239.
6. ^ ***a*** ***b*** Krenn, Mario (2016-01-01). "Automated Search for new Quantum Experiments". *Physical Review Letters*. **116** (9): 090405. arXiv:1509.02749. Bibcode:2016PhRvL.116i0405K. doi:10.1103/PhysRevLett.116.090405. PMID 26991161. S2CID 20182586.
7. ^ ***a*** ***b*** Knott, Paul (2016-03-22). "A search algorithm for quantum state engineering and metrology". *New Journal of Physics*. **18** (7): 073033. arXiv:1511.05327. Bibcode:2016NJPh...18g3033K. doi:10.1088/1367-2630/18/7/073033. S2CID 2721958.
8. **^** Dunjko, Vedran; Briegel, Hans J (2018-06-19). "Machine learning & artificial intelligence in the quantum domain: a review of recent progress". *Reports on Progress in Physics*. **81** (7): 074001. arXiv:1709.02779. Bibcode:2018RPPh...81g4001D. doi:10.1088/1361-6633/aab406. hdl:1887/71084. ISSN 0034-4885. PMID 29504942. S2CID 3681629.
9. **^** Melnikov, Alexey A.; Nautrup, Hendrik Poulsen; Krenn, Mario; Dunjko, Vedran; Tiersch, Markus; Zeilinger, Anton; Briegel, Hans J. (1221). "Active learning machine learns to create new quantum experiments". *Proceedings of the National Academy of Sciences*. **115** (6): 1221–1226. arXiv:1706.00868. doi:10.1073/pnas.1714936115. ISSN 0027-8424. PMC 5819408. PMID 29348200.
10. **^** Behler, Jörg; Parrinello, Michele (2007-04-02). "Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces". *Physical Review Letters*. **98** (14): 146401. Bibcode:2007PhRvL..98n6401B. doi:10.1103/PhysRevLett.98.146401. PMID 17501293.
11. ^ ***a*** ***b*** Carleo, Giuseppe; Troyer, Matthias (2017-02-09). "Solving the quantum many-body problem with artificial neural networks". *Science*. **355** (6325): 602–606. arXiv:1606.02318. Bibcode:2017Sci...355..602C. doi:10.1126/science.aag2302. PMID 28183973. S2CID 206651104.
12. ^ ***a*** ***b*** Sentís, Gael; Calsamiglia, John; Muñoz-Tapia, Raúl; Bagan, Emilio (2012). "Quantum learning without quantum memory". *Scientific Reports*. **2**: 708. arXiv:1106.2742. Bibcode:2012NatSR...2E.708S. doi:10.1038/srep00708. PMC 3464493. PMID 23050092.
13. **^** Wiebe, Nathan; Granade, Christopher; Ferrie, Christopher; Cory, David (2014). "Quantum Hamiltonian learning using imperfect quantum resources". *Physical Review A*. **89** (4): 042314. arXiv:1311.5269. Bibcode:2014PhRvA..89d2314W. doi:10.1103/physreva.89.042314. hdl:10453/118943. S2CID 55126023.
14. ^ ***a*** ***b*** Bisio, Alessandro; Chiribella, Giulio; D'Ariano, Giacomo Mauro; Facchini, Stefano; Perinotti, Paolo (2010). "Optimal quantum learning of a unitary transformation". *Physical Review A*. **81** (3): 032324. arXiv:0903.0543. Bibcode:2010PhRvA..81c2324B. doi:10.1103/PhysRevA.81.032324. S2CID 119289138.
15. ^ ***a*** ***b*** Jeongho; Junghee Ryu, Bang; Yoo, Seokwon; Pawłowski, Marcin; Lee, Jinhyoung (2014). "A strategy for quantum algorithm design assisted by machine learning". *New Journal of Physics*. **16** (1): 073017. arXiv:1304.2169. Bibcode:2014NJPh...16a3017K. doi:10.1088/1367-2630/16/1/013017. S2CID 54494244.
16. **^** Granade, Christopher E.; Ferrie, Christopher; Wiebe, Nathan; Cory, D. G. (2012-10-03). "Robust Online Hamiltonian Learning". *New Journal of Physics*. **14** (10): 103013. arXiv:1207.1655. Bibcode:2012NJPh...14j3013G. doi:10.1088/1367-2630/14/10/103013. ISSN 1367-2630. S2CID 9928389.
17. **^** Wiebe, Nathan; Granade, Christopher; Ferrie, Christopher; Cory, D. G. (2014). "Hamiltonian Learning and Certification Using Quantum Resources". *Physical Review Letters*. **112** (19): 190501. arXiv:1309.0876. Bibcode:2014PhRvL.112s0501W. doi:10.1103/PhysRevLett.112.190501. ISSN 0031-9007. PMID 24877920. S2CID 39126228.
18. **^** Wiebe, Nathan; Granade, Christopher; Ferrie, Christopher; Cory, David G. (2014-04-17). "Quantum Hamiltonian Learning Using Imperfect Quantum Resources". *Physical Review A*. **89** (4): 042314. arXiv:1311.5269. Bibcode:2014PhRvA..89d2314W. doi:10.1103/PhysRevA.89.042314. hdl:10453/118943. ISSN 1050-2947. S2CID 55126023.
19. **^** Sasaki, Madahide; Carlini, Alberto; Jozsa, Richard (2001). "Quantum Template Matching". *Physical Review A*. **64** (2): 022317. arXiv:quant-ph/0102020. Bibcode:2001PhRvA..64b2317S. doi:10.1103/PhysRevA.64.022317. S2CID 43413485.
20. **^** Sasaki, Masahide (2002). "Quantum learning and universal quantum matching machine". *Physical Review A*. **66** (2): 022303. arXiv:quant-ph/0202173. Bibcode:2002PhRvA..66b2303S. doi:10.1103/PhysRevA.66.022303. S2CID 119383508.
21. **^** Sentís, Gael; Guţă, Mădălin; Adesso, Gerardo (2015-07-09). "Quantum learning of coherent states". *EPJ Quantum Technology*. **2** (1): 17. arXiv:1410.8700. doi:10.1140/epjqt/s40507-015-0030-4. ISSN 2196-0763. S2CID 6980007.
22. **^** Lee, Sang Min; Lee, Jinhyoung; Bang, Jeongho (2018-11-02). "Learning unknown pure quantum states". *Physical Review A*. **98** (5): 052302. arXiv:1805.06580. Bibcode:2018PhRvA..98e2302L. doi:10.1103/PhysRevA.98.052302. S2CID 119095806.
23. **^** Zahedinejad, Ehsan; Ghosh, Joydip; Sanders, Barry C. (2016-11-16). "Designing High-Fidelity Single-Shot Three-Qubit Gates: A Machine Learning Approach". *Physical Review Applied*. **6** (5): 054005. arXiv:1511.08862. Bibcode:2016PhRvP...6e4005Z. doi:10.1103/PhysRevApplied.6.054005. ISSN 2331-7019. S2CID 7299645.
24. **^** Banchi, Leonardo; Pancotti, Nicola; Bose, Sougato (2016-07-19). "Quantum gate learning in qubit networks: Toffoli gate without time-dependent control". *npj Quantum Information*. **2**: 16019. Bibcode:2016npjQI...216019B. doi:10.1038/npjqi.2016.19.
25. **^** Ness, Gal; Vainbaum, Anastasiya; Shkedrov, Constantine; Florshaim, Yanay; Sagi, Yoav (2020-07-06). "Single-exposure absorption imaging of ultracold atoms using deep learning". *Physical Review Applied*. **14** (1): 014011. arXiv:2003.01643. Bibcode:2020PhRvP..14a4011N. doi:10.1103/PhysRevApplied.14.014011. S2CID 211817864.
26. **^** von Lilienfeld, O. Anatole (2018-04-09). "Quantum Machine Learning in Chemical Compound Space". *Angewandte Chemie International Edition*. **57** (16): 4164–4169. doi:10.1002/anie.201709686. PMID 29216413.
27. **^** Bartok, Albert P.; Payne, Mike C.; Risi, Kondor; Csanyi, Gabor (2010). "Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons" (PDF). *Physical Review Letters*. **104** (13): 136403. arXiv:0910.1019. Bibcode:2010PhRvL.104m6403B. doi:10.1103/PhysRevLett.104.136403. PMID 20481899. S2CID 15918457.
28. **^** Rupp, Matthias; Tkatchenko, Alexandre; Müller, Klaus-Robert; von Lilienfeld, O. Anatole (2012-01-31). "Fast and Accurate Modeling of Molecular Atomization Energies With Machine Learning". *Physical Review Letters*. **355** (6325): 602. arXiv:1109.2618. Bibcode:2012PhRvL.108e8301R. doi:10.1103/PhysRevLett.108.058301. PMID 22400967. S2CID 321566.
29. **^** Xia, Rongxin; Kais, Sabre (2018-10-10). "Quantum machine learning for electronic structure calculations". *Nature Communications*. **9** (1): 4195. arXiv:1803.10296. Bibcode:2018NatCo...9.4195X. doi:10.1038/s41467-018-06598-z. PMC 6180079. PMID 30305624.
30. **^** van Nieuwenburg, Evert; Liu, Ye-Hua; Huber, Sebastian (2017). "Learning phase transitions by confusion". *Nature Physics*. **13** (5): 435. arXiv:1610.02048. Bibcode:2017NatPh..13..435V. doi:10.1038/nphys4037. S2CID 119285403.
31. **^** Hentschel, Alexander (2010-01-01). "Machine Learning for Precise Quantum Measurement". *Physical Review Letters*. **104** (6): 063603. arXiv:0910.0762. Bibcode:2010PhRvL.104f3603H. doi:10.1103/PhysRevLett.104.063603. PMID 20366821. S2CID 14689659.
32. **^** Quek, Yihui; Fort, Stanislav; Ng, Hui Khoon (2018-12-17). "Adaptive Quantum State Tomography with Neural Networks". arXiv:1812.06693 [quant-ph].
33. **^** "Variational Circuits — Quantum Machine Learning Toolbox 0.7.1 documentation". *qmlt.readthedocs.io*. Retrieved 2018-12-06.
34. **^** Schuld, Maria (2018-06-12). "Quantum Machine Learning 1.0". *XanaduAI*. Retrieved 2018-12-07.
35. **^** Alexandru, Andrei; Bedaque, Paulo F.; Lamm, Henry; Lawrence, Scott (2017). "Deep Learning Beyond Lefschetz Thimbles". *Physical Review D*. **96** (9): 094505. arXiv:1709.01971. Bibcode:2017PhRvD..96i4505A. doi:10.1103/PhysRevD.96.094505. S2CID 119074823.
36. **^** Raissi, M.; Perdikaris, P.; Karniadakis, G. E. (2019-02-01). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations". *Journal of Computational Physics*. **378**: 686–707. Bibcode:2019JCoPh.378..686R. doi:10.1016/j.jcp.2018.10.045. ISSN 0021-9991. OSTI 1595805. S2CID 57379996.
37. **^** Mao, Zhiping; Jagtap, Ameya D.; Karniadakis, George Em (2020-03-01). "Physics-informed neural networks for high-speed flows". *Computer Methods in Applied Mechanics and Engineering*. **360**: 112789. Bibcode:2020CMAME.360k2789M. doi:10.1016/j.cma.2019.112789. ISSN 0045-7825. S2CID 212755458.
38. **^** Raissi, Maziar; Yazdani, Alireza; Karniadakis, George Em (2020-02-28). "Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations". *Science*. **367** (6481): 1026–1030. Bibcode:2020Sci...367.1026R. doi:10.1126/science.aaw4741. PMC 7219083. PMID 32001523.
39. ^ ***a*** ***b*** Piloto, Luis S.; Weinstein, Ari; Battaglia, Peter; Botvinick, Matthew (11 July 2022). "Intuitive physics learning in a deep-learning model inspired by developmental psychology". *Nature Human Behaviour*. **6** (9): 1257–1267. doi:10.1038/s41562-022-01394-8. ISSN 2397-3374. PMC 9489531. PMID 35817932.
40. ^ ***a*** ***b*** "DeepMind AI learns physics by watching videos that don't make sense". *New Scientist*. Retrieved 21 August 2022.
41. ^ ***a*** ***b*** Feldman, Andrey (11 August 2022). "Artificial physicist to unravel the laws of nature". *Advanced Science News*. Retrieved 21 August 2022.
42. **^** Chen, Boyuan; Huang, Kuang; Raghupathi, Sunand; Chandratreya, Ishaan; Du, Qiang; Lipson, Hod (July 2022). "Automated discovery of fundamental variables hidden in experimental data". *Nature Computational Science*. **2** (7): 433–442. doi:10.1038/s43588-022-00281-6. ISSN 2662-8457. S2CID 251087119.



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




| * v
* t
* e
Quantum information science |
| --- |
| General | 
* DiVincenzo's criteria
* NISQ era
* Quantum computing
	+ timeline
* Quantum information
* Quantum programming
* Quantum simulation
* Qubit
	+ physical vs. logical
* Quantum processors
	+ cloud-based


 |
| Theorems | 
* Bell's
* Eastin–Knill
* Gleason's
* Gottesman–Knill
* Holevo's
* No-broadcasting
* No-cloning
* No-communication
* No-deleting
* No-hiding
* No-teleportation
* PBR
* Quantum speed limit
* Threshold
* Solovay–Kitaev
* Purification


 |
| Quantumcommunication | 
* Classical capacity
	+ entanglement-assisted
	+ quantum capacity
* Entanglement distillation
* Monogamy of entanglement
* LOCC
* Quantum channel
	+ quantum network
* Quantum teleportation
	+ quantum gate teleportation
* Superdense coding




|  |  |
| --- | --- |
| Quantum cryptography | 
* Post-quantum cryptography
* Quantum coin flipping
* Quantum money
* Quantum key distribution
	+ BB84
	+ SARG04
	+ other protocols
* Quantum secret sharing


 |


 |
| Quantum algorithms | 
* Amplitude amplification
* Bernstein–Vazirani
* Boson sampling
* Deutsch–Jozsa
* Grover's
* HHL
* Hidden subgroup
* Quantum annealing
* Quantum counting
* Quantum Fourier transform
* Quantum optimization
* Quantum phase estimation
* Shor's
* Simon's
* VQE


 |
| Quantumcomplexity theory | 
* BQP
* EQP
* QIP
* QMA
* PostBQP


 |
| Quantum  processor benchmarks | 
* Quantum supremacy
* Quantum volume
* Randomized benchmarking
	+ XEB
* Relaxation times
	+ *T*1
	+ *T*2


 |
| Quantumcomputing models | 
* Adiabatic quantum computation
* Continuous-variable quantum information
* One-way quantum computer
	+ cluster state
* Quantum circuit
	+ quantum logic gate
* Quantum machine learning
	+ quantum neural network
* Quantum Turing machine
* Topological quantum computer


 |
| Quantumerror correction | 
* Codes
	+ CSS
	+ quantum convolutional
	+ stabilizer
	+ Shor
	+ Bacon–Shor
	+ Steane
	+ Toric
	+ *gnu*
* Entanglement-assisted


 |
| Physicalimplementations | 

|  |  |
| --- | --- |
| Quantum optics | 
* Cavity QED
* Circuit QED
* Linear optical QC
* KLM protocol


 |
| Ultracold atoms | 
* Optical lattice
* Trapped-ion QC


 |
| Spin-based | 
* Kane QC
* Spin qubit QC
* NV center
* NMR QC


 |
| Superconducting | 
* Charge qubit
* Flux qubit
* Phase qubit
* Transmon


 |

 |
| Quantumprogramming | 
* OpenQASM–Qiskit–IBM QX
* Quil–Forest/Rigetti QCS
* Cirq
* Q#
* libquantum
* many others...


 |
| 
* Quantum information science
* Quantum mechanics topics


 |





![](https://login.wikimedia.org/wiki/Special:CentralAutoLogin/start?type=1x1)
Retrieved from "https://en.wikipedia.org/w/index.php?title=Machine\_learning\_in\_physics&oldid=1189519876"
Categories: * Machine learning
* Quantum information science
* Theoretical computer science
* Quantum programming
Hidden categories: * Articles with short description
* Short description matches Wikidata
* Articles with excerpts
* All articles needing additional references
* Articles needing additional references from August 2022






* This page was last edited on 12 December 2023, at 10:27 (UTC).
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





