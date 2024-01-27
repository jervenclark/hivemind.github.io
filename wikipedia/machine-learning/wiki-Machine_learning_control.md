


















Contents
--------


move to sidebar
hide

* (Top)
* 1Types of problems and tasks
* 2Applications
* 3References
* 4Further reading














Toggle the table of contents







Machine learning control
========================





2 languages



* فارسی
* 中文


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


Subfield of machine learning, intelligent control and control theory
**Machine learning control** (**MLC**) is a subfield of machine learning, intelligent control and control theory
which solves optimal control problems with methods of machine learning.
Key applications are complex nonlinear systems
for which linear control theory methods are not applicable.




Types of problems and tasks[edit]
---------------------------------


Four types of problems are commonly encountered.



* Control parameter identification: MLC translates to a parameter identification[1] if the structure of the control law is given but the parameters are unknown. One example is the genetic algorithm for optimizing coefficients of a PID controller[2] or discrete-time optimal control.[3]
* Control design as regression problem of the first kind: MLC approximates a general nonlinear mapping from sensor signals to actuation commands, if the sensor signals and the optimal actuation command are known for every state. One example is the computation of sensor feedback from a known full state feedback. A neural network is commonly used technique for this task.[4]
* Control design as regression problem of the second kind: MLC may also identify arbitrary nonlinear control laws which minimize the cost function of the plant. In this case, neither a model, nor the control law structure, nor the optimizing actuation command needs to be known. The optimization is only based on the control performance (cost function) as measured in the plant. Genetic programming is a powerful regression technique for this purpose.[5]
* Reinforcement learning control: The control law may be continually updated over measured performance changes (rewards) using reinforcement learning.[6]


MLC comprises, for instance, neural network control, 
genetic algorithm based control, 
genetic programming control,
reinforcement learning control, 
and has methodological overlaps with other data-driven control,
like artificial intelligence and robot control.



Applications[edit]
------------------


MLC has been successfully applied
to many nonlinear control problems,
exploring unknown and often unexpected actuation mechanisms.
Example applications include



* Attitude control of satellites.[7]
* Building thermal control.[8]
* Feedback turbulence control.[2][9]
* Remotely operated underwater vehicles.[10]
* Many more engineering MLC application are summarized in the review article of PJ Fleming & RC Purshouse (2002).[11]


As for all general nonlinear methods,
MLC comes with no guaranteed convergence, 
optimality or robustness for a range of operating conditions.



References[edit]
----------------



1. **^** Thomas Bäck & Hans-Paul Schwefel (Spring 1993) "An overview of evolutionary algorithms for parameter optimization", Journal of Evolutionary Computation (MIT Press), vol. 1, no. 1, pp. 1-23
2. ^ ***a*** ***b*** N. Benard, J. Pons-Prats, J. Periaux, G. Bugeda, J.-P. Bonnet & E. Moreau, (2015) "Multi-Input Genetic Algorithm for Experimental Optimization of the Reattachment Downstream of a Backward-Facing Step with Surface Plasma Actuator", Paper AIAA 2015-2957 at 46th AIAA Plasmadynamics and Lasers Conference, Dallas, TX, USA, pp. 1-23.
3. **^** Zbigniew Michalewicz, Cezary Z. Janikow & Jacek B. Krawczyk (July 1992) "A modified genetic algorithm for optimal control problems", [Computers & Mathematics with Applications], vol. 23, no 12, pp. 83-94.
4. **^** C. Lee, J. Kim, D. Babcock & R. Goodman (1997) "Application of neural networks to turbulence control for drag reduction", Physics of Fluids, vol. 6, no. 9, pp. 1740-1747
5. **^** D. C. Dracopoulos & S. Kent (December 1997) "Genetic programming for prediction and control", Neural Computing & Applications (Springer), vol. 6, no. 4, pp. 214-228.
6. **^** Andrew G. Barto (December 1994) "Reinforcement learning control", Current Opinion in Neurobiology, vol. 6, no. 4, pp. 888–893
7. **^** Dimitris. C. Dracopoulos & Antonia. J. Jones (1994) 
Neuro-genetic adaptive attitude control, Neural Computing & Applications (Springer), vol. 2, no. 4, pp. 183-204.
8. **^** Jonathan A. Wright, Heather A. Loosemore & Raziyeh Farmani (2002) "Optimization of building thermal design and control by multi-criterion genetic algorithm, [Energy and Buildings], vol. 34, no. 9, pp. 959-972.
9. **^** Steven J. Brunton & Bernd R. Noack (2015) Closed-loop turbulence control: Progress and challenges, Applied Mechanics Reviews, vol. 67, no. 5, article 050801, pp. 1-48.
10. **^** J. Javadi-Moghaddam, & A. Bagheri (2010 "An adaptive neuro-fuzzy sliding mode based genetic algorithm control system for under water remotely operated vehicle", Expert Systems with Applications, vol. 37 no. 1, pp. 647-660.
11. **^** Peter J. Fleming, R. C. Purshouse (2002 "Evolutionary algorithms in control systems engineering: a survey"
Control Engineering Practice, vol. 10, no. 11, pp. 1223-1241

Further reading[edit]
---------------------



* Dimitris C Dracopoulos (August 1997) "Evolutionary Learning Algorithms for Neural Adaptive Control", Springer. ISBN 978-3-540-76161-7.
* Thomas Duriez, Steven L. Brunton & Bernd R. Noack (November 2016) "Machine Learning Control - Taming Nonlinear Dynamics and Turbulence", Springer. ISBN 978-3-319-40624-4.






![](https://login.wikimedia.org/wiki/Special:CentralAutoLogin/start?type=1x1)
Retrieved from "https://en.wikipedia.org/w/index.php?title=Machine\_learning\_control&oldid=1178520214"
Categories: * Machine learning
* Control theory
* Cybernetics
Hidden categories: * Articles with short description
* Short description matches Wikidata






* This page was last edited on 4 October 2023, at 05:30 (UTC).
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





