##Mps-mvRBRL

Accurate prediction of multi-label protein subcellular localization through multi-view feature learning with RBRL classifier

###Guiding principles:

**The dataset file contains Gram-positive bacteria dataset, plant dataset, virus dataset and Gram-negative bacteria dataset .

**Feature extraction
   psepssm.m is the implementation of PsePSSM.
   PSSM-TPC1.m,PSSM-TPC2.m,PSSM-TPC3.py,PSSM-TPC.py is the implementation of PSSM-TPC.
   PAAC.m,mainpseaac.m is the implementation of PseAAC.
   Dipeptide composition can be found from http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/#.
   Gene Ontology can be found from http://www.ebi.ac.uk/GOA/.

** Differential Evolution:
   testFun.m,mutation.m,DE.m,crossover.m is the implementation of DE.

** Dimensional reduction:
   wMLDAb_transform.m, weight_Park2008_Binary.m represents the wMLDAb.
   MDDM_transform.m represents MDDM.
   PCA_transform.m represents PCA.
   MLSI_transform represents MLSI.
   MVMD_transform represents MVMD.

** Classifier:
   Predict.m, train_linear_RBRL_APG.m is the implementation of RBRL.
   LIFT.m is the implementation of LIFT.
   MLKNN_test.m,MLKNN_train.m are the implementation of MLKNN.
   ML_GKR.m is the implementation of ML_GKR.
   MIML_RBF_test.m, MIML_RBF_train.m is the implementation of ML_RBF.
   RankSVM_train.m,RankSVM_test.m is the implementation of RankSVM.
   MIML_kNN_test.m, MIML_kNN_train.m is the implementation of MIML_kNN.

** Demo:
   An example is included in the Demo file.  
   And you can run the demo.m in MATLAB.

