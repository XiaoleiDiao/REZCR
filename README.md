# REZCR

REZCR is a zero-shot framework for character recognition. It extracts radical information (including radical categories and relative structural relations between radicals) via neural networks, then performs character recognition by exploiting this information for reasoning in a character knowledge graph.


### Files:
1. train.py: Model training pipeline
2. ger_RPs_SP.py: predict two radical information, radical predictions (RPs) and structural relation predictions (SP).
3. KGR.py: reasoning in the character knowledge graph (CKG) via RPs and SP.
4. oracle_779_KG.owl: an example of an oracle character knowledge graph.  
5. radical_train.txt: a txt file with images path and their radical annotation information, generated by the voc_annotation.py.
6. radical_annotation.py: generating radical_train.txt, which is used for training models.


### Folders:
1. GenerateXMLforSingleRadical: automatically generate the XML files for characters with a single radical, which helps to alleviate the annotation cost.
2. nets: models and layers. 
    + RIE.py: main network used to extract radical categories and relative structural relations between radicals in parallel.
    + DSAL.py: building an attention layer. 
    + radical_loss.py: the loss function of RIE. 
3. utils: Helpers and tools.
    + config.py: offer some parameters used in RIE.
    + dataloader.py: loading images from the target path.
    + utils.py: some function used in RIE.

### Requirements:
  + CUDA==10.2 
  + cudnn==8.0
  + python==3.6
  + torch==1.8.0
  + torchvision==0.9.0
  + DNN-printer==0.0.2
  + numpy
  + rdflib
  + collections
  + PIL
  + tqdm
  + wmi (optional)


