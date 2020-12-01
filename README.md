# LaMuse
LaMuse : utilisation de méthodes de deep learning pour créer des sources d'inspiration pour les peintres.

Ce dossier utilise le dépôt de matterport Mask_RCNN : https://github.com/matterport/Mask_RCNN

# Utilisation :

Pour fonctionner il est nécessaire d'aouter à la racine du projet le fichier [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) détaillant les poids du modèle Mask_RCNN.h5 (modèle Keras).


Le code Python est censé être invoqué depuis la racine du dépôt. Pour l'instant certains chemins sont codés en dur.

Pour lancer un exécutable, il faut donc faire :

``python3 -m Tools.create_original_case_study``


# Version Démo :

``python3 -m LaMuse --demo``
