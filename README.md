# Projet-Galaxie-2020

## Arborescence

| Fichier | Contenu | Type |
| :-----: | :-----: | :-----: |
| pic_process.py | Toutes les fonctions de traitement d'images | Pack de fonctions |
| minkos.py | Toutes les fonctions relatives aux fonctionelles de M | Pack de fonctions |
| main.py | Calcule les fonctionelles d'une image en format DAT ou FITS | Programme |
| trasismooth.py | Représente les variations des fonctionelles en fonction du lissage des images | Programme |
| morpho.py | Utilise la méthode conventionelle pour calculer les paramètres d'une galaxie | Programme |

| **Attention: Les fichiers de type "Pack de fonctions" sont utilisés par plusieurs programmes, faire attention lors de leurs modifications.** |
| --- |

## main.py

### Appel:
  `python main.py [FITS ou DAT file] [optional args]`

  - Le chemin vers l'image peut être absolu ou relatif.
  - Les arguments optionels et leurs utilisations peuvent être vus avec `python main.py --h`


### Exemple de résultat:
  ![alt text][Test1]

##### Tous les résultats:
  [lien](all_res.md)


## transismooth.py
  `python transismooth.py [FITS ou DAT file] [optional args]`

  - `-smooth [NOMBRE]` pour le nombre d'itérations. Attention, à plus de 6, le temps de calcul est très long.

### Exemple de résultat:

<img src="datfiles/spiraleBarreeA_Chi.png" alt="drawing" width="200"/><img src="datfiles/spiraleBarreeA_U.png" alt="drawing" width="200"/><img src="datfiles/spiraleBarreeA_F.png" alt="drawing" width="200"/>

## morpho.py
  `python morpho.py [FITS ou DAT file] [optional args]`




[Test1]: img/yesyes.png "SuperLesFonctionelles"
