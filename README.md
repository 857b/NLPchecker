# NLPchecker

### Génération des ensembles d'exemples

`make datasets` effectue un précalcul des distances d'éditions entres les
tokens et génère 3 ensembles sous `_data`: `train`, `test` et `validation`.

### Évaluation de la méthode par seuil sur le logit

On utilise les fonctions `evaluate_maskedLogit` et `seuil_optimal` de
[run.py](./run.py) pour évaluer la méthode sur un ensemble de données et
calculer le seuil optimal sur cet ensemble. `plot_maskedLogitError` de
[plot.py](./plot.py) permet d'afficher les erreurs effectuées en fonction
du seuil choisi.

### Entraînement de la méthode utilisant les embeddings

L'entraînement peur être lancé depuis un terminal via [cli.py](./cli.py).
Par exemple:

```
python3 ./cli.py --train _train/0 \
	--train-data _data/train --test-data _data/test \
	--lr '5e-3' --epochs 5 --eval-period 20000 --save-period 10
```

où `_train/0` est un dossier existant dans lequel sera exporté des
checkpoints des paramètres du modèle, les paramètres finaux (`final.json`)
et la trace des loss et précision au cours de l'entraînement (`loss.json`).

On peut ensuite exploiter ces sorties pour évaluer le modèle sur un autre
ensemble de données (`load_classifier` et `evaluate_classifier` dans
[run.py](./run.py)) ou afficher le déroulement de l'entraînement
(`plot_training` dans [plot.py](plot.py)).
