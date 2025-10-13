# 🚀 Lance le Code Maintenant!

## ✅ Tout est Prêt - 3 Façons de Lancer

### Option 1: Quick Test (⭐ RECOMMANDÉ - 2 minutes)

```bash
cd /Users/ulyssetrin/Desktop/matteo/error-estimation
source venv/bin/activate
python quick_test.py
```

**Ce que ça fait**:
- Télécharge ViT-Tiny automatiquement
- Génère des données synthétiques
- Teste le sample splitting (Phase 1)
- Vérifie que tout fonctionne

**Résultat attendu**:
```
✅ Model loading (ViT-Tiny from timm)
✅ Sample splitting (Phase 1 fix)
✅ Independence validation
✅ Empty cluster detection
✅ Model inference
```

---

### Option 2: CIFAR-10 Demo (5-10 minutes)

```bash
cd /Users/ulyssetrin/Desktop/matteo/error-estimation
source venv/bin/activate
python run_cifar10_demo.py
```

**Ce que ça fait**:
- Charge CIFAR-10 (déjà téléchargé!)
- Fait le sample splitting
- Charge ViT-Tiny
- Calcule les predictions
- Montre les statistiques

---

### Option 3: Tests Unitaires (30 secondes)

```bash
cd /Users/ulyssetrin/Desktop/matteo/error-estimation
source venv/bin/activate
PYTHONPATH=$(pwd) pytest tests/test_sample_splitting.py -v
```

**Résultat attendu**: 10/10 tests ✅

---

## 🎯 Commandes Utiles

### Vérifier ce qui est téléchargé
```bash
ls -lh ./data/
# Tu devrais voir:
# cifar-10-batches-py/
# cifar-100-python/
```

### Vérifier l'environnement
```bash
source venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
```

### Lire la documentation
```bash
cat READY_TO_RUN.md     # Guide complet
cat GETTING_STARTED.md  # Setup datasets/modèles
cat PHASE1_FIXES.md     # Explications techniques
```

---

## 🐛 Si Tu As Un Problème

### Erreur: "No module named 'timm'"
```bash
source venv/bin/activate
pip install timm transformers
```

### Erreur: "CIFAR-10 not found"
```bash
python setup_complete.py  # Re-télécharge tout
```

### Le code est lent
C'est normal sur CPU. Pour accélérer:
1. Réduis batch_size dans les scripts
2. Utilise un GPU si disponible
3. Teste avec quick_test.py qui est plus rapide

---

## ✅ Checklist

Avant de lancer, vérifie:
- [ ] Tu es dans le bon dossier: `/Users/ulyssetrin/Desktop/matteo/error-estimation`
- [ ] L'environnement est activé: `source venv/bin/activate`
- [ ] Les datasets sont téléchargés: `ls ./data/`
- [ ] Les dépendances sont installées: `pip list | grep timm`

---

## 🎉 Résumé

**Commande la plus simple** (commence par ça):
```bash
cd /Users/ulyssetrin/Desktop/matteo/error-estimation
source venv/bin/activate
python quick_test.py
```

Ça va prendre 2-3 minutes et te montrer que tout fonctionne!

Bon courage! 🚀
