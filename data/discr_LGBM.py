import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import random

# False entraîne sur 80% du training et teste sur les 20% restant ; True entraîne sur 100% du training et écrit les résultats
FINAL = False

# Nombre de tests à effectuer pour prendre la valeur moyenne et avoir une bonne estimation du coefficient kappa (indépendante de la répartition du training set)
N_tests = 1

# Nombre d'entraînements pour chaque modèle (afin de réduire les biais)
N_trains = 3

# Nombre de coeurs
N_cores = 4

# Nombre d'estimateurs
N_est = 10

# Faut-il normaliser les données
NORMALIZE = True

# Faut-il réduire le nombre de features
REDUCE_FEATURES = False

# Faut-il utiliser des clusters
USE_CLUSTERS = True

# Features différentes
ranges = (slice(1, 200), slice(200, 2248), slice(2248, 4296), slice(4296, 4309))
class_col = 4296

params = (
	{	"class_weight": "balanced",
		"colsample_bytree": 0.6711844025319801,
		"learning_rate": 0.02,
		"min_child_samples": 21,
		"min_child_weight": 0.007935208891156327,
		"n_estimators": N_est,
		"n_jobs": N_cores,
		"num_leaves": 32,
		"reg_alpha": 0.7712033135332051,
		"reg_lambda": 0.7156020062219308,
		"subsample_freq": 2},
	{   
		"class_weight": "balanced",
		"colsample_bytree": 0.5185878473906118,
		"learning_rate": 0.2819183874082652,
		"min_child_samples": 16,
		"min_child_weight": 0.007076389406189793,
		"n_estimators": N_est,
		"n_jobs": N_cores,
		"num_leaves": 64,
		"reg_alpha": 0.9288233666023282,
		"reg_lambda": 0.551821503605106,
		"subsample": 0.9955253005922045,
		"subsample_freq": 1},
	{
		"class_weight": "balanced",
		"colsample_bytree": 0.7397790694843642,
		"learning_rate": 0.02,
		"min_child_samples": 5,
		"min_child_weight": 0.0033323989081406563,
		"n_estimators": N_est,
		"n_jobs": N_cores,
		"num_leaves": 128,
		"reg_alpha": 0.017293089124126082,
		"reg_lambda": 0.9519087533792667,
		"subsample": 0.021044508473887474,
	}
)

# paramètres avec moins d'overfit

# params = [{
# 	"boosting_type": "dart",
# 	"class_weight": "balanced",
# 	"colsample_bytree": 0.9408719209255059,
# 	"learning_rate": 0.1,
# 	"min_child_samples": 50,
# 	"min_child_weight": 0.004092554311293762,
# 	"n_estimators": 300,
# 	"n_jobs": 4,
# 	"num_leaves": 32,
# 	"random_state": random.randint(1, 1000),
# 	"reg_alpha": 1,
# 	"reg_lambda": 1,
# 	"subsample": 0.7,
# 	"subsample_freq": 3,
# 	"verbose": -1,
# 	}]*3

# lire le dataframe

#df_train = pd.read_csv("data/train.csv")
df_train = pd.read_csv("data/train_cluster.csv")
# df_test1 = pd.read_csv("data/test_1_cluster.csv")
df_test2 = pd.read_csv("data/test_2_cluster.csv")
# if "series" in df_test2.columns:
# 	df_test2.drop(columns = ["series"])

# for i in range(13):
# 	df_train.drop(columns=["series_"+str(i)])

# normaliser (les 200 premières sont normalisées indépendamment, et les 2 blocs de 2048 ensemble)

def normalize(X_train, X_test):
	cc = pd.concat((X_train, X_test))

	for i in range(ranges[0].start, ranges[0].stop):
		m, s = cc.iloc[:, i].mean(), cc.iloc[:, i].std()

		if s != 0:
			X_train.iloc[:, i] = (X_train.iloc[:, i] - m) / s
			X_test.iloc[:, i] = (X_test.iloc[:, i] - m) / s

	for r in ranges[1:3]:
		m, s = cc.iloc[:, r].to_numpy().flatten().mean(), cc.iloc[:, r].to_numpy().flatten().std()
		
		X_train.iloc[:, r] = (X_train.iloc[:, r] - m) / s
		X_test.iloc[:, r] = (X_test.iloc[:, r] - m) / s

# pour évaluer la qualité

def kappa_cohen(y_test, y_pred):
	t = [0,0,0,0]
	l = [(1,1),(1,0),(0,1),(0,0)]

	for i in range(len(y_test)):
		t[l.index((y_test[i], y_pred[i]))] += 1

	tot = len(y_test)
	p_acc = (t[0] + t[3]) / tot
	p_has = ((t[0] + t[1]) * (t[0] + t[2]) + (t[2] + t[3]) * (t[1] + t[3])) / tot**2
	return (p_acc - p_has) / (1 - p_has)

# fit X_test en ne prenant en compte que les features dans range

def LGBM(X_train, y_train, X_test, params):
	params["random_state"] = random.randint(1, 1000)
	params["verbose"] = -1

	# model = LGBMClassifier(
	# 	boosting_type="dart",
	# 	class_weight="balanced",
	# 	colsample_bytree=0.9408719209255059,
	# 	learning_rate=0.7,
	# 	min_child_samples=30,
	# 	min_child_weight=0.004092554311293762,
	# 	n_estimators=700,
	# 	n_jobs=4,
	# 	num_leaves=64,
	# 	random_state=random.randint(1, 1000),
	# 	reg_alpha=0.720249010918679,
	# 	reg_lambda=0.9322794252081102,
	# 	subsample=0.01,
	# 	verbose=-1)

	model = LGBMClassifier(**params)

	model.fit(X_train, y_train)
	
	return model.predict_proba(X_train), model.predict_proba(X_test)

def RFC(X_train, y_train, X_test):
	X_train = X_train.fillna(0)
	X_test = X_test.fillna(0)

	model = RandomForestClassifier(
		n_estimators=400,
		max_depth=6,
		class_weight="balanced",
		random_state=random.randint(1, 1000)
	)

	model.fit(X_train, y_train)

	return model.predict_proba(X_train), model.predict_proba(X_test)

	# param_grid = {
	# 	'n_estimators': [100, 500, 1000],
	# 	'learning_rate': [0.01, 0.1, 0.5],
	# 	'max_depth': [7, 10, 15],
	# 	'num_leaves': [31, 50, 100],
	# 	'subsample': [0.01, 0.1, 1.0],
	# 	'colsample_bytree': [0.8, 1.0]
	# }
	
	# model = LGBMClassifier(boosting_type = "dart", class_weight = "balanced", min_child_samples = 30, min_child_weight = 0.004092554311293762,
	# 					n_jobs = 4, reg_alpha = 0.720249010918679, reg_lambda = 0.9322794252081102)

	# grid_search = RandomizedSearchCV(model, param_grid, scoring = "accuracy", verbose = 1, n_jobs = 4, cv = StratifiedKFold(n_splits=5, shuffle = True), n_iter=10)
	# grid_search.fit(X_train, y_train)

	# print ("Avec la range", range)
	# print ("Meilleurs paramètres:", grid_search.best_params_)
	# print ("Score:", grid_search.best_score_)

def compute(df_train, df_test, output):
	# séparer en fichiers de train et de test

	if FINAL:
		X_train = df_train.copy()
		X_test = df_test.copy()
		y_train = df_train.iloc[:, class_col]
	
	else:
		X_train, X_test, y_train, y_test = train_test_split(df_train, df_train.iloc[:,class_col], test_size = 0.2)
	
	smiles = X_test.iloc[:, 0]
	
	X_train = X_train.drop(columns="class")
	if "class" in X_test.columns:
		X_test = X_test.drop(columns="class")

	feats_train_base = [X_train.iloc[:, r] for r in ranges]
	feats_test_base = [X_test.iloc[:, r] for r in ranges]

	# normaliser les entrées
	if NORMALIZE:
		normalize(X_train, X_test)

	# réduire les features
	if REDUCE_FEATURES:
		selector = VarianceThreshold(threshold = 0.01)
		selector.fit(X_train)

		selected_features = X_train.columns[selector.get_support()]

		X_train = pd.DataFrame(X_train, columns = selected_features)
		X_test = pd.DataFrame(X_test, columns = selected_features)

	# on calcule le résultat majoritaire entre celui donné en se basant sur les 200 premières features, les 2048 suivantes et les 2048 suivantes

	if USE_CLUSTERS:
		feats_train = [pd.concat((feats_train_base[k], feats_train_base[3]), axis=1) for k in range(3)]
		feats_test = [pd.concat((feats_test_base[k], feats_test_base[3]), axis=1) for k in range(3)]
	else:
		feats_train = feats_train_base[:3]
		feats_test = feats_test_base[:3]
	
	# print (feats_train[0].head())
	# print (feats_train[1].head())
	# print (feats_train[2].head())

	y_preds_tot = tuple(LGBM(feats_train[k], y_train, feats_test[k], params[k]) for k in range(3) for i in range(N_trains))

	y_preds_train = [t[0] for t in y_preds_tot]
	y_preds = [t[1] for t in y_preds_tot]

	# on moyenne les probabilités d'obtenir 1, et on prend 1 pour valeur finale si cette proba est supérieure à 1
	probas = [sum(y_preds[k][i][1] for k in range(len(y_preds))) / len(y_preds) for i in range(len(y_preds[0]))]
	y_pred = [int(p >= 0.5) for p in probas]

	probas_train = [sum(y_preds_train[k][i][1] for k in range(len(y_preds_train))) / len(y_preds_train) for i in range(len(y_preds_train[0]))]
	y_pred_train = [int(p >= 0.5) for p in probas_train]

	if FINAL:
		answer = pd.DataFrame()
		answer["smiles"] = smiles
		answer["class"] = y_pred
		# if "series" in df_test.columns:
		# 	answer["series"] = series
		# 	answer.sort_values("series")
		answer["probas"] = [max(i, 1-i) for i in probas]

		answer = answer.sort_values("probas", ascending = False)
		answer = answer.drop(columns=["probas"])

		answer.to_csv(output, index = False)

		return answer

	else:
		# print ("Moyenne des conf pour les résultats justes:", np.array([probas[i] for i in range(len(y_pred)) if y_pred[i] == y_test.iloc[i]]).mean())
		# print ("Moyenne des conf pour les résultats faux:", np.array([probas[i] for i in range(len(y_pred)) if y_pred[i] != y_test.iloc[i]]).mean())
		train_accuracy = accuracy_score(y_train, y_pred_train)
		test_accuracy = accuracy_score(y_test, y_pred)
		
		train_auc = roc_auc_score(y_train.values, probas_train)
		test_auc = roc_auc_score(y_test.values, probas)

		print (f"Accuracy entraînement : {train_accuracy:.4f}")
		print (f"Accuracy test : {test_accuracy:.4f}")
		print(f"AUC-ROC entraînement : {train_auc:.4f}")
		print(f"AUC-ROC test : {test_auc:.4f}")

		print("kappa_cohen total, en prenant le plus fréquent:", kappa_cohen(y_test.array, y_pred))

		for j in range(3):
			proba_part = [sum(y_preds[k][i][1] for k in range(j * N_trains, (j+1) * N_trains)) / N_trains for i in range(len(y_preds[0]))]
			y_pred_part = [int(p >= 0.5) for p in proba_part]
			# print ("kappa_cohen avec le range", i, ":", kappa_cohen(y_test.array, list(map(lambda t: int(t[1] >= 0.5), y_pred_part))))
			print ("kappa_cohen avec le range", i, ":", kappa_cohen(y_test.array, y_pred_part))

		return kappa_cohen(y_test.array, y_pred)

if FINAL:
	#compute(df_train, df_test1, "pred_1.csv")
	compute(df_train, df_test2, "pred_2.csv")
else:
	ans = 0
	for i in range(N_tests):
		ans += compute(df_train, None, None)
	print ("\nEn moyenne:", ans / N_tests)

#compute()