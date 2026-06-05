import itertools, random, time, ast
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#inicialización de variables
RANDOM_STATE = 95
N_ITER = 1000 #iteracións da busca aleatoria
MIN_DATASET= 0.7 #tamaño mínimo de validacion
FREQ_THRESHOLD = 0.07 #frecuencia de aparición dun material
TARGET = "temperaturaImpresion"
OTHER_COLS = ["printerBrandType", "preparationTemp", "platformTemp", "relationExtrusionWeight", "drugConcentration"]
    #Diccionario cos hiperparámetros
HP_SPACE = {
    "RF": {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2", 0.5],
    },
    "ET": {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2", 0.5],
    },
    "SVM": {
        "C":      [0.1, 1, 10, 100],
        "kernel": ["rbf", "poly", "linear"],
        "gamma":  ["scale", "auto"],
        "epsilon": [0.01, 0.1, 0.5, 1.0],
    },
    "NN": {
        "hidden_layer_sizes": [(64,), (128,), (256,), (64, 64), (128, 128), (256, 256)],
        "activation":         ["relu", "tanh"],
        "alpha":              [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 5e-4, 1e-4],
        "max_iter":           [1000],
        "early_stopping":     [False],
    },
}

#Diccionario cos modelos
MODEL_FACTORIES = {
    "RF": lambda hp: RandomForestRegressor(**hp, random_state=RANDOM_STATE, n_jobs=-1),
    "ET": lambda hp: ExtraTreesRegressor(**hp, random_state=RANDOM_STATE, n_jobs=-1),
    "SVM": lambda hp: SVR(**hp, max_iter=10000),
    "NN": lambda hp: MLPRegressor(**hp, random_state=RANDOM_STATE),
}

def sample_hp(model_name: str) -> dict:
    """Devolve uns hiperparámetros aleatorios dependendo dun modelo"""
    return {k: random.choice(v) for k, v in HP_SPACE[model_name].items()}

def get_Xy(data: pd.DataFrame, use_cols: list):
    """Devolve X e y (codificada)"""
    X = data[use_cols].copy()
    y = data[TARGET].values
    return X, y

def train_model(data_tr, data_val, use_cols, mat_cols, other_sub, model_name, hp, dim_method, n_pca_comp, do_scale):
    """Preprocesa o conxunto de training e o de validación/test se este é facilitado para finalmente entrenar o modelo"""
    X_tr, y_tr = get_Xy(data_tr, use_cols)
    if data_val is not None:
        X_val, y_val = get_Xy(data_val, use_cols)
    else:
        X_val, y_val = None, None

    if "printerBrandType" in (other_sub or []):
        ohe_tr = enc_printer.transform(X_tr[["printerBrandType"]])
        ohe_cols = enc_printer.get_feature_names_out(["printerBrandType"])

        X_tr = X_tr.drop(columns=["printerBrandType"])
        X_tr = pd.concat([X_tr, pd.DataFrame(ohe_tr, columns=ohe_cols, index=X_tr.index)], axis=1)

        if data_val is not None:
            ohe_val = enc_printer.transform(X_val[["printerBrandType"]])
            X_val = X_val.drop(columns=["printerBrandType"])
            X_val = pd.concat([X_val, pd.DataFrame(ohe_val, columns=ohe_cols, index=X_val.index)], axis=1)

    pca_obj = None
    scaler_obj = None
    scaler_y = None

    if dim_method == "pca":
        X_tr_mat = X_tr[mat_cols].values
        actual_other = [c for c in X_tr.columns if c not in mat_cols]
        X_tr_othr = X_tr[actual_other].values
        pca_obj = PCA(n_components=n_pca_comp, random_state=RANDOM_STATE)
        X_tr_mat = pca_obj.fit_transform(X_tr_mat)
        X_tr = np.hstack([X_tr_mat, X_tr_othr])

        if data_val is not None:
            X_val_mat = X_val[mat_cols].values
            actual_other_val = [c for c in X_val.columns if c not in mat_cols]
            X_val_othr = X_val[actual_other_val].values
            X_val_mat = pca_obj.transform(X_val_mat)
            X_val = np.hstack([X_val_mat, X_val_othr])

    if do_scale:
        scaler_obj = StandardScaler()
        X_tr = scaler_obj.fit_transform(X_tr)
        if data_val is not None:
            X_val = scaler_obj.transform(X_val)
        scaler_y = StandardScaler()
        y_tr = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

    clf = MODEL_FACTORIES[model_name](hp)
    clf.fit(X_tr, y_tr)

    return clf, pca_obj, scaler_y, scaler_obj, X_val, y_val


#inicialización da aleatoridade coa semente
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

#lectura dos datos e preprocesado
df = pd.read_csv("DatosLimpiados.csv")

all_cols = list(df.columns) #lista con todas as columnas
lastMaterial_idx = next( i for i, c in enumerate(all_cols) if c.strip().lower() == "xanthan gum") #índice do último material
ingredient_cols = all_cols[:lastMaterial_idx + 1] #columnas dos materiais

# Eliminar outlier coñecido
outlier_mask = df["High methoxyl pectin (ESS-4400)"] == 99.5
df = df[~outlier_mask].reset_index(drop=True)

# Materiais a proporcións
row_sums = df[ingredient_cols].sum(axis=1)
df[ingredient_cols] = (df[ingredient_cols].div(row_sums, axis=0))

# División training-test
n_total = len(df)
n_train = int(np.floor(n_total * 0.8))
train_df = df.iloc[:n_train].copy()
test_df  = df.iloc[n_train:].copy()

# Codificación das variables categóricas
enc_printer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
enc_printer.fit(train_df[["printerBrandType"]])

#combinacións das características que non son materiais
other_combos = []
for r in range(1, len(OTHER_COLS) + 1):
    for combo in itertools.combinations(OTHER_COLS, r):
        other_combos.append(list(combo))

#busca aleatoria
results = []
train_len = len(train_df.dropna(subset=[TARGET]))
splits = [ 0.60, 0.80, 1]
seen_configs = set()  # rexitro de combinacións xa probadas
t0 = time.time()#Inicializo o tempo que se tarda en entrenar os modelos
iteration=0

print(f"Tamaño do conxunto de datos de validación:{train_len}")

while iteration < N_ITER and iteration < 430896 : #para no numero maximo de combinacións en caso de alcanzalo
    #Elección dunha combinación aleatoria
    model_name = random.choice(["RF", "ET", "SVM", "NN"])
    hp = sample_hp(model_name) # hiperparámetros
    feature_mode = random.choice(["ingredients", "other_sub", "both"])
    filter_low_temp = random.choice([True, False])
    if feature_mode!="other_sub":
        dim_method = random.choice(["freq", "pca"])
        if dim_method == "pca":  # elixo unha das tres posibilidades de reduccións pca
            n_pca_comp = min(random.choice([5, 10, 15, 20]), len(ingredient_cols))
        else:
            n_pca_comp=None
    else:
        dim_method=None
        n_pca_comp=None
    if feature_mode!="ingredients":
        other_sub = random.choice(other_combos)
    else:
        other_sub=None
    do_scale = random.choice([True, False])

    # Firma única desta configuración
    config_key = (
        feature_mode,
        model_name,
        tuple(sorted(hp.items())),
        dim_method,
        n_pca_comp,
        tuple(sorted(other_sub)) if other_sub else (),
        do_scale,
        filter_low_temp,

    )
    if config_key in seen_configs: #se xa probei a configuración salto a seguinte
        continue

    iteration += 1 #abanzo no bucle
    seen_configs.add(config_key)# engado a combinación ás probadas

    fold_metrics = []
    y_all_val, y_pred_all_val = [], []  # Para a matriz de confusión

    if dim_method == "freq": # Eliminación dos ingredientes con menor frecuencia a 0.7
        use_cols = ingredient_cols + (other_sub or [])
        sub = train_df[use_cols + [TARGET]].dropna(subset=[TARGET])
        freq = {col: (sub[col] != 0).sum() / len(sub) for col in ingredient_cols}
        freq_series = pd.Series(freq)
        mat_cols = freq_series[freq_series >= FREQ_THRESHOLD].index.tolist()
    elif dim_method=="pca": #se pca todos os materiais
        mat_cols = ingredient_cols
    else:
        mat_cols=[]

    use_cols = mat_cols + (other_sub or [])

    sub = train_df.dropna(subset=[TARGET])
    if filter_low_temp:
        sub = sub[sub[TARGET] > 25]
    sublen=len(sub)
    sub = sub[use_cols + [TARGET]].dropna() #eliminar nulos e quedar solo coas caracteristicas indicadas

    if len(sub) < MIN_DATASET*sublen:  # Descartar se quedan < 100 instancias
        continue

    for (val_end) in splits: # en cada fold

        fold_train = sub.iloc[:int(val_end*0.8*len(sub))]
        fold_val = sub.iloc[int(val_end*0.8*len(sub)):int(val_end*len(sub))]

        clf, _, scaler_y,_, X_val, y_val = train_model(fold_train, fold_val, use_cols, mat_cols, other_sub, model_name, hp, dim_method, n_pca_comp, do_scale)

        y_pred = clf.predict(X_val) #evaluo sobre o conxunto de validación

        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        y_all_val.extend(y_val)
        y_pred_all_val.extend(y_pred)

        fold_metrics.append({
            "mae": mean_absolute_error(y_val, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
            "r2": r2_score(y_val, y_pred),
        })

    # Agregación dos folds
    agg = {m: np.mean([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
    agg["r2_std"] = np.std([f["r2"] for f in fold_metrics])
    agg["score"] = agg["r2"] - agg["r2_std"]  # penalizar inestabilidad

    results.append({
        "iteration": iteration,
        "model": model_name,
        "hp": str(hp),
        "dim_method": dim_method,
        "do_scale": do_scale,
        "filter_low_temp": filter_low_temp,
        "n_pca": n_pca_comp if n_pca_comp else "N/A",
        "other_cols": str(other_sub or []),
        "dataset_size": len(sub),
        **agg,
    })

    #imprimo os parametros da iteración
    elapsed  = time.time() - t0
    pca_str  = f"PCA={n_pca_comp}" if n_pca_comp else dim_method
    best_so_far = max((r["score"] for r in results if r["score"] is not None), default=None)
    normalice= "normalizado" if do_scale else "no normalizado"
    print(
        f"  [{iteration}/{N_ITER}] {model_name} | {pca_str} | {normalice} | "
        f"extras={len(other_sub or [])} | "
        f"filter={filter_low_temp} | "
        f"MAE={agg['mae']:.3f} | "
        f"R2={agg['r2']:.4f}±{agg['r2_std']:.4f} | "
        f"score={agg['score']:.4f} | "
        f"tamaño dataset={len(sub)} | "
        f"mellor={best_so_far} | "
        f"{elapsed:.1f}s"
    )

print(f"\nBusca rematada en {time.time()-t0:.1f}s  |  "
      f"Configuracións válidas: {len(results)}")

#selección do mellor modelo
    #ordeno os resultados con mellor R2 - desviación
results_df = pd.DataFrame(results).sort_values(["score"],ascending=[False])

#gardo os resultados
results_df.to_csv("temperatura.csv", index=False)

#imprimo o mellor resultado
best = results_df.iloc[0]
print("\nMellor configuración:")
print(f"  Modelo:          {best['model']}")
print(f"  Hiperparámetros: {best['hp']}")
print(f"  Dim. method:     {best['dim_method']}")
print(f"  n_pca:           {best['n_pca']}")
print(f"  Outras cols:     {best['other_cols']}")
print(f"  Normalizado:     {best['do_scale']}")
print(f"  Filter low temp: {best['filter_low_temp']}")
print(f"  R2 media:         {best['r2']}  ±  {best['r2_std']}")
print(f"  Score (R2-σ):     {best['score']}")
print(f"  MAE:        {best['mae']}")


#Adestro co mellor modelo o 80% dos datos que dedicara a treining e validation
best_model_name = best["model"]
best_hp = ast.literal_eval(best["hp"])
best_dim = best["dim_method"]
best_other = ast.literal_eval(best["other_cols"])
best_n_pca = best["n_pca"]
best_do_scale = bool(best["do_scale"])

    # Calcular mat_cols igual que no loop
if best_dim == "freq":
    sub_freq = train_df[ingredient_cols + [TARGET]].dropna(subset=[TARGET])
    freq = {col: (sub_freq[col] != 0).sum() / len(sub_freq) for col in ingredient_cols}
    freq_series = pd.Series(freq)
    best_mat_cols = freq_series[freq_series >= FREQ_THRESHOLD].index.tolist()
elif best_dim=="pca":
    best_mat_cols = ingredient_cols
else:
    best_mat_cols=[]

use_cols = best_mat_cols + best_other
subtrain = train_df[use_cols + [TARGET]].dropna()  # eliminar nulos e quedar solo coas caracteristicas indicadas train
subtest = test_df[use_cols + [TARGET]].dropna()  # eliminar nulos e quedar solo coas caracteristicas indicadas test
if best["filter_low_temp"]:
    subtrain = subtrain[subtrain[TARGET] > 25]
    subtest = subtest[subtest[TARGET] > 25]
print(f"tamaño dataset={len(subtrain)+len(subtest)}")

clf_final, _, scaler_y,_ , X_tst, y_tst = train_model(subtrain, subtest, use_cols, best_mat_cols, best_other, best_model_name, best_hp, best_dim, best_n_pca, best_do_scale)

y_pred_test = clf_final.predict(X_tst)

if scaler_y is not None:
    y_pred_test = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

print("\nAvaliación no conxunto de test:")
print(f"  MAE:   {mean_absolute_error(y_tst, y_pred_test):.4f}")
print(f"  RMSE:  {np.sqrt(mean_squared_error(y_tst, y_pred_test)):.4f}")
print(f"  R2:    {r2_score(y_tst, y_pred_test):.4f}")


subtrainTotal = df[use_cols + [TARGET]].dropna()  # eliminar nulos e quedar solo coas caracteristicas indicadas train
if best["filter_low_temp"]:
    subtrainTotal = subtrainTotal[subtrainTotal[TARGET] > 25]

    #Entreno o modelo final con todos os datos
clf_final_guardar, pca, scaler_y,scaler_final, _,_ = train_model(subtrainTotal, None, use_cols, best_mat_cols, best_other, best_model_name, best_hp, best_dim, best_n_pca, best_do_scale)

# Grado o modelo
joblib.dump({
    "clf":             clf_final_guardar,
    "scaler":          scaler_final,
    "scaler_y":        scaler_y,
    "enc_final":       enc_printer if "printerBrandType" in best_other else None,
    "pca_final":       pca,
    "dim_method":      best_dim,
    "other_cols":      best_other,
    "mat_cols":        best_mat_cols,
    "ingredient_cols": ingredient_cols,
    "freq_threshold":  FREQ_THRESHOLD,
}, "temperatura.pkl")

print("\nModelo gardado en temperatura.pkl")
print("Resultados da busca gardados en  temperatura.csv")