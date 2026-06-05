import itertools, random, time, ast
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# inicialización de variables
RANDOM_STATE = 95
N_ITER = 80  # iteracións da busca aleatoria

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
        "epsilon": [0.01, 0.1, 0.5],
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

# Diccionario cos modelos
MODEL_FACTORIES = {
    "RF": lambda hp: MultiOutputRegressor(RandomForestRegressor(**hp, random_state=RANDOM_STATE, n_jobs=-1)),
    "ET": lambda hp: MultiOutputRegressor(ExtraTreesRegressor(**hp, random_state=RANDOM_STATE, n_jobs=-1)),
    "SVM": lambda hp: MultiOutputRegressor(SVR(**hp)),
    "NN": lambda hp: MLPRegressor(**hp, random_state=RANDOM_STATE),
}

def sample_hp(model_name: str) -> dict:
    """Devolve uns hiperparámetros aleatorios dependendo dun modelo"""
    return {k: random.choice(v) for k, v in HP_SPACE[model_name].items()}

def build_missing_material_dataset(df: pd.DataFrame, ingredient_cols: list):
    col_idx = {c: i for i, c in enumerate(ingredient_cols)}
    n = len(ingredient_cols)

    X_rows, Y_rows, output_sets = [], [], []

    for _, row in df.iterrows():
        present = [c for c in ingredient_cols if row[c] > 0]
        if len(present) < 2:
            continue
        for n_known in range(1, len(present)):
            known = random.sample(present, n_known)
            unknown = [c for c in present if c not in known]

            sum_known = sum(row[c] for c in known)
            R = 1.0 - sum_known
            if R <= 0:
                continue

            x_vec = np.zeros(n)
            y_vec = np.zeros(n)
            for c in known:
                x_vec[col_idx[c]] = row[c]
            for c in unknown:
                y_vec[col_idx[c]] = row[c] / R

            X_rows.append(x_vec)
            Y_rows.append(y_vec)
            output_sets.append(unknown)

    return np.array(X_rows), np.array(Y_rows), output_sets

def evaluate_predictions(Y_true: np.ndarray, Y_pred: np.ndarray, output_sets: list, ingredient_cols: list) -> dict:
    """
    Avalia as prediccions só nas columnas de saída de cada instancia
    """
    col_idx = {c: i for i, c in enumerate(ingredient_cols)}
    maes, rmses, r2s = [], [], []
    all_true = []
    all_pred = []

    for i, out_cols in enumerate(output_sets):
        idxs = [col_idx[c] for c in out_cols]
        yt = Y_true[i, idxs]
        yp = Y_pred[i, idxs]

        maes.append(mean_absolute_error(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        all_true.extend(Y_true[i, idxs])
        all_pred.extend(Y_pred[i, idxs])

    r2 = r2_score(all_true, all_pred)




    return { #Metrica representativa de todas las instancias
        "mae":  float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r2":   float(r2),
    }


def train_model(X_tr, Y_tr, X_val, model_name, hp, n_pca_comp, do_scale):
    """
    Preprocesa e adestra o modelo
    """
    scaler_obj = None

    # Aplicar PCA
    pca_obj = PCA(n_components=n_pca_comp, random_state=RANDOM_STATE)
    X_tr_pca  = pca_obj.fit_transform(X_tr)
    X_val_pca = pca_obj.transform(X_val) if X_val is not None else None

    # Normalización opcional
    if do_scale:
        scaler_obj = StandardScaler()
        X_tr_pca  = scaler_obj.fit_transform(X_tr_pca)
        if X_val_pca is not None:
            X_val_pca = scaler_obj.transform(X_val_pca)

    clf = MODEL_FACTORIES[model_name](hp)
    clf.fit(X_tr_pca, Y_tr)

    return clf, pca_obj, scaler_obj, X_val_pca


# inicialización da aleatoridade coa semente
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# lectura dos datos e preprocesado
df = pd.read_csv("DatosLimpiados.csv")

all_cols = list(df.columns)
lastMaterial_idx = next(i for i, c in enumerate(all_cols) if c.strip().lower() == "xanthan gum")
ingredient_cols = all_cols[:lastMaterial_idx + 1]

# Eliminar outlier coñecido
outlier_mask = df["High methoxyl pectin (ESS-4400)"] == 99.5
df = df[~outlier_mask].reset_index(drop=True)

# Materiais a proporcións
row_sums = df[ingredient_cols].sum(axis=1)
df[ingredient_cols] = df[ingredient_cols].div(row_sums, axis=0)

# División training-test
n_total = len(df)
n_train = int(np.floor(n_total * 0.8))
train_df = df.iloc[:n_train].copy()
test_df  = df.iloc[n_train:].copy()

# Construir o conxunto de datos de adestramento e test a partir das formulacións
print("Construíndo instancias de adestramento...")
# Solo dejar estas:
X_train, Y_train, out_sets_train = build_missing_material_dataset(train_df, ingredient_cols)
X_test,  Y_test,  out_sets_test  = build_missing_material_dataset(test_df,  ingredient_cols)
train_len = len(X_train)

print(f"Instancias de adestramento xeradas: {train_len}")
print(f"Instancias de test xeradas: {len(X_test)}")

# proporcións de datos usadas nos folds de validación
splits = [0.60, 0.80, 1.0]

# busca aleatoria
results = []
seen_configs = set()
t0 = time.time()
iteration = 0

while iteration < N_ITER:
    # Elección dunha combinación aleatoria
    model_name = random.choice(["RF", "ET", "SVM", "NN"])
    hp = sample_hp(model_name)

    # Aplico PCA
    n_pca_comp = min(random.choice([5, 10, 15, 20]), len(ingredient_cols))
    do_scale   = random.choice([True, False])

    # Firma única desta configuración
    config_key = (model_name, tuple(sorted(hp.items())), n_pca_comp, do_scale)
    if config_key in seen_configs:
        continue

    iteration += 1
    seen_configs.add(config_key)

    fold_metrics = []

    for val_end in splits:  # en cada fold
        n_fold = int(val_end * train_len)
        n_fold_tr = int(0.8 * n_fold)

        X_fold_tr = X_train[:n_fold_tr]
        Y_fold_tr = Y_train[:n_fold_tr]
        X_fold_val = X_train[n_fold_tr:n_fold]
        Y_fold_val = Y_train[n_fold_tr:n_fold]
        out_sets_val = out_sets_train[n_fold_tr:n_fold]

        clf, pca_obj, scaler_obj, X_val_t = train_model( X_fold_tr, Y_fold_tr, X_fold_val, model_name, hp, n_pca_comp, do_scale)

        Y_pred_val = clf.predict(X_val_t)
        metrics = evaluate_predictions(Y_fold_val, Y_pred_val, out_sets_val, ingredient_cols)
        fold_metrics.append(metrics)

    # Agregación dos folds
    agg = {
        "mae":  np.mean([f["mae"]  for f in fold_metrics]),
        "rmse": np.mean([f["rmse"] for f in fold_metrics]),
        "r2": float(np.mean([f["r2"] for f in fold_metrics if f["r2"] is not None]))
    }

    #Score
    r2_vals = [f["r2"] for f in fold_metrics if f["r2"] is not None]
    agg["r2_std"] = float(np.std(r2_vals)) if r2_vals else 0.0
    agg["score"] = agg["r2"] - agg["r2_std"]  # maximizar R2 estable


    results.append({
        "iteration":    iteration,
        "model":        model_name,
        "hp":           str(hp),
        "n_pca":        n_pca_comp,
        "do_scale":     do_scale,
        "dataset_size": train_len,
        **agg,
    })

    elapsed = time.time() - t0
    best_so_far = max((r["score"] for r in results), default=None)
    normalice = "normalizado" if do_scale else "no normalizado"
    print(
        f"  [{iteration}/{N_ITER}] {model_name} | PCA={n_pca_comp} | {normalice} | "
        f"MAE={agg['mae']:.4f} | RMSE={agg['rmse']:.4f} | "
        f"R2={agg['r2']:.4f}±{agg['r2_std']:.4f} | score={agg['score']:.4f} | "
        f"mellor={best_so_far:.4f} | {elapsed:.1f}s"
    )

print(f"\nBusca rematada en {time.time()-t0:.1f}s  |  Configuracións válidas: {len(results)}")

# selección do mellor modelo
results_df = pd.DataFrame(results).sort_values("score", ascending=False)
results_df.to_csv("materiais_faltantes.csv", index=False)

# imprimir o mellor resultado
best = results_df.iloc[0]
print("\nMellor configuración:")
print(f"  Modelo:          {best['model']}")
print(f"  Hiperparámetros: {best['hp']}")
print(f"  n_pca:           {best['n_pca']}")
print(f"  Normalizado:     {best['do_scale']}")
print(f"  MAE media:       {best['mae']:.4f}")
print(f"  RMSE media:      {best['rmse']:.4f}")
print(f"  R2 media:        {best['r2']:.4f}  ±  {best['r2_std']:.4f}")
print(f"  Score (R2-σ): {best['score']:.4f}")

# Adestrar co mellor modelo
best_model_name = best["model"]
best_hp         = ast.literal_eval(best["hp"])
best_n_pca      = int(best["n_pca"])
best_do_scale   = bool(best["do_scale"])

clf_final, pca_final, scaler_final, X_tst_t = train_model(X_train, Y_train, X_test, best_model_name, best_hp, best_n_pca, best_do_scale)

Y_pred_test = clf_final.predict(X_tst_t)
test_metrics = evaluate_predictions(Y_test, Y_pred_test, out_sets_test, ingredient_cols)

print("\nAvaliación no conxunto de test:")
print(f"  MAE:   {test_metrics['mae']:.4f}")
print(f"  RMSE:  {test_metrics['rmse']:.4f}")
print(f"  R2:    {test_metrics['r2']}")

# Adestrar o modelo final con TODOS os datos e gardalo
X_all = np.concatenate([X_train, X_test])
Y_all = np.concatenate([Y_train, Y_test])

clf_final_guardar, pca_gardar, scaler_gardar, _ = train_model( X_all, Y_all, None, best_model_name, best_hp, best_n_pca, best_do_scale)

# Gardar o modelo
joblib.dump({
    "clf":             clf_final_guardar,
    "pca":             pca_gardar,
    "scaler":          scaler_gardar,
    "ingredient_cols": ingredient_cols,
    "n_pca":           best_n_pca,
    "do_scale":        best_do_scale,
}, "materiais_faltantes.pkl")

print("\nModelo gardado en materiais_faltantes.pkl")
print("Resultados da busca gardados en materiais_faltantes.csv")