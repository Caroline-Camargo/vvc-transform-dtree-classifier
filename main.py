import os
import pandas as pd
from config import file_path, MODELS, MAX_SAMPLES_PER_CLASS, is_validation_curve
from logger import log_message
from grouping import *
from data_prep import prepare_balance_and_split_data
from feature_selection import recursive_feature_elimination_cv
from model_tuning import grid_search
from validation_curves import generate_validation_curves
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report

def run_all_experiments():
    GROUPINGS = ['area', 'max', 'orientation', 'aspect_ratio', 'all', 'single']

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].clip(lower=-1e18, upper=1e18)

    for grouping in GROUPINGS:
        if grouping == 'max': df['BlockGroup'] = df.apply(lambda r: determine_size_group(r['Width'], r['Height']), axis=1)
        elif grouping == 'area': df['BlockGroup'] = df.apply(lambda r: determine_area_group(r['Width'], r['Height']), axis=1)
        elif grouping == 'all': df['BlockGroup'] = df.apply(lambda r: determine_all_group(r['Width'], r['Height']), axis=1)
        elif grouping == 'orientation': df['BlockGroup'] = df.apply(lambda r: determine_orientation_group(r['Width'], r['Height']), axis=1)
        elif grouping == 'aspect_ratio': df['BlockGroup'] = df.apply(lambda r: determine_aspect_ratio_group(r['Width'], r['Height']), axis=1)
        elif grouping == 'single': df['BlockGroup'] = 'single'

        block_groups = sorted([g for g in df['BlockGroup'].unique() if g not in ["other", "invalid"]])

        for model in MODELS:
            for block_group in block_groups:
                try:
                    log_message(f"--- Processing ({grouping}) | Model {model} | Group {block_group} ---")

                    X_train, X_test, y_train, y_test = prepare_balance_and_split_data(df, model, block_group)

                    df_train_temp = pd.concat([X_train, y_train], axis=1)
                    df_train_sampled = pd.concat([
                        resample(g, replace=False, n_samples=min(len(g), MAX_SAMPLES_PER_CLASS))
                        for _, g in df_train_temp.groupby(y_train.name)
                    ])
                    X_train_sampled = df_train_sampled.drop(columns=[y_train.name])
                    y_train_sampled = df_train_sampled[y_train.name]

                    if is_validation_curve:
                        generate_validation_curves(X_train_sampled, y_train_sampled, grouping, model, block_group)
                        continue

                    selected_mask = recursive_feature_elimination_cv(X_train_sampled, y_train_sampled)
                    selected_features = X_train.columns[selected_mask]

                    best_params = grid_search(X_train_sampled[selected_features], y_train_sampled)

                    dt_final = DecisionTreeClassifier(random_state=42, **best_params)
                    dt_final.fit(X_train[selected_features], y_train)

                    final_preds = dt_final.predict(X_test[selected_features])
                    final_acc = accuracy_score(y_test, final_preds)
                    final_report = classification_report(y_test, final_preds, zero_division=0)

                    results_dir = f'results/{grouping}'
                    os.makedirs(results_dir, exist_ok=True)
                    filename = f"Result_{grouping}_model{model}_{block_group.replace('×', 'x')}.txt"
                    with open(os.path.join(results_dir, filename), 'w') as f:
                        f.write(final_report)

                    export_tree_to_cpp(dt_final, list(selected_features), sorted(y_train.unique()), f"tree_model_{grouping}_model{model}_{block_group.replace('x', '_')}", f'cpp_exports/{grouping}')

                except Exception as e:
                    log_message(f"✖ ERROR processing ({grouping}) Model {model}, Group {block_group}: {e}")


def export_tree_to_cpp(model, feature_names, class_names, function_name, output_dir):
    try:
        import DecisionTreeToCpp as to_cpp
        os.makedirs(output_dir, exist_ok=True)
        to_cpp.save_code(model, feature_names, class_names, function_name=function_name, output_dir=output_dir)
        log_message(f"✓ Model exported to C++ at: {output_dir}")
    except ImportError:
        log_message(f"⚠ DecisionTreeToCpp module not found. C++ export skipped.")
    except Exception as e:
        log_message(f"✖ Error exporting to C++: {e}")


if __name__ == "__main__":
    run_all_experiments()
