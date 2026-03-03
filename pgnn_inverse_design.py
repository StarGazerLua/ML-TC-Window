import os

# 1. Environment Setup
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from keras import layers, optimizers, callbacks, regularizers

# Random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Plotting style configuration
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F7CAC']
sns.set_palette(colors)
sns.set_style("whitegrid")


# ==================================================================================
# 1. Data Processor (Strict Zero-Leakage Version: Rigorous Train/Val/Test Split)
# ==================================================================================
class VO2DataProcessor:
    def __init__(self):
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        self.export_data = {}

    def load_data_from_excel(self, file_path):
        try:
            data = pd.read_excel(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found.")

        # Column name compatibility
        if 'Size' in data.columns:
            data = data.rename(columns={
                'Size': 'vo2_diameter', 'Thickness': 'pmma_thickness',
                '△E': 'delta_emissivity', '△T': 'delta_tnir'
            })
        self.export_data['original_data'] = data.copy()
        print(f"Data loaded: {len(data)} rows")
        return data

    def split_data(self, data, test_size=0.15, val_size=0.15, random_state=42):
        """
        [Important Fix 1]: Split the dataset into Train, Val, and Test partitions.
        The Test set is strictly isolated, and the Val set is used only for EarlyStopping.
        """
        print("Strictly splitting data into Train, Val, and Test partitions...")
        
        # 1. Slice out the Test set (strictly isolated)
        temp_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, shuffle=True)

        # 2. Slice out the Val set from the remaining data
        # Calculate relative ratio: val_size relative to the remaining data (1 - test_size)
        val_ratio = val_size / (1.0 - test_size)
        train_data, val_data = train_test_split(temp_data, test_size=val_ratio, random_state=random_state, shuffle=True)

        print(f"Split Result -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)

    def augment_train_data(self, train_data, augmentation_factor=5):
        """
        [Important Fix 2]: Removed subjective degradation formula, exclusively employing 
        Gaussian physical noise injection (Domain Randomization).
        """
        print(f"Augmenting TRAINING Data Only (Factor: {augmentation_factor})...")
        augmented_list = []
        
        for _, row in train_data.iterrows():
            augmented_list.append(row.to_dict())

        v_min, v_max = train_data['vo2_diameter'].min(), train_data['vo2_diameter'].max()
        p_min, p_max = train_data['pmma_thickness'].min(), train_data['pmma_thickness'].max()

        for _ in range(augmentation_factor * len(train_data)):
            base = train_data.sample(n=1).iloc[0]

            # Only inject physical measurement/manufacturing error noise (Gaussian perturbation)
            # Particle size fluctuation ~1.5nm, spacer thickness fluctuation ~0.005um
            v_new = np.clip(base['vo2_diameter'] + np.random.normal(0, 1), v_min, v_max)
            p_new = np.clip(base['pmma_thickness'] + np.random.normal(0, 0.005), p_min, p_max)

            # Spectral performance allows for minor observation noise
            e_new = np.clip(base['delta_emissivity'] + np.random.normal(0, 0.005), 0, 1.0)
            t_new = np.clip(base['delta_tnir'] + np.random.normal(0, 0.005), 0, 1.0)

            augmented_list.append({'vo2_diameter': v_new, 'pmma_thickness': p_new,
                                   'delta_emissivity': e_new, 'delta_tnir': t_new})

        return pd.DataFrame(augmented_list)

    def prepare_data(self, train_data, val_data, test_data):
        X_train = train_data[['vo2_diameter', 'pmma_thickness']].values
        y_train = train_data[['delta_emissivity', 'delta_tnir']].values

        X_val = val_data[['vo2_diameter', 'pmma_thickness']].values
        y_val = val_data[['delta_emissivity', 'delta_tnir']].values

        X_test = test_data[['vo2_diameter', 'pmma_thickness']].values
        y_test = test_data[['delta_emissivity', 'delta_tnir']].values

        # Fit only on the Train set (preventing distribution leakage)
        X_train_sc = self.scaler_features.fit_transform(X_train)
        y_train_sc = self.scaler_targets.fit_transform(y_train)

        # Transform Val and Test sets
        X_val_sc = self.scaler_features.transform(X_val)
        y_val_sc = self.scaler_targets.transform(y_val)

        X_test_sc = self.scaler_features.transform(X_test)
        y_test_sc = self.scaler_targets.transform(y_test)

        return X_train_sc, y_train_sc, X_val_sc, y_val_sc, X_test_sc, y_test_sc, X_train, y_train, X_test, y_test


# ==================================================================================
# 2. PGNN Model (Physical Constraints: Non-negativity + Boundary Limit [0, 1] + Smoothness)
# ==================================================================================
class PhysicsGuidedNN:
    def __init__(self, input_dim=2, output_dim=2, processor=None,
                 physics_weight=1e-3, bound_weight=0.5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.processor = processor
        self.physics_weight = physics_weight  # Smoothness weight (gradient norm)
        self.bound_weight = bound_weight      # Boundary and non-negativity constraint weight
        self.model = self._build_model()
        self.history = None

    class PGNN_KerasModel(tf.keras.Model):
        def __init__(self, input_dim, output_dim, physics_weight, bound_weight, target_scaler):
            super().__init__()
            self.physics_weight = physics_weight
            self.bound_weight = bound_weight

            # Extract inverse normalization parameters to apply constraints in the physical domain
            self.t_mean = tf.constant(target_scaler.mean_, dtype=tf.float32)
            self.t_scale = tf.constant(target_scaler.scale_, dtype=tf.float32)

            reg = regularizers.l2(1e-5)

            self.d1 = layers.Dense(64, activation='swish', kernel_initializer='he_normal', kernel_regularizer=reg)
            self.bn1 = layers.BatchNormalization()
            self.d2 = layers.Dense(128, activation='swish', kernel_initializer='he_normal', kernel_regularizer=reg)
            self.bn2 = layers.BatchNormalization()
            self.d3 = layers.Dense(64, activation='swish', kernel_initializer='he_normal', kernel_regularizer=reg)
            self.d4 = layers.Dense(32, activation='swish', kernel_initializer='he_normal')
            self.out = layers.Dense(output_dim, kernel_initializer='he_normal')

            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
            self.mse_tracker = tf.keras.metrics.Mean(name="mse")
            self.phy_tracker = tf.keras.metrics.Mean(name="phy")
            self.bound_tracker = tf.keras.metrics.Mean(name="bound")
            self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")

        def call(self, x):
            x = self.d1(x)
            x = self.bn1(x)
            x = self.d2(x)
            x = self.bn2(x)
            x = self.d3(x)
            x = self.d4(x)
            return self.out(x)

        def compute_loss(self, x, y, training=False):
            with tf.GradientTape() as t2:
                t2.watch(x)
                y_pred = self(x, training=training)

            mse = tf.reduce_mean(tf.square(y - y_pred))

            grads = t2.batch_jacobian(y_pred, x)
            grad_norm = tf.reduce_mean(tf.square(grads))

            # Physical Validity & Boundary Constraints
            y_pred_phys = y_pred * self.t_scale + self.t_mean

            loss_neg = tf.reduce_mean(tf.nn.relu(-y_pred_phys))
            loss_over = tf.reduce_mean(tf.nn.relu(y_pred_phys - 1.0))
            bound_loss = loss_neg + loss_over

            total_loss = mse + (self.physics_weight * grad_norm) + (self.bound_weight * bound_loss)

            return y_pred, total_loss, mse, grad_norm, bound_loss

        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                y_pred, loss, mse, phy, bound = self.compute_loss(x, y, training=True)

            grads = tape.gradient(loss, self.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.loss_tracker.update_state(loss)
            self.mse_tracker.update_state(mse)
            self.phy_tracker.update_state(phy)
            self.bound_tracker.update_state(bound)
            self.mae_tracker.update_state(y, y_pred)

            return {
                "loss": self.loss_tracker.result(),
                "mse": self.mse_tracker.result(),
                "phy": self.phy_tracker.result(),
                "bound": self.bound_tracker.result(),
                "mae": self.mae_tracker.result()
            }

        def test_step(self, data):
            x, y = data
            y_pred, loss, mse, phy, bound = self.compute_loss(x, y, training=False)

            self.loss_tracker.update_state(loss)
            self.mse_tracker.update_state(mse)
            self.phy_tracker.update_state(phy)
            self.bound_tracker.update_state(bound)
            self.mae_tracker.update_state(y, y_pred)

            return {
                "loss": self.loss_tracker.result(),
                "mse": self.mse_tracker.result(),
                "phy": self.phy_tracker.result(),
                "bound": self.bound_tracker.result(),
                "mae": self.mae_tracker.result()
            }

        @property
        def metrics(self):
            return [self.loss_tracker, self.mse_tracker, self.phy_tracker, self.bound_tracker, self.mae_tracker]

    def _build_model(self):
        model = self.PGNN_KerasModel(self.input_dim, self.output_dim, self.physics_weight, self.bound_weight,
                                     self.processor.scaler_targets)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005))
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=500):
        # Monitoring exclusively the Val set here, not the Test set
        es = callbacks.EarlyStopping(monitor='val_mse', patience=150, restore_best_weights=True)
        rlr = callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.5, patience=30, min_lr=1e-6)

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[es, rlr],
            verbose=1
        )

    def predict(self, X):
        return self.model.predict(X, verbose=0)


# ==================================================================================
# 3. Inverse Design & Visualization (Kept as original with translated comments)
# ==================================================================================
class InverseDesignModel:
    def __init__(self, forward_model, scaler_features, scaler_targets):
        self.model = forward_model
        self.s_feat = scaler_features
        self.s_targ = scaler_targets

    def find_optimal(self, target_e, target_t, n_cand=5000):
        v = np.random.uniform(30, 110, n_cand)
        p = np.random.uniform(1, 3, n_cand)
        X_cand = np.column_stack((v, p))

        X_cand_sc = self.s_feat.transform(X_cand)
        y_pred_sc = self.model.predict(X_cand_sc)
        y_pred = self.s_targ.inverse_transform(y_pred_sc)

        scores = 0.5 * (y_pred[:, 0] / target_e) + 0.5 * (y_pred[:, 1] / target_t)
        best_idx = np.argmax(scores)

        return {
            'best_design': X_cand[best_idx],
            'best_perf': y_pred[best_idx],
            'candidates': X_cand,
            'scores': scores,
            'predictions': y_pred
        }


class FeatureAnalyzer:
    def __init__(self, model, processor):
        self.model = model
        self.proc = processor

    def analyze_in_range(self, X_data, y_data, range_constraints=None):
        X_orig = self.proc.scaler_features.inverse_transform(X_data)
        y_orig = self.proc.scaler_targets.inverse_transform(y_data)

        mask = np.ones(len(X_orig), dtype=bool)
        if range_constraints:
            if 'vo2_min' in range_constraints: mask &= (X_orig[:, 0] >= range_constraints['vo2_min'])
            if 'vo2_max' in range_constraints: mask &= (X_orig[:, 0] <= range_constraints['vo2_max'])
            if 'pmma_min' in range_constraints: mask &= (X_orig[:, 1] >= range_constraints['pmma_min'])
            if 'pmma_max' in range_constraints: mask &= (X_orig[:, 1] <= range_constraints['pmma_max'])

        X_subset = X_data[mask]
        y_subset_orig = y_orig[mask]

        if len(X_subset) == 0:
            print("Warning: No samples found in the specified range!")
            return None

        print(f"Analyzing importance on {len(X_subset)} samples within range...")

        pred_base_sc = self.model.predict(X_subset, verbose=0)
        pred_base = self.proc.scaler_targets.inverse_transform(pred_base_sc)

        mse_base_e = mean_squared_error(y_subset_orig[:, 0], pred_base[:, 0])
        mse_base_t = mean_squared_error(y_subset_orig[:, 1], pred_base[:, 1])

        importances = {'Delta_E': [], 'Delta_T': []}
        feature_names = ['VO2 Size', 'Spacer Thickness']

        for i in range(2):
            X_perm = X_subset.copy()
            np.random.shuffle(X_perm[:, i])

            pred_perm_sc = self.model.predict(X_perm, verbose=0)
            pred_perm = self.proc.scaler_targets.inverse_transform(pred_perm_sc)

            mse_perm_e = mean_squared_error(y_subset_orig[:, 0], pred_perm[:, 0])
            mse_perm_t = mean_squared_error(y_subset_orig[:, 1], pred_perm[:, 1])

            imp_e = mse_perm_e - mse_base_e
            imp_t = mse_perm_t - mse_base_t

            importances['Delta_E'].append(max(0, imp_e))
            importances['Delta_T'].append(max(0, imp_t))

        total_e = sum(importances['Delta_E']) + 1e-10
        total_t = sum(importances['Delta_T']) + 1e-10

        results = {
            'features': feature_names,
            'E_scores': [x / total_e for x in importances['Delta_E']],
            'T_scores': [x / total_t for x in importances['Delta_T']],
            'range_info': range_constraints
        }
        return results


class VO2Visualizer:
    def __init__(self, processor, inverse_model):
        self.proc = processor
        self.inv = inverse_model

    def plot_history(self, h):
        def smooth_curve(points, factor=0.8):
            smoothed_points = []
            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points

        epochs = range(1, len(h.history['loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        l1, = ax1.plot(epochs, smooth_curve(h.history['loss']), label='Total Loss (Left)', color=colors[0])
        l2, = ax1.plot(epochs, smooth_curve(h.history['mse']), label='MSE (Left)', linestyle='--', color=colors[1])
        if 'val_mse' in h.history:
            l3, = ax1.plot(epochs, smooth_curve(h.history['val_mse']), label='Val MSE (Left)', linestyle=':', color=colors[2])

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss / MSE')
        ax1.set_title('Loss Convergence (Smoothed)')

        if 'phy' in h.history:
            ax1_right = ax1.twinx()
            l4, = ax1_right.plot(epochs, smooth_curve(h.history['phy']), label='Physics Loss (Gradient Norm)',
                                 color='gray', alpha=0.5)
            if 'bound' in h.history:
                l5, = ax1_right.plot(epochs, smooth_curve(h.history['bound']), label='Boundary Loss (0-1 Limit)',
                                     color='green', alpha=0.5, linestyle='-.')
                lines = [l1, l2, l3, l4, l5]
            else:
                lines = [l1, l2, l3, l4]

            ax1_right.set_ylabel('Physics Penalty', color='gray')
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
        else:
            ax1.legend()

        ax2.plot(epochs, smooth_curve(h.history['mae']), label='Train MAE', color=colors[3])
        if 'val_mae' in h.history:
            ax2.plot(epochs, smooth_curve(h.history['val_mae']), label='Val MAE', color=colors[4])
        ax2.set_title('MAE Convergence (Smoothed)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_surface_with_points(self, train_raw, test_raw, model):
        """Prediction Surface + Ground Truth Data Points (Optimized for Visibility)"""
        v = np.linspace(30, 110, 50)
        p = np.linspace(1.1, 3.1, 50)
        V, P = np.meshgrid(v, p)
        flat = np.column_stack((V.ravel(), P.ravel()))
        flat_sc = self.proc.scaler_features.transform(flat)
        pred_sc = model.predict(flat_sc, verbose=0)
        pred = self.proc.scaler_targets.inverse_transform(pred_sc)
        E_grid, T_grid = pred[:, 0].reshape(V.shape), pred[:, 1].reshape(V.shape)

        fig = plt.figure(figsize=(18, 8))

        # ================== Plot Delta E ==================
        ax1 = fig.add_subplot(121, projection='3d')

        # ================= Updated Code =================
        # 1. Surface body: keep highly transparent, remove default anti-aliasing black edges
        surf1 = ax1.plot_surface(V, P, E_grid, cmap='viridis', alpha=0.2, edgecolor='none', antialiased=True)

        # 2. Grid lines: introduce rstride and cstride to make the mesh sparser!
        # rstride=3, cstride=3 means drawing a line every 3 data points, instantly reducing grid density!
        # linewidth=0.6 makes the lines thin, alpha=0.5 makes them faint
        ax1.plot_wireframe(V, P, E_grid, color='white', alpha=0.5, linewidth=0.6, rstride=3, cstride=3)
        # ================================================

        # 2. De-emphasize training points
        ax1.scatter(train_raw['vo2_diameter'], train_raw['pmma_thickness'], train_raw['delta_emissivity'],
                    c='gray', s=30, alpha=0.5, label='Train (n=84)')

        # 3. Emphasize testing points (red with black edges, opaque)
        ax1.scatter(test_raw['vo2_diameter'], test_raw['pmma_thickness'], test_raw['delta_emissivity'],
                    c='#d7522a', s=70, alpha=1.0, label='Test (n=15)', zorder=10)

        # ================== Polish ax1 Title, Font, and Colorbar ==================
        # 1. Axis titles: use LaTeX syntax for subscripts and special characters, enlarge font
        ax1.set_xlabel('VO$_2$ Size (nm)', fontsize=12, labelpad=10)
        ax1.set_ylabel('Spacer thickness ($\mu$m)', fontsize=12, labelpad=10)
        ax1.set_zlabel('$\Delta$E', fontsize=12, labelpad=10)

        # 2. Enlarge axis tick labels (labelsize)
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # 3. Add vertical colorbar on the right
        # shrink controls length, aspect controls thickness, pad controls distance from main plot
        cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=15, pad=0.08)
        cbar1.ax.tick_params(labelsize=20)  # Enlarge colorbar font
        cbar1.set_label('$\Delta$E', fontsize=12)  # Add label to colorbar (optional)
        ax1.set_title('PGNN Prediction Surface vs Ground Truth (ΔE)')
        ax1.legend(loc='upper left')

        # ======= Adjust ax1 axis tick density =======
        # Y-axis (Spacer) tick every 0.5 (1.0, 1.5, 2.0...)
        ax1.yaxis.set_major_locator(MultipleLocator(0.5))

        # Z-axis (Delta E) tick every 0.1 (0.1, 0.2, 0.3...)
        ax1.zaxis.set_major_locator(MultipleLocator(0.1))

        # 4. Adjust viewing angle: lower elevation (elev), rotate azimuth (azim)
        ax1.view_init(elev=30, azim=-60)

        # ================== Plot Delta T ==================
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(V, P, T_grid, cmap='plasma', alpha=0.2, edgecolor='none', antialiased=True)

        # 2. Grid lines: introduce rstride and cstride to make the mesh sparser!
        ax2.plot_wireframe(V, P, T_grid, color='white', alpha=0.5, linewidth=0.6, rstride=3, cstride=3)
        # ================================================

        # 2. De-emphasize training points
        ax2.scatter(train_raw['vo2_diameter'], train_raw['pmma_thickness'], train_raw['delta_tnir'],
                    c='gray', s=30, alpha=0.5, label='Train (n=84)')

        # 3. Emphasize testing points
        ax2.scatter(test_raw['vo2_diameter'], test_raw['pmma_thickness'], test_raw['delta_tnir'],
                    c='#3282b8', s=70, alpha=1.0, label='Test (n=15)', zorder=10)

        # ================== Polish ax2 Title, Font, and Colorbar ==================
        ax2.set_xlabel('VO$_2$ Size (nm)', fontsize=20, labelpad=10)
        ax2.set_ylabel('Spacer thickness ($\mu$m)', fontsize=20, labelpad=10)
        ax2.set_zlabel('$\Delta$T', fontsize=20, labelpad=10)

        # 2. Enlarge axis tick labels
        ax2.tick_params(axis='both', which='major', labelsize=18)

        # 3. Add vertical colorbar
        cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=15, pad=0.08)
        cbar2.ax.tick_params(labelsize=20) 
        cbar2.set_label('$\Delta$T', fontsize=12) 
        ax2.set_title('PGNN Prediction Surface vs Ground Truth (ΔT)')
        ax2.legend(loc='upper left')

        # ======= Adjust ax2 axis tick density =======
        # Y-axis (Spacer) tick every 0.5
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.xaxis.set_major_locator(MultipleLocator(10))
        # Z-axis (Delta T) tick every 0.05
        ax2.zaxis.set_major_locator(MultipleLocator(0.05))

        # 4. Adjust viewing angle
        ax2.view_init(elev=30, azim=-60)

        plt.tight_layout()
        plt.show()

    def plot_full_optimization_analysis(self, res, target_e, target_t, test_raw):
        cand, scores = res['candidates'], res['scores']
        pred = res['predictions']
        best = res['best_design']

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        sc1 = axes[0, 0].scatter(cand[:, 0], cand[:, 1], c=scores, cmap='plasma', s=20, alpha=0.6)
        axes[0, 0].set_xlabel('VO$_2$ Size (nm)', fontsize=20)
        axes[0, 0].set_ylabel('Spacer thickness(um)', fontsize=20)
        axes[0, 0].set_title('Candidate Design Space', fontsize=20)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=20)
        # Enlarge colorbar font
        plt.colorbar(sc1, ax=axes[0, 0], label='Score')

        v_grid, p_grid = np.meshgrid(np.linspace(30, 110, 100), np.linspace(1.1, 3.1, 100))
        pts = np.column_stack([v_grid.ravel(), p_grid.ravel()])
        pts_sc = self.proc.scaler_features.transform(pts)
        pred_grid_sc = self.inv.model.predict(pts_sc, verbose=0)
        pred_grid = self.proc.scaler_targets.inverse_transform(pred_grid_sc)
        score_grid = (0.5 * pred_grid[:, 0] / target_e + 0.5 * pred_grid[:, 1] / target_t).reshape(v_grid.shape)

        # 1. Base coloring: lower levels to ~20, or add transparency so the background is not too glaring
        cf = axes[0, 1].contourf(v_grid, p_grid, score_grid, levels=20, cmap='plasma', alpha=0.85)

        # 2. [Core Addition] Surface contour lines: use contour to draw solid black contour lines
        # levels=10 means 10 main lines, colors='black' makes them stand out against the plasma background
        contours = axes[0, 1].contour(v_grid, p_grid, score_grid, levels=10, colors='black', linewidths=1.2, alpha=0.7)

        # 3. [Top Journal Bonus] Add specific numerical labels on the lines!
        # inline=True breaks the lines at the numbers, fmt='%.2f' keeps two decimal places
        axes[0, 1].clabel(contours, inline=True, fontsize=14, fmt='%.2f')
        axes[0, 1].set_xlabel('VO$_2$ Size (nm)', fontsize=20)
        axes[0, 1].set_ylabel('Spacer thickness (um)', fontsize=20)
        axes[0, 1].set_title('Comprehensive Score Contour', fontsize=20)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=20)
        plt.colorbar(cf, ax=axes[0, 1], label='Score')

        sc3 = axes[1, 0].scatter(pred[:, 0], pred[:, 1], c=scores, cmap='plasma', alpha=0.6)
        axes[1, 0].set_xlabel('Δ${E}$$_{LWIR}$', fontsize=20)
        axes[1, 0].set_ylabel('Δ${T}$$_{NIR}$', fontsize=20)
        axes[1, 0].set_title('Pareto Front Analysis')
        axes[1, 0].tick_params(axis='both', which='major', labelsize=20)
        plt.colorbar(sc3, ax=axes[1, 0], label='Score')

        fixed_p = best[1]
        v_test = np.linspace(30, 110, 100)
        test_pts = np.column_stack([v_test, np.full(100, fixed_p)])
        test_pts_sc = self.proc.scaler_features.transform(test_pts)
        sens_pred = self.proc.scaler_targets.inverse_transform(self.inv.model.predict(test_pts_sc, verbose=0))

        axes[1, 1].plot(v_test, sens_pred[:, 0], label='ΔE', lw=3)
        axes[1, 1].plot(v_test, sens_pred[:, 1], label='ΔT', lw=3)
        axes[1, 1].set_xlabel('VO$_2$ Size (nm)')
        axes[1, 1].set_title(f'Sensitivity at Spacer={fixed_p:.2f}um')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_prediction_accuracy(self, y_true, y_pred, r2_e, r2_t):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.scatter(y_true[:, 0], y_pred[:, 0], color=colors[0], alpha=0.7, s=50)
        min_val, max_val = min(y_true[:, 0].min(), y_pred[:, 0].min()), max(y_true[:, 0].max(), y_pred[:, 0].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_title(f'ΔE Accuracy (R²={r2_e:.4f})')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')

        ax2.scatter(y_true[:, 1], y_pred[:, 1], color=colors[1], alpha=0.7, s=50)
        min_val, max_val = min(y_true[:, 1].min(), y_pred[:, 1].min()), max(y_true[:, 1].max(), y_pred[:, 1].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax2.set_title(f'ΔT Accuracy (R²={r2_t:.4f})')
        ax2.set_xlabel('True Values')
        ax2.set_ylabel('Predicted Values')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, res):
        if res is None: return
        features = res['features']
        e_scores = res['E_scores']
        t_scores = res['T_scores']
        x = np.arange(len(features))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width / 2, e_scores, width, label='Impact on ΔE', color=colors[0], alpha=0.8)
        rects2 = ax.bar(x + width / 2, t_scores, width, label='Impact on ΔT', color=colors[1], alpha=0.8)

        ax.set_ylabel('Relative Importance Score')
        ax.set_title(f"Feature Importance Analysis\n(Range: VO2 30-100nm, Spacer 1.1-2.1um)")
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1%}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        plt.tight_layout()
        plt.show()


# ==================================================================================
# 3.5 Origin Data Exporter (Used to export CSV data for Origin plotting)
# ==================================================================================
class OriginDataExporter:
    def __init__(self, output_dir="origin_export_data"):
        self.out_dir = output_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        print(f"\n📂 Origin data export folder is ready: ./{self.out_dir}/")

    def export_history(self, history):
        """Export training process Loss and MAE curve data"""
        df = pd.DataFrame(history.history)
        df.index += 1  # Make Epoch start from 1
        df.to_csv(f"{self.out_dir}/1_training_history.csv", index_label="Epoch")

    def export_surface(self, proc, model, resolution=50):
        """Export 3D surface grid data (including X, Y, Z1, Z2)"""
        v = np.linspace(30, 110, resolution)
        p = np.linspace(1.1, 3.1, resolution)
        V, P = np.meshgrid(v, p)
        flat = np.column_stack((V.ravel(), P.ravel()))

        flat_sc = proc.scaler_features.transform(flat)
        pred = proc.scaler_targets.inverse_transform(model.predict(flat_sc, verbose=0))

        df = pd.DataFrame({
            'VO2_Size_X': flat[:, 0],
            'Spacer_Y': flat[:, 1],
            'Delta_E_Z1': pred[:, 0],
            'Delta_T_Z2': pred[:, 1]
        })
        df.to_csv(f"{self.out_dir}/2_3d_surface_grid.csv", index=False)

    def export_inverse_design(self, res):
        """Export inverse design scatter plot, contour plot, and Pareto front data"""
        df = pd.DataFrame({
            'VO2_Size_X': res['candidates'][:, 0],
            'Spacer_Y': res['candidates'][:, 1],
            'Delta_E_Pred': res['predictions'][:, 0],
            'Delta_T_Pred': res['predictions'][:, 1],
            'Comprehensive_Score': res['scores']
        })
        df.to_csv(f"{self.out_dir}/3_inverse_design_scatter.csv", index=False)

        # Save the best point separately
        best_df = pd.DataFrame({
            'Best_VO2': [res['best_design'][0]], 'Best_Spacer': [res['best_design'][1]],
            'Best_E': [res['best_perf'][0]], 'Best_T': [res['best_perf'][1]]
        })
        best_df.to_csv(f"{self.out_dir}/3_best_design_point.csv", index=False)

    def export_accuracy(self, y_train_true, y_train_pred, y_test_true, y_test_pred):
        """Export both Train and Test true and predicted values simultaneously for side-by-side comparison in Origin"""
        # Create a DataFrame containing both, distinguished by the 'Dataset' column
        n_train = len(y_train_true)
        n_test = len(y_test_true)

        dataset_labels = ['Train'] * n_train + ['Test'] * n_test

        true_e = np.concatenate([y_train_true[:, 0], y_test_true[:, 0]])
        pred_e = np.concatenate([y_train_pred[:, 0], y_test_pred[:, 0]])
        true_t = np.concatenate([y_train_true[:, 1], y_test_true[:, 1]])
        pred_t = np.concatenate([y_train_pred[:, 1], y_test_pred[:, 1]])

        df = pd.DataFrame({
            'Dataset': dataset_labels,
            'True_Delta_E': true_e,
            'Pred_Delta_E': pred_e,
            'True_Delta_T': true_t,
            'Pred_Delta_T': pred_t
        })
        df.to_csv(f"{self.out_dir}/4_accuracy_train_and_test.csv", index=False)

    def export_feature_importance(self, res):
        """Export feature importance bar chart data"""
        if res is None: return
        df = pd.DataFrame({
            'Feature': res['features'],
            'Delta_E_Importance': res['E_scores'],
            'Delta_T_Importance': res['T_scores']
        })
        df.to_csv(f"{self.out_dir}/5_feature_importance.csv", index=False)

    def export_sensitivity_1d(self, proc, model, best_spacer_thickness, resolution=200):
            """Export 1D sensitivity curve data showing performance vs VO2 size at optimal Spacer thickness"""
            # VO2 size range (30 nm to 110 nm)
            v_sizes = np.linspace(30, 110, resolution)
            
            # Fix Spacer thickness to the optimal value found by inverse design
            p_fixed = np.full_like(v_sizes, best_spacer_thickness)

            # Combine input features and make predictions
            X_1d = np.column_stack((v_sizes, p_fixed))
            X_1d_sc = proc.scaler_features.transform(X_1d)
            y_1d_pred_sc = model.predict(X_1d_sc, verbose=0)
            y_1d_pred = proc.scaler_targets.inverse_transform(y_1d_pred_sc)

            # Save as CSV
            df = pd.DataFrame({
                'VO2_Size_X': v_sizes,
                f'Fixed_Spacer_Y_Ignore': p_fixed,  # For record only, can be ignored during plotting
                'Delta_E_Y1': y_1d_pred[:, 0],
                'Delta_T_Y2': y_1d_pred[:, 1]
            })
            df.to_csv(f"{self.out_dir}/6_sensitivity_1d_curve.csv", index=False)


# ==================================================================================
# 4. Main
# ==================================================================================
def main():
    print("=== VO2 Physics-Guided Inverse Design (GPT Bug Fixes Applied) ===")

    proc = VO2DataProcessor()
    try:
        raw_data = proc.load_data_from_excel('dataset.xlsx')
    except Exception as e:
        print(f"Error: {e}")
        return

    # [Fix 1 implementation]: Three completely independent sets are obtained here
    train_df, val_df, test_df = proc.split_data(raw_data, test_size=0.15, val_size=0.15)

    # Augment only the training set
    train_aug = proc.augment_train_data(train_df, augmentation_factor=5)

    # [Fix 1 implementation]: Add val data flow during Scale processing
    X_train, y_train, X_val, y_val, X_test, y_test, _, _, _, _ = proc.prepare_data(train_aug, val_df, test_df)

    print("\nTraining Model...")
    pgnn = PhysicsGuidedNN(processor=proc, physics_weight=1e-3, bound_weight=0.5)

    # [Fix 1 implementation]: Pass in X_val, y_val for early stopping
    pgnn.train(X_train, y_train, X_val, y_val, epochs=300)

    # Inverse Design (Kept unchanged)
    inv_model = InverseDesignModel(pgnn.model, proc.scaler_features, proc.scaler_targets)
    target_e = train_df['delta_emissivity'].max()
    target_t = train_df['delta_tnir'].max()
    res = inv_model.find_optimal(target_e, target_t)

    print("\n=== Optimal Design ===")
    print(f"VO2 Size: {res['best_design'][0]:.2f} nm")
    print(f"Spacer:   {res['best_design'][1]:.2f} um")
    print(f"Pred ΔE:  {res['best_perf'][0]:.3f}")
    print(f"Pred ΔT:  {res['best_perf'][1]:.3f}")

    # Visualization
    vis = VO2Visualizer(proc, inv_model)
    vis.plot_history(pgnn.history)
    vis.plot_surface_with_points(train_df, test_df, pgnn.model)
    vis.plot_full_optimization_analysis(res, target_e, target_t, test_df)

    # [Fix 1 implementation]: This is the first and only use of the Test set
    print("\n=== Strict Test Set Evaluation (Unseen Data) ===")
    test_pred_sc = pgnn.predict(X_test)
    test_pred = proc.scaler_targets.inverse_transform(test_pred_sc)
    test_true = test_df[['delta_emissivity', 'delta_tnir']].values

    # === [Added/Fixed] Get training set predictions for background plotting ===
    # The predictions are standardized values, need inverse_transform to restore to physical values
    train_pred_sc = pgnn.model.predict(X_train, verbose=0)
    train_pred = proc.scaler_targets.inverse_transform(train_pred_sc)

    # [Core Fix] y_train is also standardized, must inverse_transform to restore physical values!
    train_true = proc.scaler_targets.inverse_transform(y_train)

    r2_e = r2_score(test_true[:, 0], test_pred[:, 0])
    r2_t = r2_score(test_true[:, 1], test_pred[:, 1])
    print(f"Test Set R² (ΔE): {r2_e:.4f}")
    print(f"Test Set R² (ΔT): {r2_t:.4f}")

    vis.plot_prediction_accuracy(test_true, test_pred, r2_e, r2_t)

    print("\n=== Range-Specific Feature Importance Analysis ===")
    analyzer = FeatureAnalyzer(pgnn.model, proc)
    constraints = {'vo2_min': 30, 'vo2_max': 100, 'pmma_min': 1.1, 'pmma_max': 2.1}
    X_all = np.vstack((X_train, X_val, X_test))
    y_all = np.vstack((y_train, y_val, y_test))

    importance_res = analyzer.analyze_in_range(X_all, y_all, constraints)
    if importance_res:
        print("\nImportance Scores (Normalized):")
        print(f"For ΔE: VO2 Size = {importance_res['E_scores'][0]:.4f}, Spacer = {importance_res['E_scores'][1]:.4f}")
        print(f"For ΔT: VO2 Size = {importance_res['T_scores'][0]:.4f}, Spacer = {importance_res['T_scores'][1]:.4f}")
        vis.plot_feature_importance(importance_res)

    print("\nSystem execution completed successfully. Data Leakage effectively resolved.")
    
    # ======== Add this at the end of the main() function ========
    print("\n=== Starting Origin plot data export ===")
    exporter = OriginDataExporter()
    exporter.export_history(pgnn.history)
    exporter.export_surface(proc, pgnn.model)
    exporter.export_inverse_design(res)
    exporter.export_accuracy(train_true, train_pred, test_true, test_pred)
    exporter.export_feature_importance(importance_res)
    # ==================================================
    print("✅ All plotting data successfully exported to 'origin_export_data' folder!")

    print("\nSystem execution completed successfully. Data Leakage effectively resolved.")


if __name__ == "__main__":
    main()
