import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config
import data_generator
import preprocessing
import modeling

class TestMLPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up temporary filenames for testing data generator and preprocessing."""
        cls.test_clients_path = os.path.join(config.DATA_DIR, "test_clients.csv")
        cls.test_properties_path = os.path.join(config.DATA_DIR, "test_properties.csv")
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test files if they exist."""
        for path in [cls.test_clients_path, cls.test_properties_path]:
            if os.path.exists(path):
                os.remove(path)
                
    def test_data_generation(self):
        """Tests if the data generator runs successfully and creates non-empty dataframes."""
        df_c, df_p = data_generator.generate_datasets(
            num_clients=100, seed=42, write_to_disk=True,
            clients_path=self.test_clients_path,
            properties_path=self.test_properties_path
        )
        
        self.assertIsNotNone(df_c)
        self.assertIsNotNone(df_p)
        self.assertGreater(len(df_c), 100) # accounts for duplicates
        self.assertEqual(len(df_c), len(df_p))
        self.assertIn(config.COL_CLIENT_ID, df_c.columns)
        self.assertIn(config.COL_PURCHASE_PRICE, df_p.columns)
        
    def test_preprocessing(self):
        """Tests load_raw_data, clean_data, and feature transformations."""
        # Ensure raw data is generated
        if not os.path.exists(self.test_clients_path):
            self.test_data_generation()
            
        # 1. Test Load and Join
        df_merged = preprocessing.load_raw_data(
            clients_path=self.test_clients_path, 
            properties_path=self.test_properties_path
        )
        self.assertGreater(len(df_merged), 0)
        self.assertIn(config.COL_CLIENT_ID, df_merged.columns)
        self.assertIn(config.COL_PURCHASE_PRICE, df_merged.columns)
        
        # 2. Test Cleaning
        df_cleaned = preprocessing.clean_data(df_merged)
        self.assertFalse(df_cleaned.duplicated().any())
        self.assertNotIn(config.COL_DOB, df_cleaned.columns)
        self.assertIn(config.COL_AGE, df_cleaned.columns)
        self.assertFalse(df_cleaned.isnull().any().any())
        
        # 3. Test Feature Transforms
        X, feature_names, scaler, encoder = preprocessing.fit_transform_features(df_cleaned, is_training=True)
        self.assertEqual(len(X), len(df_cleaned))
        self.assertGreater(X.shape[1], len(config.NUMERICAL_COLS))
        self.assertIn(config.COL_AGE, feature_names)
        self.assertIsNotNone(scaler)
        self.assertIsNotNone(encoder)
        
    def test_modeling(self):
        """Tests metrics calculation, optimal cluster fit, and PCA."""
        # Setup dummy dataset
        np.random.seed(42)
        X = np.random.randn(50, 15) # 50 samples, 15 features
        
        # 1. Metrics evaluation
        metrics_df = modeling.run_clustering_metrics(X, max_k=4)
        self.assertEqual(len(metrics_df), 3) # for k=2,3,4
        self.assertListEqual(list(metrics_df["k"]), [2, 3, 4])
        self.assertTrue("silhouette" in metrics_df.columns)
        self.assertTrue("davies_bouldin" in metrics_df.columns)
        
        # 2. Optimal selection
        opt_k = modeling.determine_optimal_k(metrics_df)
        self.assertIn(opt_k, [2, 3, 4])
        
        # 3. Model training
        kmeans, labels = modeling.fit_kmeans_model(X, opt_k)
        self.assertEqual(len(labels), len(X))
        self.assertEqual(len(kmeans.cluster_centers_), opt_k)
        
        # 4. PCA
        pca, X_pca = modeling.apply_pca(X, n_components=3)
        self.assertEqual(X_pca.shape, (50, 3))
        
if __name__ == "__main__":
    unittest.main()
