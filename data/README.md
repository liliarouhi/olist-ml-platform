# Data

**Source:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(Kaggle): about 100,000 real orders placed between 2016 and 2018, in 9 CSV files (about 120 MB).

The raw files are not stored in this repository because of their size; they are downloaded from
the original source, and everything derived from them is rebuilt by `scripts/run_all.sh`.

## Setup
1. Download the dataset from the link above (a free Kaggle account is required).
2. Extract the archive into `data/raw/`.

Expected files in `data/raw/`:

| File | Rows |
|---|---:|
| `olist_customers_dataset.csv` | 99,441 |
| `olist_geolocation_dataset.csv` | 1,000,163 |
| `olist_orders_dataset.csv` | 99,441 |
| `olist_order_items_dataset.csv` | 112,650 |
| `olist_order_payments_dataset.csv` | 103,886 |
| `olist_order_reviews_dataset.csv` | 99,224 |
| `olist_products_dataset.csv` | 32,951 |
| `olist_sellers_dataset.csv` | 3,095 |
| `product_category_name_translation.csv` | 71 |

## Generated data
`data/processed/order_features.csv` (the model dataset) is created by `scripts/eda.py`.
Neither folder is tracked by git.
