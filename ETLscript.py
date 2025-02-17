from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, regexp_replace, to_date
from pyspark.sql.types import IntegerType, DoubleType, StringType, StructType, StructField, DateType, FloatType
import logging
import re


# Configure logging
logging.basicConfig(filemode='a')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Add this line to set the logger's level

fh = logging.FileHandler('etl_pipeline.log')
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)


# Configuration
CONFIG = {
    "DATE_DEFAULT": "2025-01-01",
    "PRICE_TOLERANCE": 0.01,
    "VALID_CATEGORIES": ["Beauty", "Clothing", "Electronics"],
    "AGE_RANGE": {"min": 0, "max": 120},
    "SCHEMA": StructType([
        StructField("TransactionID", IntegerType(), False),
        StructField("TransactionDate", StringType(), True),
        StructField("CustomerID", StringType(), True),
        StructField("Gender", StringType(), True),
        StructField("Age", IntegerType(), True),
        StructField("ProductCategory", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("PricePerUnit", FloatType(), True),
        StructField("TotalPrice", FloatType(), True)
    ])
}

def contains_duplicates(df, column_name):
    """Returns True if the DataFrame contains duplicate values in the specified column."""
    try:
        transaction_counts = df.groupBy(column_name).count()
        duplicate_transactions = transaction_counts.filter(col("count") > 1)
        if duplicate_transactions.count() > 0:
            logger.warning(f"Found {duplicate_transactions.count()} duplicate(s) in {column_name}")
        return not(duplicate_transactions.count() == 0)
    except Exception as e:
        logger.error(f"Error checking duplicates: {str(e)}")
        raise

def clean_date(prev_day, current_day, next_day):
    """Checks if date column is in correct format, and if not, replaces it with the closest valid date."""
    expr = "^([0-9]{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$"
    if re.match(expr, current_day):
        return current_day
    else:
        if prev_day and re.match(expr, prev_day):
            return prev_day
        elif next_day and re.match(expr, next_day):
            return next_day
        return CONFIG["DATE_DEFAULT"]

def check_clean_date(df, spark):
    """Process and clean dates in the DataFrame."""
    try:
        rows = df.collect()
        cleaned_rows = []

        for i in range(len(rows)):
            prev_day = rows[i-1][1] if i > 0 else None
            next_day = rows[i+1][1] if i < len(rows)-1 else None
            current_row = rows[i]
            
            cleaned_row = (
                current_row[0],
                clean_date(prev_day, current_row[1], next_day),
                current_row[2],
                current_row[3],
                current_row[4],
                current_row[5],
                current_row[6],
                current_row[7],
                current_row[8]
            )
            cleaned_rows.append(cleaned_row)

        return spark.createDataFrame(cleaned_rows, df.schema)
    except Exception as e:
        logger.error(f"Error in date cleaning: {str(e)}")
        raise

def validate_age(df):
    """Validates age values are within reasonable range"""
    return df.withColumn("Age", 
        when((col("Age") < CONFIG["AGE_RANGE"]["min"]) | 
             (col("Age") > CONFIG["AGE_RANGE"]["max"]), None)
        .otherwise(col("Age")))

def validate_categories(df):
    """Validates product categories against allowed list"""
    return df.withColumn("ProductCategory",
        when(~col("ProductCategory").isin(CONFIG["VALID_CATEGORIES"]), None)
        .otherwise(col("ProductCategory")))

def validate_price_per_unit(row):
    """Validates and corrects price per unit based on quantity and total price."""
    try:
        if row[6] is None or row[7] is None or row[8] is None:
            if row[7] is None and row[6] is not None and row[8] is not None:
                quantity = float(row[6])
                total_amount = float(row[8])
                calculated_price = total_amount / quantity if quantity != 0 else 0
                return (
                    row[0], row[1], row[2], row[3], row[4], row[5], 
                    row[6], calculated_price, row[8]
                )
            return row

        quantity = float(row[6])
        price_per_unit = float(row[7]) if row[7] is not None else None
        total_amount = float(row[8])
        
        calculated_price = total_amount / quantity if quantity != 0 else 0
        
        if price_per_unit is not None:
            is_valid = abs(price_per_unit - calculated_price) <= CONFIG["PRICE_TOLERANCE"]
        else:
            is_valid = False
            
        if not is_valid:
            logger.info(f"Correcting price for TransactionID {row[0]}: {price_per_unit} -> {calculated_price}")
            
        return (
            row[0], row[1], row[2], row[3], row[4], row[5],
            row[6],
            calculated_price if not is_valid else price_per_unit,
            row[8]
        )

    except Exception as e:
        logger.error(f"Error processing row {row[0]}: {str(e)}")
        return row

def count_null_values_per_column(df):
    """Counts null values in each column of the DataFrame."""
    null_counts = {}
    for column in df.columns:
        count = df.filter(col(column).isNull()).count()
        null_counts[column] = count
        if count > 0:
            logger.info(f"Found {count} null values in column {column}")
    return null_counts

def run_etl_pipeline(spark, input_path):
    """Main ETL pipeline function"""
    try:
        logger.info("Starting ETL pipeline")
        
        # Load dataset
        df = spark.read.csv(input_path, schema=CONFIG["SCHEMA"], header=True)
        logger.info(f"Loaded {df.count()} rows from {input_path}")
        
        # Data validation
        contains_duplicates(df, "TransactionID")
        null_counts = count_null_values_per_column(df)
        
        # Data cleaning
        cleaned_df = check_clean_date(df, spark)
        cleaned_df = validate_age(cleaned_df)
        cleaned_df = validate_categories(cleaned_df)
        cleaned_df = cleaned_df.na.drop(subset=["TotalPrice"])
        cleaned_df = cleaned_df.rdd.map(validate_price_per_unit).toDF(df.schema)
        
        logger.info("ETL pipeline completed successfully")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("PySpark_ETL_Customer_Transactions") \
        .getOrCreate()
    
    try:
        # Run pipeline
        file_path = "corrupted_retail_sales_dataset.csv"
        result_df = run_etl_pipeline(spark, file_path)
        
        # Show results
        logger.info("Sample of cleaned data:")
        result_df.write.parquet('cleaned_data.parquet', mode='overwrite')
        
    except Exception as e:
        logger.error(f"Main ETL pipeline failed: {str(e)}")
        
    finally:
        spark.stop()