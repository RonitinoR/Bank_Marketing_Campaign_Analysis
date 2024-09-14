from loading import load_data
from data_tranformation import Transformation
from modeling import model

if __name__ == "__main__":
    # Define the filename for the data
    filename = "src/resources/data.csv"
    cleaned = "src/processed_goods/data_clean.csv"
    
    # Step 1: Load and clean the data
    print("Loading and cleaning data...")
    data_loader = load_data(filename)
    data_loader.save_cleaned_csv()  # This saves a cleaned CSV file

    # Step 2: Transform the data (e.g., label encoding, scaling, etc.)
    print("Transforming data...")
    transformer = Transformation(cleaned)
    train, test = transformer.TTSplit()  # Perform train-test split
    print("Train and Test data split successfully.")

    # Step 3: Model training and evaluation
    print("Training the model and evaluating...")
    Model = model(cleaned)
    results = Model.decisiontree()

    # Step 4: Output the results
    print("\nModel Evaluation Results:")
    print("Accuracy:", results['accuracy'])
    print("Classification Report:\n", results['classification_report'])
    print("Confusion Matrix:\n", results['confusion_matrix'])




