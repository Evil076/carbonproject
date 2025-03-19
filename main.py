import os
from scripts import prepare_data, train_model, predict

def main():
    # Prepare the data
    prepare_data.main()

    # Train the model
    train_model.main()

    # Make predictions
    predict.main()

if __name__ == "__main__":
    main()